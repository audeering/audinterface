from __future__ import annotations

import collections
from collections.abc import Callable
from collections.abc import Sequence
import inspect
import itertools

import numpy as np
import pandas as pd

import audeer
import audformat

from audinterface.core import utils


def identity(signal, sampling_rate, starts, ends) -> list[np.ndarray]:
    r"""Default processing function.

    This function is used,
    when ``ProcessWithContext`` is instantiated
    with ``process_func=None``.
    It returns all given segments.

    Args:
        signal: signal
        sampling_rate: sampling rate in Hz
        starts: start indices
        ends: end indices

    Returns:
        list of segments

    """
    return [signal[:, start:end] for start, end in zip(starts, ends)]


class ProcessWithContext:
    r"""Alternate processing interface that provides signal context.

    In contrast to :class:`Process` this interface does not look at segments
    in isolation, but passes the complete signal together with a list of
    segments to the processing function. By doing so, it becomes possible to
    process segments in context, e.g. by taking into account surrounding
    signal values or other segments.

    Args:
        process_func: processing function,
            which expects four positional arguments:

            * ``signal``
            * ``sampling_rate``
            * ``starts`` sequence with start indices
            * ``ends`` sequence with end indices

            and any number of additional keyword arguments
            (see ``process_func_args``).
            There are the following special arguments:
            ``'idx'``, ``'file'``, ``'root'``.
            If expected by the function,
            but not specified in
            ``process_func_args``,
            they will be replaced with:
            a running index,
            the currently processed file,
            the root folder.
            The function must return a sequence of results for every segment
        process_func_args: (keyword) arguments passed on to the processing
            function
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal
        resample: if ``True`` enforces given sampling rate by resampling
        channels: channel selection, see :func:`audresample.remix`
        mixdown: apply mono mix-down on selection
        verbose: show debug messages

    Raises:
        ValueError: if ``resample = True``, but ``sampling_rate = None``

    Examples:
        >>> def running_mean(signal, sampling_rate, starts, ends):
        ...     means_per_segment = [
        ...         signal[:, start:end].mean() for start, end in zip(starts, ends)
        ...     ]
        ...     cumsum = np.cumsum(np.pad(means_per_segment, 1))
        ...     return (cumsum[2:] - cumsum[:-2]) / float(2)
        >>> interface = ProcessWithContext(process_func=running_mean)
        >>> signal = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        >>> sampling_rate = 3
        >>> starts = [0, sampling_rate]
        >>> ends = [sampling_rate, 2 * sampling_rate]
        >>> interface(signal, sampling_rate, starts, ends)
        array([2., 1.])
        >>> # Apply interface on an audformat conform index of a dataframe
        >>> import audb
        >>> db = audb.load(
        ...     "emodb",
        ...     version="1.3.0",
        ...     media="wav/03a01Fa.wav",
        ...     full_path=False,
        ...     verbose=False,
        ... )
        >>> files = list(db.files) * 3
        >>> starts = [0, 0.1, 0.2]
        >>> ends = [0.5, 0.6, 0.7]
        >>> index = audformat.segmented_index(files, starts, ends)
        >>> interface.process_index(index, root=db.root)
        file             start                   end
        wav/03a01Fa.wav  0 days 00:00:00         0 days 00:00:00.500000   -0.000261
                         0 days 00:00:00.100000  0 days 00:00:00.600000   -0.000199
                         0 days 00:00:00.200000  0 days 00:00:00.700000   -0.000111
        dtype: float32

    """  # noqa: E501

    def __init__(
        self,
        *,
        process_func: Callable[..., Sequence[object]] | None = None,
        process_func_args: dict[str, object] | None = None,
        sampling_rate: int | None = None,
        resample: bool = False,
        channels: int | Sequence[int] | None = None,
        mixdown: bool = False,
        verbose: bool = False,
    ):
        if channels is not None:
            channels = audeer.to_list(channels)

        if resample and sampling_rate is None:
            raise ValueError("sampling_rate has to be provided for resample = True.")

        process_func = process_func or identity
        signature = inspect.signature(process_func)
        self._process_func_signature = signature.parameters
        r"""Arguments present in processing function."""

        self.channels = channels
        r"""Channel selection."""

        self.mixdown = mixdown
        r"""Mono mixdown."""

        self.process_func = process_func
        r"""Process function."""

        self.process_func_args = process_func_args or {}
        r"""Additional keyword arguments to processing function."""

        self.resample = resample
        r"""Resample signal."""

        self.sampling_rate = sampling_rate
        r"""Sampling rate in Hz."""

        self.verbose = verbose
        r"""Show debug messages."""

    def process_index(
        self,
        index: pd.Index,
        *,
        root: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.Series:
        r"""Process from a segmented index conform to audformat_.

        Args:
            index: index with segment information
            root: root folder to expand relative file paths
            process_func_args: (keyword) arguments passed on
                to the processing function.
                They will temporarily overwrite
                the ones stored in
                :attr:`audinterface.ProcessWithContext.process_func_args`

        Returns:
            Series with processed segments conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if sequence returned by ``process_func``
                does not match length of ``index``

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        utils.assert_index(index)

        index = audformat.utils.to_segmented_index(index)

        if len(index) == 0:
            return pd.Series([], index=index, dtype=object)

        files = index.levels[0]
        ys = []

        with audeer.progress_bar(
            files,
            total=len(files),
            disable=not self.verbose,
            maximum_refresh_time=1,
        ) as pbar:
            for idx, file in enumerate(pbar):
                if self.verbose:  # pragma: no cover
                    desc = audeer.format_display_message(file, pbar=True)
                    pbar.set_description(desc, refresh=True)

                mask = index.isin([file], 0)
                select = index[mask].droplevel(0)

                signal, sampling_rate = utils.read_audio(file, root=root)
                y = self._process_signal_from_index(
                    signal,
                    sampling_rate,
                    select,
                    idx=idx,
                    root=root,
                    file=file,
                    process_func_args=process_func_args,
                )

                ys.append(y)

        y = list(itertools.chain.from_iterable([x for x in ys]))
        y = pd.Series(y, index)

        return y

    def _process_signal_from_index(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        index: pd.Index,
        *,
        idx: int = 0,
        root: str | None = None,
        file: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> object:
        starts_i, ends_i = utils.segments_to_indices(
            signal,
            sampling_rate,
            index,
        )
        y = self._call(
            signal,
            sampling_rate,
            starts_i,
            ends_i,
            idx=idx,
            root=root,
            file=file,
            process_func_args=process_func_args,
        )
        if not isinstance(y, collections.abc.Iterable) or len(y) != len(index):
            raise RuntimeError(
                "process_func has to return a sequence of results, "
                f"matching the length {len(index)} of the index. "
            )

        return y

    def process_signal_from_index(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        index: pd.Index,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.Series:
        r"""Split a signal into segments and process each segment.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            index: a :class:`pandas.MultiIndex` with two levels
                named `start` and `end` that hold start and end
                positions as :class:`pandas.Timedelta` objects.
                See also :func:`audinterface.utils.signal_index`
            process_func_args: (keyword) arguments passed on
                to the processing function.
                They will temporarily overwrite
                the ones stored in
                :attr:`audinterface.ProcessWithContext.process_func_args`

        Returns:
            Series with processed segments conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if sequence returned by ``process_func``
                does not match length of ``index``
            ValueError: if index contains duplicates

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        utils.assert_index(index)

        if len(index) == 0:
            return pd.Series([], index=index, dtype=object)

        y = self._process_signal_from_index(
            signal,
            sampling_rate,
            index,
            process_func_args=process_func_args,
        )
        y = pd.Series(y, index)

        return y

    def _call(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        starts: Sequence[int],
        ends: Sequence[int],
        *,
        idx: int = 0,
        root: str | None = None,
        file: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> object:
        r"""Call processing function, possibly pass special args."""
        signal, sampling_rate = utils.preprocess_signal(
            signal,
            sampling_rate,
            self.sampling_rate,
            self.resample,
            self.channels,
            self.mixdown,
        )

        process_func_args = process_func_args or self.process_func_args
        special_args = {}
        for key, value in [
            ("idx", idx),
            ("root", root),
            ("file", file),
        ]:
            if key in self._process_func_signature and key not in process_func_args:
                special_args[key] = value

        return self.process_func(
            signal,
            sampling_rate,
            starts,
            ends,
            **special_args,
            **process_func_args,
        )

    def __call__(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        starts: Sequence[int],
        ends: Sequence[int],
    ) -> object:
        r"""Apply processing to signal.

        This function processes the signal **without** transforming the output
        into a :class:`pd.Series`. Instead, it will return the raw processed
        signal. However, if channel selection, mixdown and/or resampling
        is enabled, the signal will be first remixed and resampled if the
        input sampling rate does not fit the expected sampling rate.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            starts: sequence with start values
            ends: sequence with end values

        Returns:
            Processed signal

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        """
        return self._call(signal, sampling_rate, starts, ends)
