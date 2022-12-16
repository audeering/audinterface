import collections
import inspect
import typing
import warnings

import numpy as np
import pandas as pd

import audeer
import audformat

from audinterface.core import utils


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
        ...         signal[:, start:end].mean()
        ...         for start, end in zip(starts, ends)
        ...     ]
        ...     cumsum = np.cumsum(np.pad(means_per_segment, 1))
        ...     return (cumsum[2:] - cumsum[:-2]) / float(2)
        >>> interface = ProcessWithContext(process_func=running_mean)
        >>> signal = np.array([1., 2., 3., 1., 2., 3.])
        >>> sampling_rate = 3
        >>> starts = [0, sampling_rate]
        >>> ends = [sampling_rate, 2 * sampling_rate]
        >>> interface(signal, sampling_rate, starts, ends)
        array([2., 1.])
        >>> # Apply interface on an audformat conform index of a dataframe
        >>> import audb
        >>> db = audb.load(
        ...     'emodb',
        ...     version='1.2.0',
        ...     media='wav/03a01Fa.wav',
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
            process_func: typing.Callable[
                ...,
                typing.Sequence[typing.Any]
            ] = None,
            process_func_args: typing.Dict[str, typing.Any] = None,
            sampling_rate: int = None,
            resample: bool = False,
            channels: typing.Union[int, typing.Sequence[int]] = None,
            mixdown: bool = False,
            verbose: bool = False,
            **kwargs,
    ):
        if channels is not None:
            channels = audeer.to_list(channels)

        if process_func is None:
            def process_func(signal, _, starts, ends):
                return [
                    signal[:, start:end] for start, end in zip(starts, ends)
                ]

        process_func_args = process_func_args or {}
        if kwargs:
            warnings.warn(
                utils.kwargs_deprecation_warning,
                category=UserWarning,
                stacklevel=2,
            )
            for key, value in kwargs.items():
                process_func_args[key] = value

        if resample and sampling_rate is None:
            raise ValueError(
                'sampling_rate has to be provided for resample = True.'
            )

        # figure out if special arguments
        # to pass to the processing function
        signature = inspect.signature(process_func)
        self._process_func_special_args = {
            'idx': False,
            'root': False,
            'file': False,
        }
        for key in self._process_func_special_args:
            if (
                    key in signature.parameters
                    and key not in process_func_args
            ):
                self._process_func_special_args[key] = True

        self.channels = channels
        r"""Channel selection."""

        self.mixdown = mixdown
        r"""Mono mixdown."""

        self.process_func = process_func
        r"""Process function."""

        self.process_func_args = process_func_args
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
            root: str = None,
    ) -> pd.Series:
        r"""Process from a segmented index conform to audformat_.

        Args:
            index: index with segment information
            root: root folder to expand relative file paths

        Returns:
            Series with processed segments conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if sequence returned by ``process_func``
                does not match length of ``index``

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        index = audformat.utils.to_segmented_index(index)

        if index.empty:
            return pd.Series(index=index, dtype=object)

        files = index.levels[0]
        ys = [None] * len(files)

        with audeer.progress_bar(
                files,
                total=len(files),
                disable=not self.verbose,
        ) as pbar:
            for idx, file in enumerate(pbar):
                desc = audeer.format_display_message(file, pbar=True)
                pbar.set_description(desc, refresh=True)
                mask = index.isin([file], 0)
                select = index[mask].droplevel(0)
                signal, sampling_rate = utils.read_audio(file, root=root)
                ys[idx] = pd.Series(
                    self._process_signal_from_index(
                        signal,
                        sampling_rate,
                        select,
                        idx=idx,
                        root=root,
                        file=file,
                    ).values,
                    index=index[mask],
                )

        return pd.concat(ys)

    def _process_signal_from_index(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            index: pd.Index,
            *,
            idx: int = 0,
            root: str = None,
            file: str = None,
    ) -> pd.Series:

        utils.assert_index(index)

        if len(index) == 0:
            y = pd.Series([], index=index, dtype=object)
        else:
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
            )
            if (
                    not isinstance(y, collections.abc.Iterable)
                    or len(y) != len(index)
            ):
                raise RuntimeError(
                    'process_func has to return a sequence of results, '
                    f'matching the length {len(index)} of the index. '
                )
            y = pd.Series(y, index=index)

        return y

    def process_signal_from_index(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            index: pd.Index,
    ) -> pd.Series:
        r"""Split a signal into segments and process each segment.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            index: a :class:`pandas.MultiIndex` with two levels
                named `start` and `end` that hold start and end
                positions as :class:`pandas.Timedelta` objects.
                See also :func:`audinterface.utils.signal_index`

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
        return self._process_signal_from_index(
            signal,
            sampling_rate,
            index,
        )

    def _call(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        starts: typing.Sequence[int],
        ends: typing.Sequence[int],
        *,
        idx: int = 0,
        root: str = None,
        file: str = None,
    ) -> typing.Any:
        r"""Call processing function, possibly pass special args."""

        signal, sampling_rate = utils.preprocess_signal(
            signal,
            sampling_rate,
            self.sampling_rate,
            self.resample,
            self.channels,
            self.mixdown,
        )

        special_args = {}
        for key, value in [
            ('idx', idx),
            ('root', root),
            ('file', file),
        ]:
            if self._process_func_special_args[key]:
                special_args[key] = value

        return self.process_func(
            signal,
            sampling_rate,
            starts,
            ends,
            **special_args,
            **self.process_func_args,
        )

    def __call__(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        starts: typing.Sequence[int],
        ends: typing.Sequence[int],
    ) -> typing.Any:
        r"""Apply processing to signal.

        This function processes the signal **without** transforming the output
        into a :class:`pd.Series`. Instead it will return the raw processed
        signal. However, if channel selection, mixdown and/or resampling
        is enabled, the signal will be first remixed and resampled if the
        input sampling rate does not fit the expected sampling rate.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz

        Returns:
            Processed signal

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        """
        return self._call(signal, sampling_rate, starts, ends)
