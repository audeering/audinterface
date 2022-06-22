import collections
import errno
import os
import typing
import warnings

import numpy as np
import pandas as pd

import audeer
import audformat

from audinterface.core import utils
from audinterface.core.segment import Segment
from audinterface.core.typing import (
    Timestamp,
    Timestamps,
)


class Process:
    r"""Processing interface.

    Args:
        process_func: processing function,
            which expects the two positional arguments ``signal``
            and ``sampling_rate``
            and any number of additional keyword arguments
        process_func_args: (keyword) arguments passed on to the processing
            function
        process_func_is_mono: if set to ``True`` and the input signal
            has multiple channels, ``process_func`` will be applied to
            every channel individually
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal
        resample: if ``True`` enforces given sampling rate by resampling
        channels: channel selection, see :func:`audresample.remix`
        mixdown: apply mono mix-down on selection
        segment: when a :class:`audinterface.Segment` object is provided,
            it will be used to find a segmentation of the input signal.
            Afterwards processing is applied to each segment
        keep_nat: if the end of segment is set to ``NaT`` do not replace
            with file duration in the result
        min_signal_length: minimum signal length in samples
            required by ``process_func``.
            If provided signal is shorter,
            it will be zero padded at the end
        max_signal_length: maximum signal length in samples
            required by ``process_func``.
            If provided signal is longer,
            it will be cut at the end
        num_workers: number of parallel jobs or 1 for sequential
            processing. If ``None`` will be set to the number of
            processors on the machine multiplied by 5 in case of
            multithreading and number of processors in case of
            multiprocessing
        multiprocessing: use multiprocessing instead of multithreading
        verbose: show debug messages

    Raises:
        ValueError: if ``resample = True``, but ``sampling_rate = None``

    Example:
        >>> def mean(signal, sampling_rate):
        ...     return signal.mean()
        >>> interface = Process(process_func=mean)
        >>> signal = np.array([1., 2., 3.])
        >>> interface(signal, sampling_rate=3)
        2.0
        >>> interface.process_signal(signal, sampling_rate=3)
        start   end
        0 days  0 days 00:00:01   2.0
        dtype: float64
        >>> import audb
        >>> db = audb.load(
        ...     'emodb',
        ...     version='1.2.0',
        ...     media='wav/03a01Fa.wav',
        ...     full_path=False,
        ...     verbose=False,
        ... )
        >>> index = db['emotion'].index
        >>> interface.process_index(index, root=db.root)
        file             start   end
        wav/03a01Fa.wav  0 days  0 days 00:00:01.898250    -0.000311
        dtype: float32

    """
    def __init__(
            self,
            *,
            process_func: typing.Callable[..., typing.Any] = None,
            process_func_args: typing.Dict[str, typing.Any] = None,
            process_func_is_mono: bool = False,
            sampling_rate: int = None,
            resample: bool = False,
            channels: typing.Union[int, typing.Sequence[int]] = None,
            mixdown: bool = False,
            segment: Segment = None,
            keep_nat: bool = False,
            min_signal_length: int = None,
            max_signal_length: int = None,
            num_workers: typing.Optional[int] = 1,
            multiprocessing: bool = False,
            verbose: bool = False,
            **kwargs,
    ):
        if resample and sampling_rate is None:
            raise ValueError(
                'sampling_rate has to be provided for resample = True.'
            )
        self.sampling_rate = sampling_rate
        r"""Sampling rate in Hz."""
        self.resample = resample
        r"""Resample signal."""
        self.channels = None if channels is None else audeer.to_list(channels)
        r"""Channel selection."""
        self.mixdown = mixdown
        r"""Mono mixdown."""
        self.segment = segment
        r"""Segmentation object."""
        self.keep_nat = keep_nat
        r"""Keep NaT in results."""
        self.min_signal_length = min_signal_length
        r"""Minimum signal length."""
        self.max_signal_length = max_signal_length
        r"""Maximum signal length."""
        self.num_workers = num_workers
        r"""Number of workers."""
        self.multiprocessing = multiprocessing
        r"""Use multiprocessing."""
        self.verbose = verbose
        r"""Show debug messages."""
        if process_func is None:
            def process_func(signal, _):
                return signal
        self.process_func = process_func
        r"""Processing function."""
        self.process_func_is_mono = process_func_is_mono
        r"""Process channels individually."""
        process_func_args = process_func_args or {}
        if kwargs:
            warnings.warn(
                utils.kwargs_deprecation_warning,
                category=UserWarning,
                stacklevel=2,
            )
            for key, value in kwargs.items():
                process_func_args[key] = value
        self.process_func_args = process_func_args
        r"""Additional keyword arguments to processing function."""

    def _process_file(
            self,
            file: str,
            *,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
            root: str = None,
    ) -> pd.Series:

        signal, sampling_rate = utils.read_audio(
            file,
            start=start,
            end=end,
            root=root,
        )
        y = self._process_signal(
            signal,
            sampling_rate,
            file=file,
        )

        if start is None or pd.isna(start):
            start = y.index.levels[1][0]
        if end is None or (pd.isna(end) and not self.keep_nat):
            end = y.index.levels[2][0] + start

        y.index = y.index.set_levels(
            [[start], [end]],
            level=[1, 2],
        )

        return y

    def process_file(
            self,
            file: str,
            *,
            start: Timestamp = None,
            end: Timestamp = None,
            root: str = None,
    ) -> pd.Series:
        r"""Process the content of an audio file.

        Args:
            file: file path
            start: start processing at this position.
                If value is as a float or integer it is treated as seconds.
                To specify a unit provide as string,
                e.g. ``'2ms'``.
                To specify in samples provide as string without unit,
                e.g. ``'2000'``
            end: end processing at this position.
                If value is as a float or integer it is treated as seconds.
                To specify a unit provide as string,
                e.g. ``'2ms'``.
                To specify in samples provide as string without unit,
                e.g. ``'2000'``
            root: root folder to expand relative file path

        Returns:
            Series with processed file conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        if self.segment is not None:
            index = self.segment.process_file(
                file,
                start=start,
                end=end,
                root=root,
            )
            return self._process_index_wo_segment(index, root)
        else:
            start = utils.to_timedelta(start, self.sampling_rate)
            end = utils.to_timedelta(end, self.sampling_rate)
            return self._process_file(file, start=start, end=end, root=root)

    def process_files(
            self,
            files: typing.Sequence[str],
            *,
            starts: Timestamps = None,
            ends: Timestamps = None,
            root: str = None,
    ) -> pd.Series:
        r"""Process a list of files.

        Args:
            files: list of file paths
            starts: segment start positions.
                Time values given as float or integers are treated as seconds.
                To specify a unit provide as string,
                e.g. ``'2ms'``.
                To specify in samples provide as string without unit,
                e.g. ``'2000'``.
                If a scalar is given, it is applied to all files
            ends: segment end positions.
                Time values given as float or integers are treated as seconds
                To specify a unit provide as string,
                e.g. ``'2ms'``.
                To specify in samples provide as string without unit,
                e.g. ``'2000'``.
                If a scalar is given, it is applied to all files
            root: root folder to expand relative file paths

        Returns:
            Series with processed files conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        if len(files) == 0:
            return pd.Series(dtype=object)

        if isinstance(starts, (type(None), float, int, str, pd.Timedelta)):
            starts = [starts] * len(files)
        if isinstance(ends, (type(None), float, int, str, pd.Timedelta)):
            ends = [ends] * len(files)

        params = [
            (
                (file, ),
                {
                    'start': start,
                    'end': end,
                    'root': root,
                },
            ) for file, start, end in zip(files, starts, ends)
        ]
        verbose = self.verbose
        self.verbose = False  # avoid nested progress bar
        y = audeer.run_tasks(
            self.process_file,
            params,
            num_workers=self.num_workers,
            multiprocessing=self.multiprocessing,
            progress_bar=verbose,
            task_description=f'Process {len(files)} files',
        )
        self.verbose = verbose
        return pd.concat(y)

    def process_folder(
            self,
            root: str,
            *,
            filetype: str = 'wav',
    ) -> pd.Series:
        r"""Process files in a folder.

        .. note:: At the moment does not scan in sub-folders!

        Args:
            root: root folder
            filetype: file extension

        Returns:
            Series with processed files conform to audformat_

        Raises:
            FileNotFoundError: if folder does not exist
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        root = audeer.safe_path(root)
        if not os.path.exists(root):
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                root,
            )

        files = audeer.list_file_names(root, filetype=filetype)
        files = [os.path.join(root, os.path.basename(f)) for f in files]
        return self.process_files(files)

    def _process_index_wo_segment(
            self,
            index: pd.Index,
            root: typing.Optional[str],
    ) -> pd.Series:
        r"""Like process_index, but does not apply segmentation."""
        if index.empty:
            return pd.Series(None, index=index, dtype=object)

        params = [
            (
                (file, ),
                {
                    'start': start,
                    'end': end,
                    'root': root,
                },
            )
            for file, start, end in index
        ]
        y = audeer.run_tasks(
            self._process_file,
            params,
            num_workers=self.num_workers,
            multiprocessing=self.multiprocessing,
            progress_bar=self.verbose,
            task_description=f'Process {len(index)} segments',
        )

        return pd.concat(y)

    def process_index(
            self,
            index: pd.Index,
            *,
            root: str = None,
            cache_root: str = None,
    ) -> pd.Series:
        r"""Process from an index conform to audformat_.

        If ``cache_root`` is not ``None``,
        a hash value is created from the index
        using :func:`audformat.utils.hash` and
        the result is stored as
        ``<cache_root>/<hash>.pkl``.
        When called again with the same index,
        features will be read from the cached file.

        Args:
            index: index with segment information
            root: root folder to expand relative file paths
            cache_root: cache folder (see description)

        Returns:
            Series with processed segments conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        cache_path = None

        if cache_root is not None:
            cache_root = audeer.mkdir(cache_root)
            hash = audformat.utils.hash(index)
            cache_path = os.path.join(cache_root, f'{hash}.pkl')

        if cache_path and os.path.exists(cache_path):
            y = pd.read_pickle(cache_path)
        else:
            index = audformat.utils.to_segmented_index(index)

            if self.segment is not None:
                index = self.segment.process_index(index, root=root)

            y = self._process_index_wo_segment(index, root)

            if cache_path is not None:
                y.to_pickle(cache_path, protocol=4)

        return y

    def _process_signal(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            file: str = None,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
    ) -> pd.Series:

        signal = np.atleast_2d(signal)

        # Find start and end index
        if start is None or pd.isna(start):
            start = pd.to_timedelta(0)
        if end is None or (pd.isna(end) and not self.keep_nat):
            end = pd.to_timedelta(signal.shape[-1] / sampling_rate, unit='s')
        start_i, end_i = utils.segment_to_indices(
            signal, sampling_rate, start, end,
        )

        # Trim signal and ensure it has requested min/max length
        signal = signal[:, start_i:end_i]
        num_samples = signal.shape[1]
        if (
                self.max_signal_length is not None
                and num_samples > self.max_signal_length
        ):
            signal = signal[:, :self.max_signal_length]
        if (
                self.min_signal_length is not None
                and num_samples < self.min_signal_length
        ):
            num_pad = self.min_signal_length - num_samples
            signal = np.pad(signal, ((0, 0), (0, num_pad)), 'constant')

        # Process signal
        y = self(signal, sampling_rate)

        # Create index
        if file is not None:
            index = audformat.segmented_index(file, start, end)
        else:
            index = utils.signal_index(start, end)

        return pd.Series([y], index)

    def process_signal(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            file: str = None,
            start: Timestamp = None,
            end: Timestamp = None,
    ) -> typing.Any:
        r"""Process audio signal and return result.

        .. note:: If a ``file`` is given, the index of the returned frame
            has levels ``file``, ``start`` and ``end``. Otherwise,
            it consists only of ``start`` and ``end``.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            file: file path
            start: start processing at this position.
                If value is as a float or integer it is treated as seconds.
                To specify a unit provide as string,
                e.g. ``'2ms'``.
                To specify in samples provide as string without unit,
                e.g. ``'2000'``
            end: end processing at this position.
                To specify a unit provide as string,
                e.g. ``'2ms'``.
                To specify in samples provide as string without unit,
                e.g. ``'2000'``
                If value is as a float or integer it is treated as seconds.

        Returns:
            Series with processed signal conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        if self.segment is not None:
            index = self.segment.process_signal(
                signal,
                sampling_rate,
                file=file,
                start=start,
                end=end,
            )
            return self._process_signal_from_index_wo_segment(
                signal,
                sampling_rate,
                index,
            )
        else:
            start = utils.to_timedelta(start, sampling_rate)
            end = utils.to_timedelta(end, sampling_rate)
            return self._process_signal(
                signal,
                sampling_rate,
                file=file,
                start=start,
                end=end,
            )

    def _process_signal_from_index_wo_segment(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            index: pd.Index,
    ) -> pd.Series:
        r"""Like process_signal_from_index, but does not apply segmentation."""

        if index.empty:
            return pd.Series(None, index=index, dtype=object)

        if isinstance(index, pd.MultiIndex) and len(index.levels) == 2:
            params = [
                (
                    (signal, sampling_rate),
                    {'start': start, 'end': end},
                ) for start, end in index
            ]
        else:
            index = audformat.utils.to_segmented_index(index)
            params = [
                (
                    (signal, sampling_rate),
                    {'file': file, 'start': start, 'end': end},
                ) for file, start, end in index
            ]

        y = audeer.run_tasks(
            self._process_signal,
            params,
            num_workers=self.num_workers,
            multiprocessing=self.multiprocessing,
            progress_bar=self.verbose,
            task_description=f'Process {len(index)} segments',
        )

        return pd.concat(y)

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
            index: a segmented index conform to audformat_
                or a :class:`pandas.MultiIndex` with two levels
                named `start` and `end` that hold start and end
                positions as :class:`pandas.Timedelta` objects.
                See also :func:`audinterface.utils.signal_index`

        Returns:
            Series with processed segments conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            ValueError: if index contains duplicates

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        utils.assert_index(index)

        if index.empty:
            return pd.Series(None, index=index, dtype=object)

        if self.segment is not None:
            index = self.segment.process_signal_from_index(
                signal,
                sampling_rate,
                index,
            )

        return self._process_signal_from_index_wo_segment(
            signal,
            sampling_rate,
            index,
        )

    def __call__(
            self,
            signal: np.ndarray,
            sampling_rate: int,
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
        signal, sampling_rate = utils.preprocess_signal(
            signal,
            sampling_rate,
            self.sampling_rate,
            self.resample,
            self.channels,
            self.mixdown,
        )
        if self.process_func_is_mono:
            return [
                self.process_func(
                    np.atleast_2d(channel),
                    sampling_rate,
                    **self.process_func_args,
                ) for channel in signal
            ]
        return self.process_func(
            signal,
            sampling_rate,
            **self.process_func_args,
        )


class ProcessWithContext:
    r"""Alternate processing interface that provides signal context.

    In contrast to :class:`Process` this interface does not look at segments
    in isolation, but passes the complete signal together with a list of
    segments to the processing function. By doing so, it becomes possible to
    process segments in context, e.g. by taking into account surrounding
    signal values or other segments.

    Args:
        process_func: processing function, which expects four positional
            arguments:

            * ``signal``
            * ``sampling_rate``
            * ``starts`` sequence with start indices
            * ``ends`` sequence with end indices

            and any number of additional keyword arguments.
            Must return a sequence of results for every segment
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

    Example:
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
        if resample and sampling_rate is None:
            raise ValueError(
                'sampling_rate has to be provided for resample = True.'
            )
        self.sampling_rate = sampling_rate
        r"""Sampling rate in Hz."""
        self.resample = resample
        r"""Resample signal."""
        self.channels = None if channels is None else audeer.to_list(channels)
        r"""Channel selection."""
        self.mixdown = mixdown
        r"""Mono mixdown."""
        self.verbose = verbose
        r"""Show debug messages."""
        if process_func is None:
            def process_func(signal, _, starts, ends):
                return [
                    signal[:, start:end] for start, end in zip(starts, ends)
                ]
        self.process_func = process_func
        r"""Process function."""
        process_func_args = process_func_args or {}
        if kwargs:
            warnings.warn(
                utils.kwargs_deprecation_warning,
                category=UserWarning,
                stacklevel=2,
            )
            for key, value in kwargs.items():
                process_func_args[key] = value
        self.process_func_args = process_func_args
        r"""Additional keyword arguments to processing function."""

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
                    self.process_signal_from_index(
                        signal, sampling_rate, select,
                    ).values,
                    index=index[mask],
                )

        return pd.concat(ys)

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
        utils.assert_index(index)

        if len(index) == 0:
            y = pd.Series([], index=index, dtype=object)
        else:
            starts_i, ends_i = utils.segments_to_indices(
                signal, sampling_rate, index,
            )
            y = self(signal, sampling_rate, starts_i, ends_i)
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
        signal, sampling_rate = utils.preprocess_signal(
            signal,
            sampling_rate,
            self.sampling_rate,
            self.resample,
            self.channels,
            self.mixdown,
        )
        return self.process_func(
            signal,
            sampling_rate,
            starts,
            ends,
            **self.process_func_args,
        )
