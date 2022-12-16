import errno
import inspect
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
            There is no restriction on the return type of the function
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
        win_dur: window duration,
            if processing should be applied on a sliding window.
            If value is a float or integer
            it is treated as seconds.
            See :func:`audinterface.utils.to_timedelta` for further options
        hop_dur: hop duration,
            if processing should be applied on a sliding window.
            This defines the shift between two windows.
            If value is a float or integer
            it is treated as seconds.
            See :func:`audinterface.utils.to_timedelta` for further options.
            Defaults to ``win_dur / 2``
        min_signal_dur: minimum signal length
            required by ``process_func``.
            If value is a float or integer
            it is treated as seconds.
            See :func:`audinterface.utils.to_timedelta` for further options.
            If provided signal is shorter,
            it will be zero padded at the end
        max_signal_dur: maximum signal length
            required by ``process_func``.
            If value is a float or integer
            it is treated as seconds.
            See :func:`audinterface.utils.to_timedelta` for further options.
            If provided signal is longer,
            it will be cut at the end
        segment: when a :class:`audinterface.Segment` object is provided,
            it will be used to find a segmentation of the input signal.
            Afterwards processing is applied to each segment
        keep_nat: if the end of segment is set to ``NaT`` do not replace
            with file duration in the result
        num_workers: number of parallel jobs or 1 for sequential
            processing. If ``None`` will be set to the number of
            processors on the machine multiplied by 5 in case of
            multithreading and number of processors in case of
            multiprocessing
        multiprocessing: use multiprocessing instead of multithreading
        verbose: show debug messages

    Raises:
        ValueError: if ``resample = True``, but ``sampling_rate = None``
        ValueError: if ``hop_dur`` is specified, but not ``win_dur``

    Examples:
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
        >>> # Apply interface on an audformat conform index of a dataframe
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
        >>> # Apply interface with a sliding window
        >>> interface = Process(
        ...     process_func=mean,
        ...     win_dur=1.0,
        ...     hop_dur=0.5,
        ... )
        >>> interface.process_index(index, root=db.root)
        file             start                   end
        wav/03a01Fa.wav  0 days 00:00:00         0 days 00:00:01          -0.000329
                         0 days 00:00:00.500000  0 days 00:00:01.500000   -0.000285
        dtype: float32

    """  # noqa: E501
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
            win_dur: Timestamp = None,
            hop_dur: Timestamp = None,
            min_signal_dur: Timestamp = None,
            max_signal_dur: Timestamp = None,
            segment: Segment = None,
            keep_nat: bool = False,
            num_workers: typing.Optional[int] = 1,
            multiprocessing: bool = False,
            verbose: bool = False,
            **kwargs,
    ):
        if channels is not None:
            channels = audeer.to_list(channels)

        if process_func is None:
            def process_func(signal, _):
                return signal

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

        if win_dur is None and hop_dur is not None:
            raise ValueError(
                "You have to specify 'win_dur' if 'hop_dur' is given."
            )
        if win_dur is not None and hop_dur is None:
            hop_dur = utils.to_timedelta(win_dur, sampling_rate) / 2

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

        self.keep_nat = keep_nat
        r"""Keep NaT in results."""

        self.hop_dur = hop_dur
        r"""Hop duration."""

        self.max_signal_dur = max_signal_dur
        r"""Maximum signal length."""

        self.min_signal_dur = min_signal_dur
        r"""Minimum signal length."""

        self.mixdown = mixdown
        r"""Mono mixdown."""

        self.multiprocessing = multiprocessing
        r"""Use multiprocessing."""

        self.num_workers = num_workers
        r"""Number of workers."""

        self.process_func = process_func
        r"""Processing function."""

        self.process_func_args = process_func_args
        r"""Additional keyword arguments to processing function."""

        self.process_func_is_mono = process_func_is_mono
        r"""Process channels individually."""

        self.resample = resample
        r"""Resample signal."""

        self.sampling_rate = sampling_rate
        r"""Sampling rate in Hz."""

        self.segment = segment
        r"""Segmentation object."""

        self.verbose = verbose
        r"""Show debug messages."""

        self.win_dur = win_dur
        r"""Window duration."""

    def _process_file(
            self,
            file: str,
            *,
            idx: int = 0,
            root: str = None,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
    ) -> pd.Series:

        start = utils.to_timedelta(start, self.sampling_rate)
        end = utils.to_timedelta(end, self.sampling_rate)

        signal, sampling_rate = utils.read_audio(
            file,
            start=start,
            end=end,
            root=root,
        )
        y = self._process_signal(
            signal,
            sampling_rate,
            idx=idx,
            root=root,
            file=file,
        )

        if self.win_dur is not None:

            if start is not None:
                starts = y.index.levels[1] + start
                ends = y.index.levels[2] + start
                y.index = y.index.set_levels(
                    [starts, ends],
                    level=[1, 2],
                )

        else:

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
                If value is a float or integer it is treated as seconds.
                See :func:`audinterface.utils.to_timedelta` for further options
            end: end processing at this position.
                If value is a float or integer it is treated as seconds.
                See :func:`audinterface.utils.to_timedelta` for further options
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
            return self._process_file(
                file,
                root=root,
                start=start,
                end=end,
            )

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
                See :func:`audinterface.utils.to_timedelta`
                for further options.
                If a scalar is given, it is applied to all files
            ends: segment end positions.
                Time values given as float or integers are treated as seconds.
                See :func:`audinterface.utils.to_timedelta`
                for further options.
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
                    'idx': idx,
                    'root': root,
                    'start': start,
                    'end': end,
                },
            ) for idx, (file, start, end)
            in enumerate(zip(files, starts, ends))
        ]

        verbose = self.verbose
        self.verbose = False  # avoid nested progress bar
        y = audeer.run_tasks(
            self._process_file,
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
                    'idx': idx,
                    'root': root,
                    'start': start,
                    'end': end,
                },
            )
            for idx, (file, start, end) in enumerate(index)
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
            idx: int = 0,
            root: str = None,
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
        if self.max_signal_dur is not None:
            max_signal_dur = utils.to_timedelta(
                self.max_signal_dur,
                sampling_rate,
            )
            max_samples = int(
                max_signal_dur.total_seconds()
                * sampling_rate
            )
            if num_samples > max_samples:
                end = start + max_signal_dur
                signal = signal[:, :max_samples]
        if self.min_signal_dur is not None:
            min_signal_dur = utils.to_timedelta(
                self.min_signal_dur,
                sampling_rate,
            )
            min_samples = int(
                min_signal_dur.total_seconds()
                * sampling_rate
            )
            if num_samples < min_samples:
                end = start + min_signal_dur
                num_pad = min_samples - num_samples
                signal = np.pad(signal, ((0, 0), (0, num_pad)), 'constant')

        # Process signal
        y = self._call(
            signal,
            sampling_rate,
            idx=idx,
            root=root,
            file=file,
        )

        # Create index
        if self.win_dur is not None:
            win_dur = utils.to_timedelta(self.win_dur, sampling_rate)
            hop_dur = utils.to_timedelta(self.hop_dur, sampling_rate)
            starts = pd.timedelta_range(
                start,
                freq=hop_dur,
                periods=len(y),
            )
            ends = starts + win_dur
        else:
            starts = [start]
            ends = [end]
            y = [y]

        if file is not None:
            files = [file] * len(starts)
            index = audformat.segmented_index(files, starts, ends)
        else:
            index = utils.signal_index(starts, ends)

        if len(y) == 0:
            return pd.Series([], index, dtype=object)
        else:
            return pd.Series(y, index)

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
                If value is a float or integer it is treated as seconds.
                See :func:`audinterface.utils.to_timedelta` for further options
            end: end processing at this position.
                If value is a float or integer it is treated as seconds.
                See :func:`audinterface.utils.to_timedelta` for further options

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
                    {
                        'idx': idx,
                        'start': start,
                        'end': end,
                    },
                ) for idx, (start, end) in enumerate(index)
            ]
        else:
            index = audformat.utils.to_segmented_index(index)
            params = [
                (
                    (signal, sampling_rate),
                    {
                        'idx': idx,
                        'file': file,
                        'start': start,
                        'end': end,
                    },
                ) for idx, (file, start, end) in enumerate(index)
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

    def _call(
            self,
            signal: np.ndarray,
            sampling_rate: int,
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

        def _helper(x):
            if self.process_func_is_mono:
                return [
                    self.process_func(
                        np.atleast_2d(channel),
                        sampling_rate,
                        **special_args,
                        **self.process_func_args,
                    ) for channel in x
                ]
            else:
                return self.process_func(
                    x,
                    sampling_rate,
                    **special_args,
                    **self.process_func_args,
                )

        if self.win_dur is not None:
            frames = utils.sliding_window(
                signal,
                sampling_rate,
                self.win_dur,
                self.hop_dur,
            )
            num_frames = frames.shape[-1]
            y = [_helper(frames[..., idx]) for idx in range(num_frames)]
        else:
            y = _helper(signal)

        return y

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
        return self._call(signal, sampling_rate)
