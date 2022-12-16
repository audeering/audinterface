import errno
import os
import typing
import warnings

import numpy as np
import pandas as pd

import audeer
import audformat

from audinterface.core import utils
from audinterface.core.typing import (
    Timestamp,
    Timestamps,
)


def create_process_func(
        process_func: typing.Optional[typing.Callable[..., pd.MultiIndex]],
        invert: bool,
) -> typing.Callable[..., pd.MultiIndex]:
    r"""Create processing function."""

    if process_func is None:
        def process_func(signal, sr, **kwargs):
            return utils.signal_index()

    if invert:
        def process_func_invert(signal, sr, **kwargs):
            index = process_func(signal, sr, **kwargs)
            dur = pd.to_timedelta(signal.shape[-1] / sr, unit='s')
            index = index.sortlevel('start')[0]
            index = merge_index(index)
            index = invert_index(index, dur)
            return index
        return process_func_invert
    else:
        return process_func


def invert_index(
        index: pd.MultiIndex,
        dur: pd.Timedelta,
) -> pd.MultiIndex:
    r"""Invert index.

    Assumes that index is sorted by 'start' level.

    """
    if index.empty:
        return utils.signal_index(0, dur)

    starts = index.get_level_values('start')
    ends = index.get_level_values('end')
    new_starts = ends[:-1]
    new_ends = starts[1:]
    if starts[0] != pd.to_timedelta(0):
        new_starts = new_starts.insert(0, pd.to_timedelta(0))
        new_ends = new_ends.insert(0, starts[0])
    if ends[-1] != dur:
        new_starts = new_starts.insert(len(new_starts), ends[-1])
        new_ends = new_ends.insert(len(new_ends), dur)
    return utils.signal_index(new_starts, new_ends)


def merge_index(
        index: pd.MultiIndex,
) -> pd.MultiIndex:
    r"""Merge overlapping segments.

    Assumes that index is sorted by 'start' level.

    """
    if index.empty:
        return index

    starts = index.get_level_values('start')
    ends = index.get_level_values('end')
    new_starts = []
    new_ends = []
    new_start = starts[0]
    new_end = ends[0]
    for start, end in zip(starts[1:], ends[1:]):
        if start > new_end:
            new_starts.append(new_start)
            new_ends.append(new_end)
            new_start = start
            new_end = end
        elif end > new_end:
            new_end = end
    new_starts.append(new_start)
    new_ends.append(new_end)

    return utils.signal_index(new_starts, new_ends)


class Segment:
    r"""Segmentation interface.

    Interface for models that apply a segmentation to the input signal,
    e.g. a voice activity model that detects speech regions.

    Args:
        process_func: segmentation function,
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
            Must return a :class:`pandas.MultiIndex` with two levels
            named `start` and `end` that hold start and end
            positions as :class:`pandas.Timedelta` objects
        process_func_args: (keyword) arguments passed on to the processing
            function
        invert: Invert the segmentation
        sampling_rate: sampling rate in Hz
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal
        resample: if ``True`` enforces given sampling rate by resampling
        channels: channel selection, see :func:`audresample.remix`
        mixdown: apply mono mix-down on selection
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

    Examples:
        >>> def segment(signal, sampling_rate, *, win_size=0.2, hop_size=0.1):
        ...     size = signal.shape[1] / sampling_rate
        ...     starts = pd.to_timedelta(np.arange(0, size - win_size, hop_size), unit='s')
        ...     ends = pd.to_timedelta(np.arange(win_size, size, hop_size), unit='s')
        ...     return pd.MultiIndex.from_tuples(zip(starts, ends), names=['start', 'end'])
        >>> interface = Segment(process_func=segment)
        >>> signal = np.array([1., 2., 3.])
        >>> interface(signal, sampling_rate=3)
        MultiIndex([(       '0 days 00:00:00', '0 days 00:00:00.200000'),
                    ('0 days 00:00:00.100000', '0 days 00:00:00.300000'),
                    ('0 days 00:00:00.200000', '0 days 00:00:00.400000'),
                    ('0 days 00:00:00.300000', '0 days 00:00:00.500000'),
                    ('0 days 00:00:00.400000', '0 days 00:00:00.600000'),
                    ('0 days 00:00:00.500000', '0 days 00:00:00.700000'),
                    ('0 days 00:00:00.600000', '0 days 00:00:00.800000'),
                    ('0 days 00:00:00.700000', '0 days 00:00:00.900000')],
                   names=['start', 'end'])
        >>> # Apply interface on an audformat conform index of a dataframe
        >>> import audb
        >>> db = audb.load(
        ...     'emodb',
        ...     version='1.2.0',
        ...     media='wav/03a01Fa.wav',
        ...     full_path=False,
        ...     verbose=False,
        ... )
        >>> interface = Segment(
        ...     process_func=segment,
        ...     process_func_args={'win_size': 0.5, 'hop_size': 0.25},
        ... )
        >>> interface.process_index(db['emotion'].index, root=db.root)
        MultiIndex([('wav/03a01Fa.wav',        '0 days 00:00:00', ...),
                    ('wav/03a01Fa.wav', '0 days 00:00:00.250000', ...),
                    ('wav/03a01Fa.wav', '0 days 00:00:00.500000', ...),
                    ('wav/03a01Fa.wav', '0 days 00:00:00.750000', ...),
                    ('wav/03a01Fa.wav',        '0 days 00:00:01', ...),
                    ('wav/03a01Fa.wav', '0 days 00:00:01.250000', ...)],
                   names=['file', 'start', 'end'])

    """  # noqa: E501
    def __init__(
            self,
            *,
            process_func: typing.Callable[..., pd.MultiIndex] = None,
            process_func_args: typing.Dict[str, typing.Any] = None,
            invert: bool = False,
            sampling_rate: int = None,
            resample: bool = False,
            channels: typing.Union[int, typing.Sequence[int]] = None,
            mixdown: bool = False,
            min_signal_dur: Timestamp = None,
            max_signal_dur: Timestamp = None,
            keep_nat: bool = False,
            num_workers: typing.Optional[int] = 1,
            multiprocessing: bool = False,
            verbose: bool = False,
            **kwargs,
    ):
        process_func_args = process_func_args or {}
        if kwargs:
            warnings.warn(
                utils.kwargs_deprecation_warning,
                category=UserWarning,
                stacklevel=2,
            )
            for key, value in kwargs.items():
                process_func_args[key] = value

        # avoid cycling imports
        from audinterface.core.process import Process
        process = Process(
            process_func=create_process_func(process_func, invert),
            process_func_args=process_func_args,
            sampling_rate=sampling_rate,
            resample=resample,
            channels=channels,
            mixdown=mixdown,
            min_signal_dur=min_signal_dur,
            max_signal_dur=max_signal_dur,
            keep_nat=keep_nat,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
            verbose=verbose,
        )

        self.process = process
        r"""Processing object."""

        self.invert = invert
        r"""Invert segmentation."""

    def process_file(
            self,
            file: str,
            *,
            start: Timestamp = None,
            end: Timestamp = None,
            root: str = None,
    ) -> pd.Index:
        r"""Segment the content of an audio file.

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
            Segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        if start is None or pd.isna(start):
            start = pd.to_timedelta(0)
        index = self.process.process_file(
            file,
            start=start,
            end=end,
            root=root,
        ).values[0]
        return audformat.segmented_index(
            files=[file] * len(index),
            starts=index.levels[0] + start,
            ends=index.levels[1] + start,
        )

    def process_files(
            self,
            files: typing.Sequence[str],
            *,
            starts: Timestamps = None,
            ends: Timestamps = None,
            root: str = None,
    ) -> pd.Index:
        r"""Segment a list of files.

        Args:
            files: list of file paths
            starts: segment start positions.
                Time values given as float or integers are treated as seconds.
                See :func:`audinterface.utils.to_timedelta`
                for further options.
                If a scalar is given, it is applied to all files
            ends: segment end positions.
                Time values given as float or integers are treated as seconds
                See :func:`audinterface.utils.to_timedelta`
                for further options.
                If a scalar is given, it is applied to all files
            root: root folder to expand relative file paths

        Returns:
            Segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        series = self.process.process_files(
            files,
            starts=starts,
            ends=ends,
            root=root,
        )
        if len(series) == 0:
            return audformat.filewise_index()
        objs = []
        for idx, ((file, start, _), index) in enumerate(series.items()):
            objs.append(
                audformat.segmented_index(
                    files=[file] * len(index),
                    starts=index.levels[0] + start,
                    ends=index.levels[1] + start,
                )
            )
        return audformat.utils.union(objs)

    def process_folder(
            self,
            root: str,
            *,
            filetype: str = 'wav',
    ) -> pd.Index:
        r"""Segment files in a folder.

        .. note:: At the moment does not scan in sub-folders!

        Args:
            root: root folder
            filetype: file extension

        Returns:
            Segmented index conform to audformat_

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

    def process_index(
            self,
            index: pd.Index,
            *,
            root: str = None,
    ) -> pd.Index:
        r"""Segment files or segments from an index.

        Args:
            index: index conform to audformat_
            root: root folder to expand relative file paths

        Returns:
            Segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        index = audformat.utils.to_segmented_index(index)
        utils.assert_index(index)

        if index.empty:
            return index

        return self.process_files(
            index.get_level_values('file'),
            starts=index.get_level_values('start'),
            ends=index.get_level_values('end'),
            root=root,
        )

    def process_signal(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            file: str = None,
            start: Timestamp = None,
            end: Timestamp = None,
    ) -> pd.Index:
        r"""Segment audio signal.

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
            Segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        index = self.process.process_signal(
            signal,
            sampling_rate,
            file=file,
            start=start,
            end=end,
        ).values[0]
        utils.assert_index(index)
        if start is not None:
            index = index.set_levels(
                [
                    index.levels[0] + start,
                    index.levels[1] + start,
                ],
                level=[0, 1],
            )
        if file is not None:
            index = audformat.segmented_index(
                files=[file] * len(index),
                starts=index.levels[0],
                ends=index.levels[1],
            )
        return index

    def process_signal_from_index(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            index: pd.Index,
    ) -> pd.Index:
        r"""Segment parts of a signal.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            index: a segmented index conform to audformat_
                or a :class:`pandas.MultiIndex` with two levels
                named `start` and `end` that hold start and end
                positions as :class:`pandas.Timedelta` objects.
                See also :func:`audinterface.utils.signal_index`

        Returns:
            Segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            ValueError: if index contains duplicates

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        utils.assert_index(index)

        if index.empty:
            return index

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
            self.process_signal,
            params,
            num_workers=self.process.num_workers,
            multiprocessing=self.process.multiprocessing,
            progress_bar=self.process.verbose,
            task_description=f'Process {len(index)} segments',
        )
        index = audformat.utils.union(y)

        return index

    def __call__(
            self,
            signal: np.ndarray,
            sampling_rate: int,
    ) -> pd.Index:
        r"""Apply processing to signal.

        This function processes the signal **without** transforming the output
        into a :class:`pd.MultiIndex`. Instead it will return the raw processed
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
        return self.process(
            signal,
            sampling_rate,
        )
