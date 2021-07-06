import os
import typing

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
            and any number of additional keyword arguments.
            Must return a :class:`pandas.MultiIndex` with two levels
            named `start` and `end` that hold start and end
            positions as :class:`pandas.Timedelta` objects.
        invert: Invert the segmentation
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal.
        resample: if ``True`` enforces given sampling rate by resampling
        channels: channel selection, see :func:`audresample.remix`
        mixdown: apply mono mix-down on selection
        keep_nat: if the end of segment is set to ``NaT`` do not replace
            with file duration in the result
        num_workers: number of parallel jobs or 1 for sequential
            processing. If ``None`` will be set to the number of
            processors on the machine multiplied by 5 in case of
            multithreading and number of processors in case of
            multiprocessing
        multiprocessing: use multiprocessing instead of multithreading
        verbose: show debug messages
        kwargs: additional keyword arguments to the processing function

    Raises:
        ValueError: if ``resample = True``, but ``sampling_rate = None``

    """
    def __init__(
            self,
            *,
            process_func: typing.Callable[..., pd.MultiIndex] = None,
            invert: bool = False,
            sampling_rate: int = None,
            resample: bool = False,
            channels: typing.Union[int, typing.Sequence[int]] = None,
            mixdown: bool = False,
            keep_nat: bool = False,
            num_workers: typing.Optional[int] = 1,
            multiprocessing: bool = False,
            verbose: bool = False,
            **kwargs,
    ):
        self.invert = invert
        r"""Invert segmentation."""

        # avoid cycling imports
        from audinterface.core.process import Process
        self.process = Process(
            process_func=create_process_func(process_func, invert),
            sampling_rate=sampling_rate,
            resample=resample,
            channels=channels,
            mixdown=mixdown,
            keep_nat=keep_nat,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
            verbose=verbose,
            **kwargs,
        )
        r"""Processing object."""

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
                If value is as a float or integer it is treated as seconds
            end: end processing at this position.
                If value is as a float or integer it is treated as seconds
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
        return pd.MultiIndex(
            levels=[
                [file],
                index.levels[0] + start,
                index.levels[1] + start,
            ],
            codes=[
                [0] * len(index),
                index.codes[0],
                index.codes[1],
            ],
            names=['file', 'start', 'end'],
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
                If a scalar is given, it is applied to all files
            ends: segment end positions.
                Time values given as float or integers are treated as seconds
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
        objs = []
        for idx, ((file, start, _), index) in enumerate(series.items()):
            objs.append(
                pd.MultiIndex(
                    levels=[
                        [file],
                        index.levels[0] + start,
                        index.levels[1] + start,
                    ],
                    codes=[
                        [0] * len(index),
                        index.codes[0],
                        index.codes[1],
                    ],
                    names=['file', 'start', 'end'],
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
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
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
                If value is as a float or integer it is treated as seconds
            end: end processing at this position.
                If value is as a float or integer it is treated as seconds

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
            index = pd.MultiIndex(
                levels=[
                    [file],
                    index.levels[0],
                    index.levels[1],
                ],
                codes=[
                    [0] * len(index),
                    index.codes[0],
                    index.codes[1],
                ],
                names=['file', 'start', 'end'],
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

        index = y[0]
        for obj in y[1:]:
            index = index.union(obj)

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
