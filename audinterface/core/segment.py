import os
import typing

import numpy as np
import pandas as pd

import audeer

from audinterface.core import utils
from audinterface.core.typing import (
    Timestamp,
    Timestamps,
)


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
        if process_func is None:
            def process_func(signal, sr, **kwargs):
                return pd.MultiIndex.from_arrays(
                    [
                        pd.to_timedelta([]),
                        pd.to_timedelta([]),
                    ],
                    names=['start', 'end'],
                )
        # avoid cycling imports
        from audinterface.core.process import Process
        self.process = Process(
            process_func=process_func,
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

    @audeer.deprecated_keyword_argument(
        deprecated_argument='channel',
        removal_version='0.6.0',
    )
    def process_file(
            self,
            file: str,
            *,
            start: Timestamp = None,
            end: Timestamp = None,
    ) -> pd.MultiIndex:
        r"""Segment the content of an audio file.

        Args:
            file: file path
            start: start processing at this position.
                If value is as a float or integer it is treated as seconds
            end: end processing at this position.
                If value is as a float or integer it is treated as seconds

        Returns:
            Segmented index conform to audformat

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        """
        if start is None or pd.isna(start):
            start = pd.to_timedelta(0)
        index = self.process.process_file(file, start=start, end=end).values[0]
        return pd.MultiIndex(
            [[file], index.levels[0] + start, index.levels[1] + start],
            [[0] * len(index), index.codes[0], index.codes[1]],
            names=['file', 'start', 'end'],
        )

    @audeer.deprecated_keyword_argument(
        deprecated_argument='channel',
        removal_version='0.6.0',
    )
    def process_files(
            self,
            files: typing.Sequence[str],
            *,
            starts: Timestamps = None,
            ends: Timestamps = None,
    ) -> pd.MultiIndex:
        r"""Segment a list of files.

        Args:
            files: list of file paths
            starts: segment start positions.
                Time values given as float or integers are treated as seconds.
                If a scalar is given, it is applied to all files
            ends: segment end positions.
                Time values given as float or integers are treated as seconds
                If a scalar is given, it is applied to all files

        Returns:
            Segmented index conform to audformat

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        """
        series = self.process.process_files(files, starts=starts, ends=ends)
        tuples = []
        for idx, ((file, start, _), index) in enumerate(series.items()):
            tuples.extend(
                pd.MultiIndex(
                    [[file], index.levels[0] + start, index.levels[1] + start],
                    [[0] * len(index), index.codes[0], index.codes[1]],
                ).to_list()
            )
        return pd.MultiIndex.from_tuples(
            tuples, names=['file', 'start', 'end'],
        )

    @audeer.deprecated_keyword_argument(
        deprecated_argument='channel',
        removal_version='0.6.0',
    )
    def process_folder(
            self,
            root: str,
            *,
            filetype: str = 'wav',
    ) -> pd.MultiIndex:
        r"""Segment files in a folder.

        .. note:: At the moment does not scan in sub-folders!

        Args:
            root: root folder
            filetype: file extension

        Returns:
            Segmented index conform to audformat

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        """
        files = audeer.list_file_names(root, filetype=filetype)
        files = [os.path.join(root, os.path.basename(f)) for f in files]
        return self.process_files(files)

    def process_index(
            self,
            index: pd.Index,
    ) -> pd.MultiIndex:
        r"""Segment files or segments from an index.

        Args:
            index: index conform to audformat

        Returns:
            Segmented index conform to audformat

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        """
        index = utils.to_segmented_index(index)
        utils.check_index(index)

        if index.empty:
            return index

        return self.process_files(
            index.get_level_values('file'),
            starts=index.get_level_values('start'),
            ends=index.get_level_values('end'),
        )

    def process_signal(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            file: str = None,
            start: Timestamp = None,
            end: Timestamp = None,
    ) -> pd.MultiIndex:
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
            Segmented index conform to audformat

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        """
        index = self.process.process_signal(
            signal,
            sampling_rate,
            file=file,
            start=start,
            end=end,
        ).values[0]
        utils.check_index(index)
        if start is not None:
            index = index.set_levels(
                [
                    index.levels[0] + start,
                    index.levels[1] + start,
                ], [0, 1])
        if file is not None:
            index = pd.MultiIndex(
                levels=[
                    [file], index.levels[0], index.levels[1],
                ],
                codes=[
                    [0] * len(index), index.codes[0], index.codes[1],
                ],
                names=['file', 'start', 'end'],
            )
        return index

    def __call__(
            self,
            signal: np.ndarray,
            sampling_rate: int,
    ) -> pd.MultiIndex:
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
