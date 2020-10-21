import os
import typing

import numpy as np
import pandas as pd

import audeer

from audinterface.core import utils


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
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
            channel: int = None,
    ) -> pd.MultiIndex:
        r"""Segment the content of an audio file.

        Args:
            file: file path
            channel: channel number
            start: start processing at this position
            end: end processing at this position

        Returns:
            Segmented index in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        if start is None or pd.isna(start):
            start = pd.to_timedelta(0)
        index = self.process.process_file(
            file, start=start, end=end, channel=channel,
        ).values[0]
        return pd.MultiIndex(
            [[file], index.levels[0] + start, index.levels[1] + start],
            [[0] * len(index), index.codes[0], index.codes[1]],
            names=['file', 'start', 'end'],
        )

    def process_files(
            self,
            files: typing.Sequence[str],
            *,
            starts: typing.Sequence[pd.Timedelta] = None,
            ends: typing.Sequence[pd.Timedelta] = None,
            channel: int = None,
    ) -> pd.MultiIndex:
        r"""Segment a list of files.

        Args:
            files: list of file paths
            starts: list with start positions
            ends: list with end positions
            channel: channel number

        Returns:
            Segmented index in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        series = self.process.process_files(
            files,
            starts=starts,
            ends=ends,
            channel=channel,
        )
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

    def process_folder(
            self,
            root: str,
            *,
            channel: int = None,
            filetype: str = 'wav',
    ) -> pd.MultiIndex:
        r"""Segment files in a folder.

        .. note:: At the moment does not scan in sub-folders!

        Args:
            root: root folder
            channel: channel number
            filetype: file extension

        Returns:
            Segmented index in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        files = audeer.list_file_names(root, filetype=filetype)
        files = [os.path.join(root, os.path.basename(f)) for f in files]
        return self.process_files(files, channel=channel)

    def process_signal(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            file: str = None,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
    ) -> pd.MultiIndex:
        r"""Segment audio signal.

        .. note:: If a ``file`` is given, the index of the returned frame
            has levels ``file``, ``start`` and ``end``. Otherwise,
            it consists only of ``start`` and ``end``.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            file: file path
            start: start processing at this position
            end: end processing at this position

        Returns:
            Segmented index in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        index = self.process.process_signal(
            signal, sampling_rate, file=file, start=start, end=end,
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
        signal. However, if resampling is enabled and the input sampling
        rate does not fit the expected sampling rate, the input signal will
        be resampled before the processing is applied.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz

        Returns:
            Processed signal

        """
        return self.process(
            signal,
            sampling_rate,
        )
