import os
import typing

import numpy as np
import pandas as pd

import audeer

from audinterface.core import utils
from audinterface.core.process import Process


class Segment:
    r"""Segmentation interface.

    Interface for models that apply a segmentation to the input signal,
    e.g. a voice activity model that detects speech regions.

    Args:
        segment_func: segmentation function,
            which expects the two positional arguments ``signal``
            and ``sampling_rate``
            and any number of additional keyword arguments.
            Must return a :class:`pandas.MultiIndex` with two levels
            named `start` and `end` that hold start and end
            positions as :class:`pandas.Timedelta` objects.
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal.
        num_workers: number of parallel jobs or 1 for sequential
            processing. If ``None`` will be set to the number of
            processors on the machine multiplied by 5 in case of
            multithreading and number of processors in case of
            multiprocessing
        multiprocessing: use multiprocessing instead of multithreading
        resample: if ``True`` enforces given sampling rate by resampling
        verbose: show debug messages
        kwargs: additional keyword arguments to the processing function

    Raises:
        ValueError: if ``resample = True``, but ``sampling_rate = None``

    """
    def __init__(
            self,
            *,
            segment_func: typing.Callable[..., pd.MultiIndex] = None,
            sampling_rate: int = None,
            resample: bool = False,
            num_workers: typing.Optional[int] = 1,
            multiprocessing: bool = False,
            verbose: bool = False,
            **kwargs,
    ):
        if segment_func is None:
            def segment_func(signal, sr, **kwargs):
                return pd.MultiIndex.from_arrays(
                    [pd.to_timedelta([]), pd.to_timedelta([])],
                    names=['start', 'end']
                )
        self.process = Process(
            process_func=segment_func,
            sampling_rate=sampling_rate,
            resample=resample,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
            verbose=verbose,
            **kwargs,
        )

    def segment_file(
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
        index = self.process.process_file(
            file, start=start, end=end, channel=channel,
        )
        files = [file] * len(index)
        if start is not None:
            starts = index.levels[0] + start
            ends = index.levels[1] + start
        else:
            starts = index.levels[0]
            ends = index.levels[1]
        return pd.MultiIndex.from_arrays(
            [
                files, starts, ends,
            ],
            names=['file', 'start', 'end']
        )

    def segment_files(
            self,
            files: typing.Sequence[str],
            *,
            channel: int = None,
    ) -> pd.MultiIndex:
        r"""Segment a list of files.

        Args:
            files: list of file paths
            channel: channel number

        Returns:
            Segmented index in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        series = self.process.process_files(files, channel=channel)
        files = []
        starts = []
        ends = []
        for file, index in series.items():
            files.extend([file] * len(index))
            starts.extend(index.levels[0])
            ends.extend(index.levels[1])
        return pd.MultiIndex.from_arrays(
            [
                files, starts, ends,
            ],
            names=['file', 'start', 'end']
        )

    def segment_folder(
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
        return self.segment_files(files, channel=channel)

    def segment_signal(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
    ) -> pd.MultiIndex:
        r"""Segment audio signal.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            start: start processing at this position
            end: end processing at this position

        Returns:
            Segmented index in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        index = self.process.process_signal(
            signal, sampling_rate, start=start, end=end,
        )
        utils.check_index(index)
        if start is not None:
            index = index.set_levels(
                [
                    index.levels[0] + start,
                    index.levels[1] + start,
                ], [0, 1])
        return index
