import os
import typing

import numpy as np
import pandas as pd

import audeer
import audsp

from audinterface.core import utils
from audinterface.core.segment import Segment


class Process:
    r"""Processing interface.

    Args:
        process_func: processing function,
            which expects the two positional arguments ``signal``
            and ``sampling_rate``
            and any number of additional keyword arguments.
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal.
        resample: if ``True`` enforces given sampling rate by resampling
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
        kwargs: additional keyword arguments to the processing function

    Raises:
        ValueError: if ``resample = True``, but ``sampling_rate = None``

    """
    def __init__(
            self,
            *,
            process_func: typing.Callable[..., typing.Any] = None,
            sampling_rate: int = None,
            resample: bool = False,
            segment: Segment = None,
            keep_nat: bool = False,
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
        self.segment = segment
        self.keep_nat = keep_nat
        self.num_workers = num_workers
        self.multiprocessing = multiprocessing
        self.verbose = verbose
        if process_func is None:
            def process_func(signal, _):
                return signal
        self.process_func = process_func
        self.process_func_kwargs = kwargs
        if resample:
            self.resample = audsp.Resample(
                target_rate=sampling_rate,
                quality=audsp.define.ResampleQuality.HIGH,
            )
        else:
            self.resample = None

    def _process_file(
            self,
            file: str,
            *,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
            channel: int = None,
    ) -> pd.Series:

        signal, sampling_rate = self.read_audio(
            file,
            channel=channel,
            start=start,
            end=end,
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

        y.index = y.index.set_levels([[start], [end]], [1, 2])

        return y

    def process_file(
            self,
            file: str,
            *,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
            channel: int = None,
    ) -> pd.Series:
        r"""Process the content of an audio file.

        Args:
            file: file path
            start: start processing at this position
            end: end processing at this position
            channel: channel number

        Returns:
            Series with processed file in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        if self.segment is not None:
            index = self.segment.process_file(
                file, start=start, end=end, channel=channel,
            )
            return self.process_unified_format_index(
                index=index, channel=channel,
            )
        else:
            return self._process_file(
                file, start=start, end=end, channel=channel,
            )

    def process_files(
            self,
            files: typing.Sequence[str],
            *,
            starts: typing.Sequence[pd.Timedelta] = None,
            ends: typing.Sequence[pd.Timedelta] = None,
            channel: int = None,
    ) -> pd.Series:
        r"""Process a list of files.

        Args:
            files: list of file paths
            channel: channel number
            starts: list with start positions
            ends: list with end positions

        Returns:
            Series with processed files in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        if starts is None:
            starts = [None] * len(files)
        if ends is None:
            ends = [None] * len(files)

        params = [
            (
                (file, ),
                {'start': start, 'end': end, 'channel': channel},
            ) for file, start, end in zip(files, starts, ends)
        ]
        y = audeer.run_tasks(
            self.process_file,
            params,
            num_workers=self.num_workers,
            multiprocessing=self.multiprocessing,
            progress_bar=self.verbose,
            task_description=f'Process {len(files)} files',
        )
        return pd.concat(y)

    def process_folder(
            self,
            root: str,
            *,
            channel: int = None,
            filetype: str = 'wav',
    ) -> pd.Series:
        r"""Process files in a folder.

        .. note:: At the moment does not scan in sub-folders!

        Args:
            root: root folder
            channel: channel number
            filetype: file extension

        Returns:
            Series with processed files in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        files = audeer.list_file_names(root, filetype=filetype)
        files = [os.path.join(root, os.path.basename(f)) for f in files]
        return self.process_files(files, channel=channel)

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
            end = pd.to_timedelta(
                signal.shape[-1] / sampling_rate, unit='sec'
            )
        start_i, end_i = utils.segment_to_indices(
            signal, sampling_rate, start, end,
        )

        # Trim signal and possibly resample
        signal = signal[:, start_i:end_i]
        signal, sampling_rate = self._resample(
            signal,
            sampling_rate,
        )

        # Process signal
        y = self.process_func(
            signal,
            sampling_rate,
            **self.process_func_kwargs,
        )

        # Create index
        if file is not None:
            index = pd.MultiIndex.from_tuples(
                [(file, start, end)], names=['file', 'start', 'end']
            )
        else:
            index = pd.MultiIndex.from_tuples(
                [(start, end)], names=['start', 'end']
            )

        return pd.Series([y], index)

    def process_signal(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            file: str = None,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
    ) -> typing.Any:
        r"""Process audio signal and return result.

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
            Series with processed signal in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        if self.segment is not None:
            index = self.segment.process_signal(
                signal, sampling_rate, file=file, start=start, end=end,
            )
            return self.process_signal_from_index(
                signal, sampling_rate, index,
            )
        else:
            return self._process_signal(
                signal, sampling_rate, file=file, start=start, end=end,
            )

    def process_signal_from_index(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            index: pd.MultiIndex,
    ) -> pd.Series:
        r"""Split a signal into segments and process each segment.

        .. note:: It is assumed that the index already holds segments,
            i.e. in case a ``segment`` object is given, it will be ignored.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            index: a :class:`pandas.MultiIndex` with two levels
                named `start` and `end` that hold start and end
                positions as :class:`pandas.Timedelta` objects.

        Returns:
            Series with processed segments in the Unified Format

        """
        utils.check_index(index)

        if index.empty:
            return pd.Series(None, index=index)

        if len(index.levels) == 3:
            params = [
                (
                    (signal, sampling_rate),
                    {'file': file, 'start': start, 'end': end},
                ) for file, start, end in index
            ]
        else:
            params = [
                (
                    (signal, sampling_rate),
                    {'start': start, 'end': end},
                ) for start, end in index
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

    def process_unified_format_index(
            self,
            index: pd.MultiIndex,
            *,
            channel: int = None) -> pd.Series:
        r"""Process from a segmented index conform to the `Unified Format`_.

        .. note:: Currently expects a segmented index. In the future it is
            planned to support other index types (e.g. filewise), too. Until
            then you can use audata.util.to_segmented_frame_ for conversion

        .. note:: It is assumed that the index already holds segments,
            i.e. in case a ``segment`` object is given, it will be ignored.

        Args:
            index: index with segment information
            channel: channel number

        Returns:
            Series with processed segments in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        .. _`Unified Format`: http://tools.pp.audeering.com/audata/
            data-tables.html

        .. _audata.util.to_segmented_frame: http://tools.pp.audeering.com/
            audata/api-utils.html#to-segmented-frame


        """
        utils.check_index(index)

        if index.empty:
            return pd.Series(None, index=index)

        params = [
            (
                (file, ),
                {'start': start, 'end': end, 'channel': channel},
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

    def read_audio(
            self,
            path: str,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
            channel: int = None,
    ):
        return utils.read_audio(path, start, end, channel)

    def _resample(
            self,
            signal: np.ndarray,
            sampling_rate: int,
    ) -> typing.Tuple[np.ndarray, int]:
        if (
                self.sampling_rate is not None
                and sampling_rate != self.sampling_rate
        ):
            if self.resample is not None:
                signal = self.resample(signal, sampling_rate)
                signal = np.atleast_2d(signal)
                sampling_rate = self.sampling_rate
            else:
                raise RuntimeError(
                    f'Signal sampling rate of {sampling_rate} Hz '
                    f'does not match requested model sampling rate of '
                    f'{self.sampling_rate} Hz.'
                )
        return signal, sampling_rate


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
            Must return a sequence of results for every segment.
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal.
        resample: if ``True`` enforces given sampling rate by resampling
        verbose: show debug messages
        kwargs: additional keyword arguments to the processing function

    Raises:
        ValueError: if ``resample = True``, but ``sampling_rate = None``

    """
    def __init__(
            self,
            *,
            process_func: typing.Callable[
                ...,
                typing.Sequence[typing.Any]
            ] = None,
            sampling_rate: int = None,
            resample: bool = False,
            verbose: bool = False,
            **kwargs,
    ):
        if resample and sampling_rate is None:
            raise ValueError(
                'sampling_rate has to be provided for resample = True.'
            )
        self.sampling_rate = sampling_rate
        self.verbose = verbose
        if process_func is None:
            def process_func(signal, _, starts, ends):
                return [
                    signal[:, start:end] for start, end in zip(starts, ends)
                ]
        self.process_func = process_func
        self.process_func_kwargs = kwargs
        if resample:
            self.resample = audsp.Resample(
                target_rate=sampling_rate,
                quality=audsp.define.ResampleQuality.HIGH,
            )
        else:
            self.resample = None

    def process_signal_from_index(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            index: pd.MultiIndex,
    ) -> pd.Series:
        r"""Split a signal into segments and process each segment.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            sampling_rate:
            index: a :class:`pandas.MultiIndex` with two levels
                named `start` and `end` that hold start and end
                positions as :class:`pandas.Timedelta` objects.

        Returns:
            Series with processed segments in the Unified Format

        """
        utils.check_index(index)

        signal = np.atleast_2d(signal)
        signal, sampling_rate = self._resample(signal, sampling_rate)

        # Process signal
        starts_i, ends_i = utils.segments_to_indices(
            signal, sampling_rate, index,
        )
        y = self.process_func(signal, sampling_rate, starts_i, ends_i)

        return pd.Series(y, index=index)

    def process_unified_format_index(
            self,
            index: pd.MultiIndex,
            channel: int = None) -> pd.Series:
        r"""Process from a segmented index conform to the `Unified Format`_.

        Args:
            index: index with segment information
            channel: channel number

        Returns:
            Series with processed segments in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        .. _`Unified Format`: http://tools.pp.audeering.com/audata/
            data-tables.html

        """
        if not index.names == ('file', 'start', 'end'):
            raise ValueError('Not a segmented index conform to Unified Format')

        if index.empty:
            return pd.Series(index=index)

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
                signal, sampling_rate = self.read_audio(file, channel=channel)
                ys[idx] = pd.Series(
                    self.process_signal_from_index(
                        signal, sampling_rate, select,
                    ).values,
                    index=index[mask],
                )

        return pd.concat(ys)

    def read_audio(
            self,
            path: str,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
            channel: int = None,
    ):
        return utils.read_audio(path, start, end, channel)

    def _resample(
            self,
            signal: np.ndarray,
            sampling_rate: int,
    ) -> typing.Tuple[np.ndarray, int]:
        if (
                self.sampling_rate is not None
                and sampling_rate != self.sampling_rate
        ):
            if self.resample is not None:
                signal = self.resample(signal, sampling_rate)
                signal = np.atleast_2d(signal)
                sampling_rate = self.sampling_rate
            else:
                raise RuntimeError(
                    f'Signal sampling rate of {sampling_rate} Hz '
                    f'does not match requested model sampling rate of '
                    f'{self.sampling_rate} Hz.'
                )
        return signal, sampling_rate
