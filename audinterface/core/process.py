import os
import typing

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
            and any number of additional keyword arguments.
        process_func_is_mono: if set to ``True`` and the input signal
            has multiple channels, ``process_func`` will be applied to
            every channel individually
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal.
        resample: if ``True`` enforces given sampling rate by resampling
        channels: channel selection, see :func:`audresample.remix`
        mixdown: apply mono mix-down on selection
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
            process_func_is_mono: bool = False,
            sampling_rate: int = None,
            resample: bool = False,
            channels: typing.Union[int, typing.Sequence[int]] = None,
            mixdown: bool = False,
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
        self.process_func_kwargs = kwargs
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
                If value is as a float or integer it is treated as seconds
            end: end processing at this position.
                If value is as a float or integer it is treated as seconds
            root: root folder to expand relative file path

        Returns:
            Series with processed file conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        start = utils.to_timedelta(start)
        end = utils.to_timedelta(end)
        if self.segment is not None:
            index = self.segment.process_file(
                file,
                start=start,
                end=end,
                root=root,
            )
            return self._process_index_wo_segment(index, root)
        else:
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
                If a scalar is given, it is applied to all files
            ends: segment end positions.
                Time values given as float or integers are treated as seconds
                If a scalar is given, it is applied to all files
            root: root folder to expand relative file paths

        Returns:
            Series with processed files conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        if isinstance(starts, (type(None), float, int, str, pd.Timedelta)):
            starts = [starts] * len(files)
        if isinstance(ends, (type(None), float, int, str, pd.Timedelta)):
            ends = [ends] * len(files)

        starts = utils.to_timedelta(starts)
        ends = utils.to_timedelta(ends)

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
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
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
            return pd.Series(None, index=index, dtype=float)

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
    ) -> pd.Series:
        r"""Process from an index conform to audformat_.

        Args:
            index: index with segment information
            root: root folder to expand relative file paths

        Returns:
            Series with processed segments conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        index = audformat.utils.to_segmented_index(index)

        if self.segment is not None:
            index = self.segment.process_index(index, root=root)

        return self._process_index_wo_segment(index, root)

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

        # Trim and process signal
        y = self(signal[:, start_i:end_i], sampling_rate)

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
                If value is as a float or integer it is treated as seconds
            end: end processing at this position.
                If value is as a float or integer it is treated as seconds

        Returns:
            Series with processed signal conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        start = utils.to_timedelta(start)
        end = utils.to_timedelta(end)
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
            return pd.Series(None, index=index, dtype=float)

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
            return pd.Series(None, index=index, dtype=float)

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

    @audeer.deprecated(
        removal_version='0.8.0',
        alternative='process_index',
    )
    def process_unified_format_index(
            self,
            index: pd.Index,
    ) -> pd.Series:  # pragma: nocover
        r"""Process from an index conform to the `Unified Format`_.

        .. note:: It is assumed that the index already holds segments,
            i.e. in case a ``segment`` object is given, it will be ignored.

        Args:
            index: index with segment information

        Returns:
            Series with processed segments in the Unified Format

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _`Unified Format`: http://tools.pp.audeering.com/audata/
            data-tables.html

        """

        index = audformat.utils.to_segmented_index(index)
        utils.assert_index(index)

        if index.empty:
            return pd.Series(None, index=index, dtype=float)

        params = [
            (
                (file, ),
                {
                    'start': start,
                    'end': end,
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
        if self.process_func_is_mono and signal.shape[0] > 1:
            return [
                self.process_func(
                    np.atleast_2d(channel),
                    sampling_rate,
                    **self.process_func_kwargs,
                ) for channel in signal
            ]
        return self.process_func(
            signal,
            sampling_rate,
            **self.process_func_kwargs,
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
            Must return a sequence of results for every segment.
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal.
        resample: if ``True`` enforces given sampling rate by resampling
        channels: channel selection, see :func:`audresample.remix`
        mixdown: apply mono mix-down on selection
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
        self.process_func_kwargs = kwargs
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

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        index = audformat.utils.to_segmented_index(index)

        if index.empty:
            return pd.Series(index=index, dtype=float)

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
            ValueError: if index contains duplicates

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        utils.assert_index(index)

        starts_i, ends_i = utils.segments_to_indices(
            signal, sampling_rate, index,
        )
        y = self(signal, sampling_rate, starts_i, ends_i)

        return pd.Series(y, index=index)

    @audeer.deprecated(
        removal_version='0.8.0',
        alternative='process_index',
    )
    def process_unified_format_index(
            self,
            index: pd.Index,
    ) -> pd.Series:  # pragma: nocover
        r"""Process from a index conform to the `Unified Format`_.

        Args:
            index: index with segment information

        Returns:
            Series with processed segments in the Unified Format

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        """
        index = audformat.utils.to_segmented_index(index)

        if index.empty:
            return pd.Series(index=index, dtype=float)

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
                signal, sampling_rate = utils.read_audio(file)
                ys[idx] = pd.Series(
                    self.process_signal_from_index(
                        signal, sampling_rate, select,
                    ).values,
                    index=index[mask],
                )

        return pd.concat(ys)

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
            **self.process_func_kwargs,
        )
