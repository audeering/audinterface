import os
import typing

import numpy as np
import pandas as pd

import audeer
import audsp

import audinterface.core.utils as utils


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
            def process_func(signal, _):
                return signal
        self.process_func = process_func
        self.process_func_kwargs = kwargs
        if resample:
            self.resample = audsp.Resample(
                sampling_rate=sampling_rate,
                quality=audsp.define.ResampleQuality.HIGH,
            )
        else:
            self.resample = None

    def process_file(
            self,
            file: str,
            *,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
            channel: int = None,
    ) -> typing.Any:
        r"""Process the content of an audio file.

        Args:
            file: file path
            channel: channel number
            start: start processing at this position
            end: end processing at this position

        Returns:
            Output of process function

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        signal, sampling_rate = self.read_audio(
            file,
            channel=channel,
            start=start,
            end=end,
        )
        return self.process_signal(
            signal,
            sampling_rate,
        )

    def process_files(
            self,
            files: typing.Sequence[str],
            *,
            channel: int = None,
    ) -> pd.Series:
        r"""Process a list of files.

        Args:
            files: list of file paths
            channel: channel number

        Returns:
            Dictionary mapping files to output of process function

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        data = [None] * len(files)
        with audeer.progress_bar(
                files,
                total=len(files),
                disable=not self.verbose,
        ) as pbar:
            for idx, file in enumerate(pbar):
                desc = audeer.format_display_message(file, pbar=True)
                pbar.set_description(desc, refresh=True)
                data[idx] = self.process_file(file, channel=channel)
        return pd.Series(data, index=pd.Index(files, name='file'))

    def process_folder(
            self,
            root: str,
            *,
            filetype: str = 'wav',
            channel: int = None,
    ) -> pd.Series:
        r"""Process files in a folder.

        .. note:: At the moment does not scan in sub-folders!

        Args:
            root: root folder
            filetype: file extension
            channel: channel number

        Returns:
            Dictionary mapping files to output of process function

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
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
    ) -> typing.Any:
        r"""Process audio signal and return result.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            start: start processing at this position
            end: end processing at this position

        Returns:
            Output of process function

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        """
        # Enforce 2D signals
        signal = np.atleast_2d(signal)

        # Resample signal
        if (
                self.sampling_rate is not None
                and sampling_rate != self.sampling_rate
        ):
            if self.resample is not None:
                # TODO: support stereo, see
                # https://gitlab.audeering.com/tools/pyaudsp/-/issues/20
                signal = self.resample(signal, sampling_rate)
                signal = np.atleast_2d(signal)
                sampling_rate = self.sampling_rate
            else:
                raise RuntimeError(
                    f'Signal sampling rate of {sampling_rate} Hz '
                    f'does not match requested model sampling rate of '
                    f'{self.sampling_rate} Hz.'
                )

        # Find start and end index
        start_i, end_i = utils.segment_to_indices(signal, sampling_rate,
                                                  start, end)

        # Process signal
        return self.process_func(
            signal[:, start_i:end_i],
            sampling_rate,
            **self.process_func_kwargs,
        )

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

        y = [None] * len(index)

        with audeer.progress_bar(
                index,
                total=len(index),
                disable=not self.verbose,
        ) as pbar:
            for idx, (start, end) in enumerate(pbar):
                desc = audeer.format_display_message(
                    'f{start} - {end}',
                    pbar=True,
                )
                pbar.set_description(desc, refresh=True)
                y[idx] = self.process_signal(signal, sampling_rate,
                                             start=start, end=end)

        return pd.Series(y, index=index)

    def process_unified_format_index(
            self,
            index: pd.MultiIndex,
            *,
            channel: int = None) -> pd.Series:
        r"""Process from a segmented index conform to the `Unified Format`_.

        .. note:: Currently expects a segmented index. In the future it is
            planned to support other index types (e.g. filewise), too. Until
            then you can use audata.util.to_segmented_frame_ for conversion

        Args:
            index: index with segment information
            channel: channel number (default 0)

        Returns:
            Series with processed segments in the Unified Format

        Raises:
            RuntimeError: if sampling rates of model and signal do not match

        .. _`Unified Format`: http://tools.pp.audeering.com/audata/
            data-tables.html

        .. _audata.util.to_segmented_frame: http://tools.pp.audeering.com/
            audata/api-utils.html#to-segmented-frame


        """
        if not index.names == ('file', 'start', 'end'):
            raise ValueError('Not a segmented index conform to Unified Format')

        y = [None] * len(index)

        with audeer.progress_bar(
                index,
                total=len(index),
                disable=not self.verbose,
        ) as pbar:
            for idx, (file, start, end) in enumerate(pbar):
                desc = audeer.format_display_message(file, pbar=True)
                pbar.set_description(desc, refresh=True)
                y[idx] = self.process_file(file, channel=channel, start=start,
                                           end=end)

        return pd.Series(y, index=index)

    def read_audio(
            self,
            path: str,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
            channel: int = None,
    ):
        return utils.read_audio(path, start, end, channel)


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
                sampling_rate=sampling_rate,
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

        # Enforce 2D signals
        signal = np.atleast_2d(signal)

        # Resample signal
        if (
                self.sampling_rate is not None
                and sampling_rate != self.sampling_rate
        ):
            if self.resample is not None:
                # TODO: support stereo, see
                # https://gitlab.audeering.com/tools/pyaudsp/-/issues/20
                signal = self.resample(signal, sampling_rate)
                signal = np.atleast_2d(signal)
                sampling_rate = self.sampling_rate
            else:
                raise RuntimeError(
                    f'Signal sampling rate of {sampling_rate} Hz '
                    f'does not match requested model sampling rate of '
                    f'{self.sampling_rate} Hz.'
                )

        # Process signal
        starts_i, ends_i = utils.segments_to_indices(
            signal, sampling_rate, index,
        )
        y = self.process_func(signal, sampling_rate, starts_i, ends_i)

        return pd.Series(y, index=index)

    def process_unified_format_index(
            self,
            index: pd.MultiIndex,
            *,
            channel: int = None) -> pd.Series:
        r"""Process from a segmented index conform to the `Unified Format`_.

        Args:
            index: index with segment information
            channel: channel number (default 0)

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
