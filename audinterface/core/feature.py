import os
import typing

import numpy as np
import pandas as pd

import audeer
import audformat

from audinterface.core.process import Process
from audinterface.core.segment import Segment
from audinterface.core.typing import (
    Timestamp,
    Timestamps,
)
import audinterface.core.utils as utils


class Feature:
    r"""Feature extraction interface.

    The features are returned as a :class:`pd.DataFrame`.
    If your input signal is of size ``(num_channels, num_time_steps)``,
    the returned dataframe will have ``num_channels * num_features``
    columns.
    It will have one row per file or signal.
    If features are extracted using a sliding window,
    each window will be stored as one row.
    If ``win_dur`` is specified ``start`` and ``end`` indices
    are referred from the original ``start`` and ``end`` arguments
    and the window positions.
    Otherwise, the original ``start`` and ``end`` indices
    are kept.

    Args:
        feature_names: features are stored as columns in a data frame,
            ``feature_names`` defines the names of the columns.
            If ``num_channels`` > 1,
            the channel number will be appended to the column names.
        name: name of the feature set, e.g. ``'stft'``
        params: parameters that describe the feature set,
            e.g. ``{'win_size': 512, 'hop_size': 256, 'num_fft': 512}``.
            With the parameters you can differentiate different flavors of
            the same feature set
        process_func: feature extraction function,
            which expects the two positional arguments ``signal``
            and ``sampling_rate``
            and any number of additional keyword arguments.
            The function must return features in the shape of
            ``(num_channels, num_features)``
            or ``(num_channels, num_features, num_time_steps)``.
        process_func_is_mono: apply ``process_func`` to every channel
            individually
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal.
        win_dur: window size in ``unit``,
            if features are extracted with a sliding window
        hop_dur: hop size in ``unit``,
            if features are extracted with a sliding window.
            This defines the shift between two windows.
            Defaults to ``win_dur / 2``.
        unit: unit of ``win_dur`` and ``hop_dur``.
            Can be ``'samples'``,
            or any unit supported by :func:`pandas.to_timedelta`
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
        ValueError: if ``unit == 'samples'``, ``sampling_rate is None``
            and ``win_dur is not None``
        ValueError: if ``hop_dur`` is specified, but not ``win_dur``

    """
    def __init__(
            self,
            feature_names: typing.Sequence[str],
            *,
            name: str = None,
            params: typing.Dict = None,
            process_func: typing.Callable[..., np.ndarray] = None,
            process_func_is_mono: bool = False,
            sampling_rate: int = None,
            win_dur: typing.Union[int, float] = None,
            hop_dur: typing.Union[int, float] = None,
            unit: str = 'seconds',
            resample: bool = False,
            channels: typing.Union[int, typing.Sequence[int]] = 0,
            mixdown: bool = False,
            segment: Segment = None,
            keep_nat: bool = False,
            num_workers: typing.Optional[int] = 1,
            multiprocessing: bool = False,
            verbose: bool = False,
            **kwargs,
    ):

        if win_dur is None and hop_dur is not None:
            raise ValueError(
                "You have to specify 'win_dur' if 'hop_dur' is given."
            )
        if unit == 'samples' and sampling_rate is None and win_dur is not None:
            raise ValueError(
                "You have specified 'samples' as unit, "
                "but haven't provided a sampling rate."
            )

        if process_func is None:
            def process_func(signal, _):
                return np.zeros(
                    (num_channels, len(feature_names)),
                    dtype=np.float,
                )

        if mixdown or isinstance(channels, int):
            num_channels = 1
        else:
            num_channels = len(channels)

        self.process = Process(
            process_func=process_func,
            process_func_is_mono=process_func_is_mono,
            sampling_rate=sampling_rate,
            resample=resample,
            channels=channels,
            mixdown=mixdown,
            segment=segment,
            keep_nat=keep_nat,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
            verbose=verbose,
            **kwargs,
        )
        r"""Processing object."""
        self.name = name
        r"""Name of the feature set."""
        self.params = params
        r"""Dictionary of parameters describing the feature set."""
        self.num_channels = num_channels
        r"""Expected number of channels"""
        self.num_features = len(feature_names)
        r"""Number of features."""
        self.feature_names = list(feature_names)
        r"""Feature names."""
        self.column_names = None
        r"""Feature column names."""
        if num_channels > 1:
            self.column_names = []
            for channel in range(num_channels):
                self.column_names.extend(
                    [f'{name}-{channel}' for name in feature_names]
                )
        else:
            self.column_names = self.feature_names
        self.win_dur = win_dur
        r"""Window duration."""
        self.hop_dur = hop_dur
        r"""Hop duration."""
        if win_dur is not None and hop_dur is None:
            self.hop_dur = win_dur // 2 if unit == 'samples' else win_dur / 2
        self.unit = unit
        r"""Unit of ``win_dur`` and ``hop dur``"""
        self.verbose = verbose
        r"""Show debug messages."""

    def process_file(
            self,
            file: str,
            *,
            start: Timestamp = None,
            end: Timestamp = None,
            root: str = None,
    ) -> pd.DataFrame:
        r"""Extract features from an audio file.

        Args:
            file: file path
            start: start processing at this position.
                If value is as a float or integer it is treated as seconds
            end: end processing at this position.
                If value is as a float or integer it is treated as seconds
            root: root folder to expand relative file path

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if multiple frames are returned,
                but ``win_dur`` is not set

        """
        series = self.process.process_file(
            file,
            start=start,
            end=end,
            root=root,
        )
        return self._series_to_frame(series)

    def process_files(
            self,
            files: typing.Sequence[str],
            *,
            starts: Timestamps = None,
            ends: Timestamps = None,
            root: str = None,
    ) -> pd.DataFrame:
        r"""Extract features for a list of files.

        Args:
            files: list of file paths
            starts: segment start positions.
                Time values given as float or integers are treated as seconds.
                If a scalar is given, it is applied to all files
            ends: segment end positions.
                Time values given as float or integers are treated as seconds
                If a scalar is given, it is applied to all files
            root: root folder to expand relative file paths

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if multiple frames are returned,
                but ``win_dur`` is not set

        """
        series = self.process.process_files(
            files,
            starts=starts,
            ends=ends,
            root=root,
        )
        return self._series_to_frame(series)

    def process_folder(
            self,
            root: str,
            *,
            filetype: str = 'wav',
    ) -> pd.DataFrame:
        r"""Extract features from files in a folder.

        .. note:: At the moment does not scan in sub-folders!

        Args:
            root: root folder
            filetype: file extension

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if multiple frames are returned,
                but ``win_dur`` is not set

        """
        files = audeer.list_file_names(root, filetype=filetype)
        files = [os.path.join(root, os.path.basename(f)) for f in files]
        return self.process_files(files)

    def process_index(
            self,
            index: pd.Index,
            *,
            root: str = None,
    ) -> pd.DataFrame:
        r"""Extract features from an index conform to audformat_.

        Args:
            index: index with segment information
            root: root folder to expand relative file paths

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if multiple frames are returned,
                but ``win_dur`` is not set
            ValueError: if index is not conform to audformat_

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        series = self.process.process_index(index, root=root)
        return self._series_to_frame(series)

    def process_signal(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            file: str = None,
            start: Timestamp = None,
            end: Timestamp = None,
    ) -> pd.DataFrame:
        r"""Extract features for an audio signal.

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

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if dimension of extracted features
                is greater than three
            RuntimeError: if feature extractor uses sliding window,
                but ``self.win_dur`` is not specified
            RuntimeError: if number of features does not match
                number of feature names
            RuntimeError: if multiple frames are returned,
                but ``win_dur`` is not set

        """
        series = self.process.process_signal(
            signal,
            sampling_rate,
            file=file,
            start=start,
            end=end,
        )
        return self._series_to_frame(series)

    def process_signal_from_index(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            index: pd.MultiIndex,
    ) -> pd.DataFrame:
        r"""Split a signal into segments and extract features for each segment.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            index: a :class:`pandas.MultiIndex` with two levels
                named `start` and `end` that hold start and end
                positions as :class:`pandas.Timedelta` objects.
                See also :func:`audinterface.utils.signal_index`

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if multiple frames are returned,
                but ``win_dur`` is not set
            ValueError: if index contains duplicates

        """
        series = self.process.process_signal_from_index(
            signal,
            sampling_rate,
            index,
        )
        return self._series_to_frame(series)

    @audeer.deprecated(
        removal_version='0.8.0',
        alternative='process_index',
    )
    def process_unified_format_index(
            self,
            index: pd.Index,
    ) -> pd.DataFrame:  # pragma: nocover
        r"""Extract features from an index conform to the `Unified Format`_.

        .. note:: It is assumed that the index already holds segments,
            i.e. in case a ``segment`` object is given, it will be ignored.

        Args:
            index: index with segment information

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if multiple frames are returned,
                but ``win_dur`` is not set
            ValueError: if index is not conform to the `Unified Format`_

        .. _`Unified Format`: http://tools.pp.audeering.com/audata/
             data-tables.html

        """
        series = self.process.process_unified_format_index(index)
        return self._series_to_frame(series)

    def to_numpy(
            self,
            frame: pd.DataFrame,
    ) -> np.ndarray:
        r"""Return feature values as a :class:`numpy.ndarray` in original
        shape, i.e. ``(channels, features, time)``.

        Args:
            frame: feature frame

        """
        return frame.values.T.reshape(self.num_channels, self.num_features, -1)

    def _series_to_frame(
            self,
            series: pd.Series,
    ) -> pd.DataFrame:

        if series.empty:
            return pd.DataFrame(
                columns=self.column_names,
            )

        frames = [None] * len(series)
        if len(series.index.levels) == 3:
            for idx, ((file, start, end), values) in enumerate(series.items()):
                frames[idx] = self._values_to_frame(
                    values, file=file, start=start, end=end,
                )
        else:
            for idx, ((start, end), values) in enumerate(series.items()):
                frames[idx] = self._values_to_frame(
                    values, start=start, end=end,
                )
        return pd.concat(frames, axis='index')

    def _values_to_frame(
            self,
            features: np.ndarray,
            start: pd.Timedelta,
            end: pd.Timedelta,
            *,
            file: str = None,
    ) -> pd.DataFrame:

        # Convert features to a pd.DataFrame
        # Assumed format
        # [n_channels, n_features, n_time_steps]
        # or
        # [n_channels, n_features]

        if self.win_dur is not None:
            if self.unit == 'samples':
                win_dur = pd.to_timedelta(
                    self.win_dur / self.process.sampling_rate, unit='seconds',
                )
                hop_dur = pd.to_timedelta(
                    self.hop_dur / self.process.sampling_rate, unit='seconds',
                )
            else:
                win_dur = pd.to_timedelta(self.win_dur, unit=self.unit)
                hop_dur = pd.to_timedelta(self.hop_dur, unit=self.unit)
        else:
            win_dur = None
            hop_dur = None

        if self.process.process_func_is_mono and self.num_channels > 1:
            features = np.concatenate(features)

        if not isinstance(features, np.ndarray):
            raise RuntimeError(
                "Features must be a 'np.ndarray', "
                f"not '{type(features)}'."
            )

        # features = np.array(features)
        if features.ndim < 2 or features.ndim > 3:
            raise RuntimeError(
                f'Dimension of extracted features must be 2 or 3, '
                f'not {features.ndim}.'
            )

        # Force third time step dimension
        features = np.atleast_3d(features)
        n_channels = features.shape[0]
        n_features = features.shape[1]
        n_time_steps = features.shape[2]

        if n_channels != self.num_channels:
            raise RuntimeError(
                f'Number of channels must be {self.num_channels}, '
                f'not {n_channels}.'
            )
        if n_features != len(self.feature_names):
            raise RuntimeError(
                f'Number of features must be {len(self.feature_names)}, '
                f'not {n_features}.'
            )

        # Reshape features and store channel number as first feature
        # [n_channels, n_features, n_time_steps] =>
        # [n_channels * n_features + 1, n_time_steps]
        new_shape = (n_channels * n_features, n_time_steps)
        features = features.reshape(new_shape).T

        if n_time_steps > 1:

            if win_dur is None:
                raise RuntimeError(
                    f"Got "
                    f"{n_time_steps} "
                    f"frames, but 'win_dur' is not set."
                )

            starts = pd.timedelta_range(
                start,
                freq=hop_dur,
                periods=n_time_steps,
            )
            ends = starts + win_dur
        else:
            starts = [start]
            ends = [end]

        if file is None:
            index = utils.signal_index(starts, ends)
        else:
            files = [file] * len(starts)
            index = audformat.segmented_index(files, starts, ends)

        return pd.DataFrame(features, index, columns=self.column_names)

    def __call__(
            self,
            signal: np.ndarray,
            sampling_rate: int,
    ) -> np.ndarray:
        r"""Apply processing to signal.

        This function processes the signal **without** transforming the output
        into a :class:`pd.DataFrame`. Instead it will return the raw processed
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
            RuntimeError: if multiple frames are returned,
                but ``win_dur`` is not set

        """
        y = self.process(
            signal,
            sampling_rate,
        )
        if self.process.process_func_is_mono and self.num_channels > 1:
            y = np.concatenate(y)
        return y
