import os
import typing

import audiofile as af
import numpy as np
import pandas as pd

import audeer

from audinterface.core.process import Process
from audinterface.core.segment import Segment


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
        name: name of feature extractor, e.g. ``'stft'``
        feature_names: features are stored as columns in a data frame,
            this defines the names of the columns
        feature_params: important parameters used with feature extractor,
            e.g. ``{'win_size': 512, 'hop_size': 256}``.
            With the parameters you can differentiate between
            different settings of the same feature extarctor
        process_func: feature extraction function,
            which expects the two positional arguments ``signal``
            and ``sampling_rate``
            and any number of additional keyword arguments.
            The function must return features in the shape of
            ``(num_channels, num_features)``
            or ``(num_channels, num_features, num_time_steps)``.
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal.
        num_channels: the expected number of channels
        win_dur: window size in ``unit``,
            if features are extracted with a sliding window
        hop_dur: hop size in ``unit``,
            if features are extracted with a sliding window.
            This defines the shift between two windows.
            Defaults to ``win_dur / 2``.
        unit: unit of ``win_dur`` and ``hop_dur``.
            Can be ``'samples'``,
            or any unit supported by :class:`pd.timedelta`
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
        ValueError: if ``unit == 'samples'``, ``sampling_rate is None``
            and ``win_dur is not None``
        ValueError: if ``hop_dur`` is specified, but not ``win_dur``

    """
    def __init__(
            self,
            feature_names: typing.Sequence[str],
            *,
            feature_params: typing.Dict = {},
            process_func: typing.Callable[..., np.ndarray] = None,
            sampling_rate: int = None,
            num_channels: int = 1,
            win_dur: typing.Union[int, float] = None,
            hop_dur: typing.Union[int, float] = None,
            unit: str = 'seconds',
            resample: bool = False,
            segment: Segment = None,
            keep_nat: bool = False,
            num_workers: typing.Optional[int] = 1,
            multiprocessing: bool = False,
            verbose: bool = False,
            **kwargs,
    ):
        if win_dur is None and hop_dur is not None:
            raise ValueError(
                "You have to specify 'win_dur' if 'hop_dur' is given"
            )
        if unit == 'samples' and sampling_rate is None and win_dur is not None:
            raise ValueError(
                "You have specified 'samples' as unit, "
                "but haven't provided a sampling rate"
            )

        if process_func is None:
            def process_func(signal, _):
                return np.zeros(
                    (num_channels, len(feature_names)),
                    dtype=np.float,
                )
        self.process = Process(
            process_func=process_func,
            sampling_rate=sampling_rate,
            resample=resample,
            segment=segment,
            keep_nat=keep_nat,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
            verbose=verbose,
            **kwargs,
        )
        self.feature_params = feature_params
        self.num_channels = num_channels
        self.num_features = len(feature_names)
        self.feature_names = list(feature_names)
        if num_channels > 1:
            self.column_names = []
            for channel in range(num_channels):
                self.column_names.extend(
                    [f'{name}-{channel}' for name in feature_names]
                )
        else:
            self.column_names = self.feature_names
        if win_dur is None:
            self.win_dur = None
            self.hop_dur = None
        else:
            if hop_dur is None:
                hop_dur = win_dur // 2 if unit == 'samples' else win_dur / 2
            if unit == 'samples':
                unit = 'seconds'
                win_dur = win_dur / sampling_rate
                hop_dur = hop_dur / sampling_rate
            self.win_dur = pd.to_timedelta(win_dur, unit=unit)
            self.hop_dur = pd.to_timedelta(hop_dur, unit=unit)
        self.verbose = verbose

    def process_file(
            self,
            file: str,
            *,
            start: pd.Timedelta = None,
            end: pd.Timedelta = None,
            channel: int = None,
    ) -> pd.DataFrame:
        r"""Extract features from an audio file.

        Args:
            file: file path
            channel: channel number
            start: start processing at this position
            end: end processing at this position

        Raises:
            RuntimeError: if sampling rates of feature extracted
                and signal do not match
            RuntimeError: if number of channels do not match

        """
        series = self.process.process_file(
            file, start=start, end=end, channel=channel,
        )
        return self._series_to_frame(series)

    def process_files(
            self,
            files: typing.Sequence[str],
            *,
            starts: typing.Sequence[pd.Timedelta] = None,
            ends: typing.Sequence[pd.Timedelta] = None,
            channel: int = None,
    ) -> pd.DataFrame:
        r"""Extract features for a list of files.

        Args:
            files: list of file paths
            starts: list with start positions
            ends: list with end positions
            channel: channel number

        Raises:
            RuntimeError: if sampling rates of feature extracted
                and signal do not match
            RuntimeError: if number of channels do not match

        """
        series = self.process.process_files(
            files, starts=starts, ends=ends, channel=channel,
        )
        return self._series_to_frame(series)

    def process_folder(
            self,
            root: str,
            *,
            filetype: str = 'wav',
            channel: int = None,
    ) -> pd.DataFrame:
        r"""Extract features from files in a folder.

        .. note:: At the moment does not scan in sub-folders!

        Args:
            root: root folder
            filetype: file extension
            channel: channel number

        Raises:
            RuntimeError: if sampling rates of feature extracted
                and signal do not match
            RuntimeError: if number of channels do not match

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
    ) -> pd.DataFrame:
        r"""Extract features for an audio signal.

        .. note:: If a ``file`` is given, the index of the returned frame
            has levels ``file``, ``start`` and ``end``. Otherwise,
            it consists only of ``start`` and ``end``.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            file: file path
            start: start processing at this position
            end: end processing at this position

        Raises:
            RuntimeError: if sampling rates of feature extractor
                and signal do not match
            RuntimeError: if dimension of extracted features
                is greater than three
            RuntimeError: if feature extractor uses sliding window,
                but ``self.win_dur`` is not specified
            RuntimeError: ifnumber of features does not match
                number of feature names
            RuntimeError: if number of channels do not match

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

        .. note:: It is assumed that the index already holds segments,
            i.e. in case a ``segment`` object is given, it will be ignored.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            index: a :class:`pandas.MultiIndex` with two levels
                named `start` and `end` that hold start and end
                positions as :class:`pandas.Timedelta` objects.

        Raises:
            RuntimeError: if sampling rates of feature extracted
                and signal do not match
            RuntimeError: if number of channels do not match
            ValueError: if given index has wrong format

        """
        series = self.process.process_signal_from_index(
            signal, sampling_rate, index,
        )
        return self._series_to_frame(series)

    def process_unified_format_index(
            self,
            index: pd.MultiIndex,
            *,
            channel: int = None,
    ) -> pd.DataFrame:
        r"""Extract features from a `Unified Format`_ segmented index.

        .. note:: Currently expects a segmented index. In the future it is
            planned to support other index types (e.g. filewise), too. Until
            then you can use audata.util.to_segmented_frame_ for conversion

        .. note:: It is assumed that the index already holds segments,
            i.e. in case a ``segment`` object is given, it will be ignored.

        Args:
            index: index with segment information
            channel: channel number (default 0)

        Raises:
            RuntimeError: if sampling rates of feature extractor
                and signal do not match
            RuntimeError: if number of channels do not match
            ValueError: if index is not conform to the `Unified Format`_

        .. _`Unified Format`: http://tools.pp.audeering.com/audata/
             data-tables.html

        .. _audata.util.to_segmented_frame: http://tools.pp.audeering.com/
            audata/api-utils.html#to-segmented-frame


        """
        series = self.process.process_unified_format_index(
            index, channel=channel,
        )
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

        # Force third time step dimension
        features = np.atleast_3d(features)
        if features.ndim > 3:
            raise RuntimeError(
                f'Dimension of extracted features must be not greater than 3. '
                f'Yours is {features.ndim}'
            )
        n_channels = features.shape[0]
        n_features = features.shape[1]
        n_time_steps = features.shape[2]

        # n_features + 1 as we added 'channel' to feature names
        if n_channels != self.num_channels:
            raise RuntimeError(
                f'Channel number {n_channels} does not match '
                f'number of expected channels: {self.num_channels}'
            )
        if n_features * n_channels != len(self.column_names):
            raise RuntimeError(
                f'Feature names {self.column_names} do not match '
                f'number of extracted features: {n_features * n_channels}'
            )

        # Reshape features and store channel number as first feature
        # [n_channels, n_features, n_time_steps] =>
        # [n_channels * n_features + 1, n_time_steps]
        new_shape = (n_channels * n_features, n_time_steps)
        features = features.reshape(new_shape).T

        if n_time_steps > 1:
            if self.win_dur is None:
                starts = [start] * n_time_steps
                ends = [end] * n_time_steps
            else:
                starts = pd.timedelta_range(
                    start,
                    freq=self.hop_dur,
                    periods=n_time_steps,
                )
                ends = starts + self.win_dur
        else:
            starts = [start]
            ends = [end]

        if file is None:
            index = pd.MultiIndex.from_arrays(
                [starts, ends],
                names=['start', 'end'],
            )
        else:
            files = [file] * len(starts)
            index = pd.MultiIndex.from_arrays(
                [files, starts, ends],
                names=['file', 'start', 'end'],
            )

        return pd.DataFrame(features, index, columns=self.column_names)

    def __call__(
            self,
            signal: np.ndarray,
            sampling_rate: int,
    ) -> np.ndarray:
        r"""Apply processing to signal.

        This function processes the signal **without** transforming the output
        into a :class:`pd.DataFrame`. Instead it will return the raw processed
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
