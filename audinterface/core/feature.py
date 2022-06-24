import errno
import os
import typing
import warnings

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
            ``(num_features)``,
            ``(num_channels, num_features)``,
            ``(num_features, num_frames)``,
            or ``(num_channels, num_features, num_frames)``
        process_func_args: (keyword) arguments passed on to the processing
            function
        process_func_is_mono: apply ``process_func`` to every channel
            individually
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal
        resample: if ``True`` enforces given sampling rate by resampling
        channels: channel selection, see :func:`audresample.remix`
        win_dur: window size,
            if features are extracted with a sliding window.
            If value is as a float or integer
            it is treated as seconds.
            To specify a unit provide as string,
            e.g. ``'2ms'``.
            To specify in samples provide as string without unit,
            e.g. ``'2000'``
        hop_dur: hop size,
            if features are extracted with a sliding window.
            This defines the shift between two windows.
            If value is as a float or integer
            it is treated as seconds.
            To specify a unit provide as string,
            e.g. ``'2ms'``.
            To specify in samples provide a string without unit,
            e.g. ``'2000'``.
            Defaults to ``win_dur / 2``
        min_signal_dur: minimum signal duration
            required by ``process_func``.
            If value is as a float or integer
            it is treated as seconds.
            To specify a unit provide as string,
            e.g. ``'2ms'``.
            To specify in samples provide as string without unit,
            e.g. ``'2000'``
            If provided signal is shorter,
            it will be zero padded at the end
        max_signal_dur: maximum signal duraton
            required by ``process_func``.
            If value is as a float or integer
            it is treated as seconds.
            To specify a unit provide as string,
            e.g. ``'2ms'``.
            To specify in samples provide as string without unit,
            e.g. ``'2000'``
            If provided signal is longer,
            it will be cut at the end
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

    Raises:
        ValueError: if ``win_dur`` or ``hop_dur`` are given in samples
            and ``sampling_rate is None``
        ValueError: if ``hop_dur`` is specified, but not ``win_dur``

    Example:
        >>> def mean_std(signal, sampling_rate):
        ...     return [signal.mean(), signal.std()]
        >>> interface = Feature(['mean', 'std'], process_func=mean_std)
        >>> signal = np.array([1., 2., 3.])
        >>> interface(signal, sampling_rate=3)
        array([[[2.        ],
                [0.81649658]]])
        >>> interface.process_signal(signal, sampling_rate=3)
                                mean       std
        start  end
        0 days 0 days 00:00:01   2.0  0.816497
        >>> import audb
        >>> db = audb.load(
        ...     'emodb',
        ...     version='1.2.0',
        ...     media='wav/03a01Fa.wav',
        ...     full_path=False,
        ...     verbose=False,
        ... )
        >>> index = db['emotion'].index
        >>> interface.process_index(index, root=db.root)
                                                           mean       std
        file            start  end
        wav/03a01Fa.wav 0 days 0 days 00:00:01.898250 -0.000311  0.082317

    """
    def __init__(
            self,
            feature_names: typing.Union[str, typing.Sequence[str]],
            *,
            name: str = None,
            params: typing.Dict = None,
            process_func: typing.Callable[..., typing.Any] = None,
            process_func_args: typing.Dict[str, typing.Any] = None,
            process_func_is_mono: bool = False,
            sampling_rate: int = None,
            resample: bool = False,
            channels: typing.Union[int, typing.Sequence[int]] = 0,
            mixdown: bool = False,
            win_dur: Timestamp = None,
            hop_dur: Timestamp = None,
            min_signal_dur: Timestamp = None,
            max_signal_dur: Timestamp = None,
            segment: Segment = None,
            keep_nat: bool = False,
            num_workers: typing.Optional[int] = 1,
            multiprocessing: bool = False,
            verbose: bool = False,
            **kwargs,
    ):
        feature_names = audeer.to_list(feature_names)

        # ------
        # Handle deprecated 'unit' keyword argument
        def add_unit(dur, unit):
            if unit == 'samples':
                return str(dur)
            else:
                return f'{dur}{unit}'

        if 'unit' in kwargs:
            message = (
                "'unit' argument is deprecated "
                "and will be removed with version '1.2.0'."
                "The unit can now directly specified "
                "within the 'win_dur' and 'hop_dur' arguments."
            )
            warnings.warn(
                message,
                category=UserWarning,
                stacklevel=2,
            )
            unit = kwargs.pop('unit')
            if win_dur is not None:
                win_dur = add_unit(win_dur, unit)
            if hop_dur is not None:
                hop_dur = add_unit(hop_dur, unit)
        # ------

        process_func_args = process_func_args or {}
        if kwargs:
            warnings.warn(
                utils.kwargs_deprecation_warning,
                category=UserWarning,
                stacklevel=2,
            )
            for key, value in kwargs.items():
                process_func_args[key] = value

        if win_dur is None and hop_dur is not None:
            raise ValueError(
                "You have to specify 'win_dur' if 'hop_dur' is given."
            )
        if win_dur is not None and hop_dur is None:
            hop_dur = utils.to_timedelta(win_dur, sampling_rate) / 2

        if process_func is None:
            def process_func(signal, _):
                return np.zeros(
                    (num_channels, len(feature_names)),
                    dtype=object,
                )

        if mixdown or isinstance(channels, int):
            num_channels = 1
        else:
            num_channels = len(channels)

        if num_channels > 1:
            column_names = []
            for channel in range(num_channels):
                column_names.extend(
                    [f'{name}-{channel}' for name in feature_names]
                )
        else:
            column_names = feature_names

        self.process = Process(
            process_func=process_func,
            process_func_args=process_func_args,
            process_func_is_mono=process_func_is_mono,
            sampling_rate=sampling_rate,
            resample=resample,
            channels=channels,
            mixdown=mixdown,
            min_signal_dur=min_signal_dur,
            max_signal_dur=max_signal_dur,
            segment=segment,
            keep_nat=keep_nat,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
            verbose=verbose,
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
        self.feature_names = feature_names
        r"""Feature names."""
        self.column_names = column_names
        r"""Feature column names."""
        self.win_dur = win_dur
        r"""Window duration."""
        self.hop_dur = hop_dur
        r"""Hop duration."""
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
                If value is as a float or integer it is treated as seconds.
                To specify a unit provide as string,
                e.g. ``'2ms'``.
                To specify in samples provide as string without unit,
                e.g. ``'2000'``
            end: end processing at this position.
                If value is as a float or integer it is treated as seconds.
                To specify a unit provide as string,
                e.g. ``'2ms'``.
                To specify in samples provide as string without unit,
                e.g. ``'2000'``
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
                To specify a unit provide as string,
                e.g. ``'2ms'``.
                To specify in samples provide as string without unit,
                e.g. ``'2000'``.
                If a scalar is given, it is applied to all files
            ends: segment end positions.
                Time values given as float or integers are treated as seconds
                To specify a unit provide as string,
                e.g. ``'2ms'``.
                To specify in samples provide as string without unit,
                e.g. ``'2000'``.
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
            FileNotFoundError: if folder does not exist
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if multiple frames are returned,
                but ``win_dur`` is not set

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
            cache_root: str = None,
    ) -> pd.DataFrame:
        r"""Extract features from an index conform to audformat_.

        If ``cache_root`` is not ``None``,
        a hash value is created from the index
        using :func:`audformat.utils.hash` and
        the result is stored as
        ``<cache_root>/<hash>.pkl``.
        When called again with the same index,
        features will be read from the cached file.

        Args:
            index: index with segment information
            root: root folder to expand relative file paths
            cache_root: cache folder (see description)

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if multiple frames are returned,
                but ``win_dur`` is not set
            ValueError: if index is not conform to audformat_

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        cache_path = None

        if cache_root is not None:
            cache_root = audeer.mkdir(cache_root)
            hash = audformat.utils.hash(index)
            cache_path = os.path.join(cache_root, f'{hash}.pkl')

        if cache_path and os.path.exists(cache_path):
            df = pd.read_pickle(cache_path)
        else:
            y = self.process.process_index(
                index,
                root=root,
            )
            df = self._series_to_frame(y)

            if cache_path is not None:
                df.to_pickle(cache_path, protocol=4)

        return df

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
                If value is as a float or integer it is treated as seconds.
                To specify a unit provide as string,
                e.g. ``'2ms'``.
                To specify in samples provide as string without unit,
                e.g. ``'2000'``
            end: end processing at this position.
                If value is as a float or integer it is treated as seconds.
                To specify a unit provide as string,
                e.g. ``'2ms'``.
                To specify in samples provide as string without unit,
                e.g. ``'2000'``

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

    def _reshape_3d(
            self,
            features: typing.Union[np.ndarray, pd.Series]
    ):
        r"""Reshape to [n_channels, n_features, n_frames]."""

        features = np.array(features)
        features = np.atleast_1d(features)

        if self.process.process_func_is_mono:
            # when mono processing is turned on
            # the channel dimension has to be 1
            # so we would usually omit it,
            # but since older versions required
            # a channel dimension we have to
            # consider two special cases
            if (features.ndim == 4) and \
                    (features.shape[1] == 1):
                # (channels, 1, features, frames)
                # -> (channels, features, frames)
                features = features.squeeze(axis=1)
            elif (features.ndim == 3) and \
                    (self.win_dur is None) and \
                    (features.shape[1] == 1):
                # (channels, 1, features)
                # -> (channels, features)
                features = features.squeeze(axis=1)

        if features.ndim > 3:
            raise RuntimeError(
                f'Dimension of extracted features must be 1, 2 or 3, '
                f'not {features.ndim}.'
            )

        # figure out channels, feature, frames
        if features.ndim == 1:
            n_channels = 1
            n_features = features.size
            n_frames = 1
        elif features.ndim == 2:
            if (features.shape[0] == self.num_channels) and \
                    (features.shape[1] == self.num_features):
                n_channels = features.shape[0]
                n_features = features.shape[1]
                n_frames = 1
            elif features.shape[0] == self.num_features:
                n_channels = 1
                n_features = features.shape[0]
                n_frames = features.shape[1]
            else:
                raise RuntimeError(
                    f'Cannot determine feature shape from '
                    f'{features.shape}, ',
                    f'when expected shape is '
                    f'({self.num_channels, self.num_features, -1}).'
                )
        else:
            n_channels = features.shape[0]
            n_features = features.shape[1]
            n_frames = features.shape[2]

        # assert channels and features have expected length
        if n_channels != self.num_channels:
            raise RuntimeError(
                f'Number of channels must be'
                f' {self.num_channels}, '
                f'not '
                f'{n_channels}.'
            )
        if n_features != self.num_features:
            raise RuntimeError(
                f'Number of features must be '
                f'{self.num_features}, '
                f'not '
                f'{n_features}.'
            )

        # reshape features to (channels,  features, frames)
        return features.reshape([n_channels, n_features, n_frames])

    def _series_to_frame(
            self,
            series: pd.Series,
    ) -> pd.DataFrame:

        if series.empty:
            return pd.DataFrame(
                columns=self.column_names,
                dtype=object,
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
        # Assumed formats are:
        # [n_channels, n_features, n_frames]
        # [n_channels, n_features]
        # [n_features, n_frames]
        # [n_features]

        win_dur = self.win_dur
        hop_dur = self.hop_dur
        if win_dur is not None:
            win_dur = utils.to_timedelta(win_dur, self.process.sampling_rate)
            hop_dur = utils.to_timedelta(hop_dur, self.process.sampling_rate)

        features = self._reshape_3d(features)
        n_channels, n_features, n_frames = features.shape

        # Combine features and channels into one dimension
        # f1-c1, f2-c1, ..., fN-c1, ..., f1-cM, f2-cM, ..., fN-cM
        new_shape = (n_channels * n_features, n_frames)
        features = features.reshape(new_shape).T

        if n_frames > 1 and win_dur is None:
            raise RuntimeError(
                f"Got "
                f"{n_frames} "
                f"frames, but 'win_dur' is not set."
            )

        if win_dur is not None:
            starts = pd.timedelta_range(
                start,
                freq=hop_dur,
                periods=n_frames,
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
            feature array with shape
            ``(num_channels, num_features, num_frames)``

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
        return self._reshape_3d(y)
