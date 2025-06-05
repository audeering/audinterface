from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import errno
import inspect
import os

import numpy as np
import pandas as pd

import audeer
import audformat

from audinterface.core.process import Process
from audinterface.core.segment import Segment
from audinterface.core.typing import Timestamp
from audinterface.core.typing import Timestamps
import audinterface.core.utils as utils


def zeros(signal, sampling_rate, num_channels, num_features) -> np.ndarray:
    r"""Default feature process function.

    This function is used,
    when ``Feature`` is instantiated
    with ``process_func=None``.
    It returns zeros for all channels and features.

    Args:
        signal: signal
        sampling_rate: sampling rate in Hz
        num_channels: number of audio channels
        num_features: number of features

    Returns:
        zeros with size ``(num_channels, num_features)``

    """
    return np.zeros(
        (num_channels, num_features),
        dtype=object,
    )


class Feature:
    r"""Feature extraction interface.

    The features are returned as a :class:`pandas.DataFrame`.
    If your input signal is of size ``(num_channels, num_time_steps)``,
    the returned object has ``num_channels * num_features`` columns.
    It will have one row per file or signal.

    If features are extracted using a sliding window,
    each window will be stored as one row.
    If ``win_dur`` is specified ``start`` and ``end`` indices
    are referred from the original ``start`` and ``end`` arguments
    and the window positions.
    If ``win_dur`` is ``None``,
    the original ``start`` and ``end`` indices are kept.
    If
    ``process_func_applies_sliding_window``
    is set to ``True``
    the processing function
    is responsible to apply the sliding window.
    Otherwise,
    the sliding window is applied before
    the processing function is called.

    If the arguments
    ``win_dur`` and ``hop_dur``
    are not specified in
    ``process_func_args``,
    but
    ``process_func``
    expects them,
    they are passed on automatically.

    Args:
        feature_names: features are stored as columns in a data frame,
            where ``feature_names`` defines the names of the columns.
            If ``len(channels)`` > 1,
            the data frame has a multi-column index with
            with channel ID as first level
            and ``feature_names`` as second level
        name: name of the feature set, e.g. ``'stft'``
        params: parameters that describe the feature set,
            e.g. ``{'win_size': 512, 'hop_size': 256, 'num_fft': 512}``.
            With the parameters you can differentiate different flavors of
            the same feature set
        process_func: feature extraction function,
            which expects the two positional arguments ``signal``
            and ``sampling_rate``
            and any number of additional keyword arguments
            (see ``process_func_args``).
            There are the following special arguments:
            ``'idx'``, ``'file'``, ``'root'``.
            If expected by the function,
            but not specified in
            ``process_func_args``,
            they will be replaced with:
            a running index,
            the currently processed file,
            the root folder.
            The function must return features in the shape of
            ``(num_features)``,
            ``(num_channels, num_features)``,
            ``(num_features, num_frames)``,
            or ``(num_channels, num_features, num_frames)``
        process_func_args: (keyword) arguments passed on to the processing
            function
        process_func_is_mono: apply ``process_func`` to every channel
            individually
        process_func_applies_sliding_window:
            if ``True``
            the processing function receives
            whole files or segments and is responsible
            for applying a sliding window itself.
            If ``False``,
            the sliding window is applied internally
            and the processing function
            receives individual frames instead.
            Applies only if
            features are extracted in a framewise manner
            (see ``win_dur`` and ``hop_dur``)
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal
        resample: if ``True`` enforces given sampling rate by resampling
        channels: channel selection, see :func:`audresample.remix`
        win_dur: window duration,
            if features are extracted with a sliding window.
            If value is a float or integer
            it is treated as seconds.
            See :func:`audinterface.utils.to_timedelta` for further options
        hop_dur: hop duration,
            if features are extracted with a sliding window.
            This defines the shift between two windows.
            If value is a float or integer
            it is treated as seconds.
            See :func:`audinterface.utils.to_timedelta` for further options.
            Defaults to ``win_dur / 2``
        min_signal_dur: minimum signal duration
            required by ``process_func``.
            If value is a float or integer
            it is treated as seconds.
            See :func:`audinterface.utils.to_timedelta` for further options.
            If provided signal is shorter,
            it will be zero padded at the end
        max_signal_dur: maximum signal duraton
            required by ``process_func``.
            If value is a float or integer
            it is treated as seconds.
            See :func:`audinterface.utils.to_timedelta` for further options.
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

    Examples:
        >>> def mean_std(signal, sampling_rate):
        ...     return [signal.mean(), signal.std()]
        >>> interface = Feature(["mean", "std"], process_func=mean_std)
        >>> signal = np.array([1.0, 2.0, 3.0])
        >>> interface(signal, sampling_rate=3)
        array([[[2.        ],
                [0.81649658]]])
        >>> interface.process_signal(signal, sampling_rate=3)
                                mean       std
        start  end
        0 days 0 days 00:00:01   2.0  0.816497
        >>> # Apply interface on an audformat conform index of a dataframe
        >>> import audb
        >>> db = audb.load(
        ...     "emodb",
        ...     version="1.3.0",
        ...     media="wav/03a01Fa.wav",
        ...     full_path=False,
        ...     verbose=False,
        ... )
        >>> index = db["emotion"].index
        >>> interface.process_index(index, root=db.root)
                                                           mean       std
        file            start  end
        wav/03a01Fa.wav 0 days 0 days 00:00:01.898250 -0.000311  0.082317
        >>> interface.process_index(index, root=db.root, preserve_index=True)
                        mean       std
        file
        wav/03a01Fa.wav -0.000311  0.082317
        >>> # Apply interface with a sliding window
        >>> interface = Feature(
        ...     ["mean", "std"],
        ...     process_func=mean_std,
        ...     win_dur=1.0,
        ...     hop_dur=0.25,
        ... )
        >>> interface.process_index(index, root=db.root)
                                                                           mean       std
        file            start                  end
        wav/03a01Fa.wav 0 days 00:00:00        0 days 00:00:01        -0.000329  0.098115
                        0 days 00:00:00.250000 0 days 00:00:01.250000 -0.000405  0.087917
                        0 days 00:00:00.500000 0 days 00:00:01.500000 -0.000285  0.067042
                        0 days 00:00:00.750000 0 days 00:00:01.750000 -0.000187  0.063677
        >>> # Apply the same process function on all channels
        >>> # of a multi-channel signal
        >>> import audiofile
        >>> signal, sampling_rate = audiofile.read(
        ...     audeer.path(db.root, db.files[0]),
        ...     always_2d=True,
        ... )
        >>> signal_multi_channel = np.concatenate(
        ...     [
        ...         signal - 0.5,
        ...         signal + 0.5,
        ...     ],
        ... )
        >>> interface = Feature(
        ...     ["mean", "std"],
        ...     process_func=mean_std,
        ...     process_func_is_mono=True,
        ...     channels=[0, 1],
        ... )
        >>> interface.process_signal(
        ...     signal_multi_channel,
        ...     sampling_rate,
        ... )
                                              0                   1
                                           mean       std      mean       std
        start  end
        0 days 0 days 00:00:01.898250 -0.500311  0.082317  0.499689  0.082317

    """  # noqa: E501

    def __init__(
        self,
        feature_names: str | Sequence[str],
        *,
        name: str | None = None,
        params: dict | None = None,
        process_func: Callable[..., object] | None = None,
        process_func_args: dict[str, object] | None = None,
        process_func_is_mono: bool = False,
        process_func_applies_sliding_window: bool = False,
        sampling_rate: int | None = None,
        resample: bool = False,
        channels: int | Sequence[int] = 0,
        mixdown: bool = False,
        win_dur: Timestamp | None = None,
        hop_dur: Timestamp | None = None,
        min_signal_dur: Timestamp | None = None,
        max_signal_dur: Timestamp | None = None,
        segment: Segment | None = None,
        keep_nat: bool = False,
        num_workers: int | None = 1,
        multiprocessing: bool = False,
        verbose: bool = False,
    ):
        if mixdown or isinstance(channels, int):
            num_channels = 1
        else:
            num_channels = len(channels)

        feature_names = audeer.to_list(feature_names)
        if num_channels > 1:
            column_names = []
            for channel in channels:
                column_names.extend(
                    [(channel, feature_name) for feature_name in feature_names]
                )
            column_names = pd.MultiIndex.from_tuples(column_names)
        else:
            column_names = pd.Index(feature_names)

        process_func_args = process_func_args or {}
        if process_func is None:
            process_func_args["num_channels"] = num_channels
            process_func_args["num_features"] = len(feature_names)
            process_func = zeros

        if win_dur is None and hop_dur is not None:
            raise ValueError("You have to specify 'win_dur' if 'hop_dur' is given.")
        if win_dur is not None and hop_dur is None:
            hop_dur = utils.to_timedelta(win_dur, sampling_rate) / 2

        # add 'win_dur' and 'hop_dur' to process_func_args
        # if expected by function but not yet set
        signature = inspect.signature(process_func)
        if "win_dur" in signature.parameters and "win_dur" not in process_func_args:
            process_func_args["win_dur"] = win_dur
        if "hop_dur" in signature.parameters and "hop_dur" not in process_func_args:
            process_func_args["hop_dur"] = hop_dur

        process = Process(
            process_func=process_func,
            process_func_args=process_func_args,
            process_func_is_mono=process_func_is_mono,
            sampling_rate=sampling_rate,
            resample=resample,
            channels=channels,
            mixdown=mixdown,
            win_dur=None if process_func_applies_sliding_window else win_dur,
            hop_dur=None if process_func_applies_sliding_window else hop_dur,
            min_signal_dur=min_signal_dur,
            max_signal_dur=max_signal_dur,
            segment=segment,
            keep_nat=keep_nat,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
            verbose=verbose,
        )

        self.column_names = column_names
        r"""Feature column names."""

        self.feature_names = feature_names
        r"""Feature names."""

        self.hop_dur = hop_dur
        r"""Hop duration."""

        self.name = name
        r"""Name of the feature set."""

        self.num_channels = num_channels
        r"""Expected number of channels"""

        self.num_features = len(feature_names)
        r"""Number of features."""

        self.params = params
        r"""Dictionary of parameters describing the feature set."""

        self.process = process
        r"""Processing object."""

        self.process_func_applies_sliding_window = process_func_applies_sliding_window
        r"""Controls if processing function applies sliding window."""

        self.verbose = verbose
        r"""Show debug messages."""

        self.win_dur = win_dur
        r"""Window duration."""

    def process_file(
        self,
        file: str,
        *,
        start: Timestamp | None = None,
        end: Timestamp | None = None,
        root: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        r"""Extract features from an audio file.

        Args:
            file: file path
            start: start processing at this position.
                If value is a float or integer it is treated as seconds.
                See :func:`audinterface.utils.to_timedelta` for further options
            end: end processing at this position.
                If value is a float or integer it is treated as seconds.
                See :func:`audinterface.utils.to_timedelta` for further options
            root: root folder to expand relative file path
            process_func_args: (keyword) arguments passed on
                to the processing function.
                They will temporarily overwrite
                the ones stored in
                :attr:`audinterface.Feature.process.process_func_args`

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
            process_func_args=process_func_args,
        )
        return self._series_to_frame(series)

    def process_files(
        self,
        files: Sequence[str],
        *,
        starts: Timestamps | None = None,
        ends: Timestamps | None = None,
        root: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        r"""Extract features for a list of files.

        Args:
            files: list of file paths
            starts: segment start positions.
                Time values given as float or integers are treated as seconds.
                See :func:`audinterface.utils.to_timedelta`
                for further options.
                If a scalar is given, it is applied to all files
            ends: segment end positions.
                Time values given as float or integers are treated as seconds.
                See :func:`audinterface.utils.to_timedelta`
                for further options.
                If a scalar is given, it is applied to all files
            root: root folder to expand relative file paths
            process_func_args: (keyword) arguments passed on
                to the processing function.
                They will temporarily overwrite
                the ones stored in
                :attr:`audinterface.Feature.process.process_func_args`

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
            process_func_args=process_func_args,
        )
        return self._series_to_frame(series)

    def process_folder(
        self,
        root: str,
        *,
        filetype: str = "wav",
        include_root: bool = True,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        r"""Extract features from files in a folder.

        .. note:: At the moment does not scan in sub-folders!

        Args:
            root: root folder
            filetype: file extension
            include_root: if ``True``
                the file paths are absolute
                in the index
                of the returned result
            process_func_args: (keyword) arguments passed on
                to the processing function.
                They will temporarily overwrite
                the ones stored in
                :attr:`audinterface.Feature.process.process_func_args`

        Raises:
            FileNotFoundError: if folder does not exist
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if multiple frames are returned,
                but ``win_dur`` is not set

        """
        root = audeer.path(root)
        if not os.path.exists(root):
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                root,
            )

        files = audeer.list_file_names(
            root,
            filetype=filetype,
            basenames=not include_root,
        )
        return self.process_files(
            files,
            root=root,
            process_func_args=process_func_args,
        )

    def process_index(
        self,
        index: pd.Index,
        *,
        preserve_index: bool = False,
        root: str | None = None,
        cache_root: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        r"""Extract features from an index conform to audformat_.

        If ``cache_root`` is not ``None``,
        a hash value is created from the index
        using :func:`audformat.utils.hash` and
        the result is stored as
        ``<cache_root>/<hash>.pkl``.
        When called again with the same index,
        features will be read from the cached file.

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        Args:
            index: index with segment information
            preserve_index: if ``True``
                and :attr:`audinterface.Feature.process.segment` is ``None``
                the returned index
                will be of same type
                as the original one,
                otherwise always a segmented index is returned
            root: root folder to expand relative file paths
            cache_root: cache folder (see description)
            process_func_args: (keyword) arguments passed on
                to the processing function.
                They will temporarily overwrite
                the ones stored in
                :attr:`audinterface.Feature.process.process_func_args`

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            RuntimeError: if multiple frames are returned,
                but ``win_dur`` is not set
            ValueError: if index is not conform to audformat_

        """
        cache_path = None

        if cache_root is not None:
            cache_root = audeer.mkdir(cache_root)
            hash = audformat.utils.hash(index)
            cache_path = os.path.join(cache_root, f"{hash}.pkl")

        if cache_path and os.path.exists(cache_path):
            df = pd.read_pickle(cache_path)
        else:
            y = self.process.process_index(
                index,
                root=root,
                process_func_args=process_func_args,
            )
            df = self._series_to_frame(y)

            if cache_path is not None:
                df.to_pickle(cache_path, protocol=4)

        if self.process.segment is None and preserve_index:
            # Convert segmented index to filewise index
            # if original index was filewise
            df.index = index

        return df

    def process_signal(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        *,
        file: str | None = None,
        start: Timestamp | None = None,
        end: Timestamp | None = None,
        process_func_args: dict[str, object] | None = None,
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
                If value is a float or integer it is treated as seconds.
                See :func:`audinterface.utils.to_timedelta` for further options
            end: end processing at this position.
                If value is a float or integer it is treated as seconds.
                See :func:`audinterface.utils.to_timedelta` for further options
            process_func_args: (keyword) arguments passed on
                to the processing function.
                They will temporarily overwrite
                the ones stored in
                :attr:`audinterface.Feature.process.process_func_args`

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
            process_func_args=process_func_args,
        )
        return self._series_to_frame(series)

    def process_signal_from_index(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        index: pd.MultiIndex,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        r"""Split a signal into segments and extract features for each segment.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            index: a :class:`pandas.MultiIndex` with two levels
                named `start` and `end` that hold start and end
                positions as :class:`pandas.Timedelta` objects.
                See also :func:`audinterface.utils.signal_index`
            process_func_args: (keyword) arguments passed on
                to the processing function.
                They will temporarily overwrite
                the ones stored in
                :attr:`audinterface.Feature.process.process_func_args`

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
            process_func_args=process_func_args,
        )
        return self._series_to_frame(series)

    def to_numpy(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        r"""Return feature values as a numpy array.

        The returned :class:`numpy.ndarray`
        has the original shape,
        i.e. ``(channels, features, time)``.

        Args:
            frame: feature frame

        """
        return frame.values.T.reshape(self.num_channels, self.num_features, -1)

    def _reshape_3d(self, features: np.ndarray | pd.Series):
        r"""Reshape to [n_channels, n_features, n_frames]."""
        features = np.array(features)
        features = np.atleast_1d(features)

        if self.process.process_func_is_mono:
            # when mono processing is turned on
            # the channel dimension has to be 1,
            # so we would usually omit it,
            # but since older versions required
            # a channel dimension we have to
            # consider two special cases
            if (features.ndim == 4) and (features.shape[1] == 1):
                # (channels, 1, features, frames)
                # -> (channels, features, frames)
                features = features.squeeze(axis=1)
            elif (
                (features.ndim == 3)
                and (self.win_dur is None)
                and (features.shape[1] == 1)
            ):
                # (channels, 1, features)
                # -> (channels, features)
                features = features.squeeze(axis=1)

            if self.win_dur and not self.process_func_applies_sliding_window:
                # (channels, features)
                # -> (channels, features, 1)
                features = features.reshape(self.num_channels, -1, 1)

        if features.ndim > 3:
            raise RuntimeError(
                f"Dimension of extracted features must be 1, 2 or 3, "
                f"not {features.ndim}."
            )

        # figure out channels, feature, frames
        if features.ndim == 1:
            n_channels = 1
            n_features = features.size
            n_frames = 1
        elif features.ndim == 2:
            if (features.shape[0] == self.num_channels) and (
                features.shape[1] == self.num_features
            ):
                n_channels = features.shape[0]
                n_features = features.shape[1]
                n_frames = 1
            elif features.shape[0] == self.num_features:
                n_channels = 1
                n_features = features.shape[0]
                n_frames = features.shape[1]
            else:
                raise RuntimeError(
                    f"Cannot determine feature shape from " f"{features.shape}, ",
                    f"when expected shape is "
                    f"({self.num_channels, self.num_features, -1}).",
                )
        else:
            n_channels = features.shape[0]
            n_features = features.shape[1]
            n_frames = features.shape[2]

        # assert channels and features have expected length
        if n_channels != self.num_channels:
            raise RuntimeError(
                f"Number of channels must be"
                f" {self.num_channels}, "
                f"not "
                f"{n_channels}."
            )
        if n_features != self.num_features:
            raise RuntimeError(
                f"Number of features must be "
                f"{self.num_features}, "
                f"not "
                f"{n_features}."
            )

        # reshape features to (channels,  features, frames)
        return features.reshape([n_channels, n_features, n_frames])

    def _series_to_frame(
        self,
        y: pd.Series,
    ) -> pd.DataFrame:
        if y.empty:
            if self.process.segment is None:
                index = []
            else:
                index = audformat.segmented_index()
            return pd.DataFrame(
                index=index,
                columns=self.column_names,
                dtype=object,
            )

        if self.win_dur is not None and self.process_func_applies_sliding_window:
            win_dur = utils.to_timedelta(
                self.win_dur,
                self.process.sampling_rate,
            )
            hop_dur = utils.to_timedelta(
                self.hop_dur,
                self.process.sampling_rate,
            )

            starts = []
            ends = []
            data = []

            if len(y.index.levels) == 3:
                files = []

                for idx, ((file, start, end), values) in enumerate(y.items()):
                    frames = self._values_to_frame(values)
                    data.append(frames)

                    times = pd.timedelta_range(
                        start,
                        freq=hop_dur,
                        periods=frames.shape[0],
                    )

                    starts.extend(times.to_list())
                    ends.extend((times + win_dur).to_list())
                    files.extend([file] * len(times))

                index = audformat.segmented_index(files, starts, ends)

            else:
                for idx, ((start, end), values) in enumerate(y.items()):
                    frames = self._values_to_frame(values)
                    data.append(frames)

                    times = pd.timedelta_range(
                        start,
                        freq=hop_dur,
                        periods=frames.shape[0],
                    )

                    starts.extend(times.to_list())
                    ends.extend((times + win_dur).to_list())

                index = utils.signal_index(starts, ends)

        else:
            index = y.index
            data = [self._values_to_frame(values) for values in y]

        data = np.concatenate(data)

        df = pd.DataFrame(
            data,
            index=index,
            columns=self.column_names,
        )

        return df

    def _values_to_frame(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        # Convert features to a pd.DataFrame
        # Assumed formats are:
        # [n_channels, n_features, n_frames]
        # [n_channels, n_features]
        # [n_features, n_frames]
        # [n_features]

        features = self._reshape_3d(features)
        n_channels, n_features, n_frames = features.shape

        # Combine features and channels into one dimension
        # f1-c1, f2-c1, ..., fN-c1, ..., f1-cM, f2-cM, ..., fN-cM
        new_shape = (n_channels * n_features, n_frames)
        features = features.reshape(new_shape).T

        if n_frames > 1 and self.win_dur is None:
            raise RuntimeError(
                f"Got " f"{n_frames} " f"frames, but 'win_dur' is not set."
            )

        return features

    def __call__(
        self,
        signal: np.ndarray,
        sampling_rate: int,
    ) -> np.ndarray:
        r"""Apply processing to signal.

        This function processes the signal
        **without** transforming the output
        into a :class:`pandas.DataFrame`.
        Instead, it will return the raw processed signal.
        However,
        if channel selection,
        mixdown and/or resampling is enabled,
        the signal will be first remixed
        and resampled if the input sampling rate
        does not fit the expected sampling rate.

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
