from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import errno
import os

import numpy as np
import pandas as pd

import audeer
import audformat

from audinterface.core.typing import Timestamp
from audinterface.core.typing import Timestamps
import audinterface.core.utils as utils


def empty_series(signal, sampling_rate, **kwargs) -> pd.Series:
    r"""Default segment with feature process function.

    This function is used,
    when ``SegmentWithFeature`` is instantiated
    with ``process_func=None``.
    It returns a series with an empty multi-index,
    with levels ``start`` and ``end``.

    Args:
        signal: signal
        sampling_rate: sampling rate in Hz
        **kwargs: additional keyword arguments of the processing function

    Returns:
        empty series

    """
    index = utils.signal_index()
    return pd.Series(index=index)


class SegmentWithFeature:
    r"""Segmentation with feature interface.

    Interface for functions that apply a segmentation to the input signal,
    and also compute features for those segments at the same time,
    e.g. a speech recognition model that recognizes speech
    and also provides the time stamps of that speech.

    The features are returned as a :class:`pandas.DataFrame`
    with ``num_features`` columns
    and one row per detected segment.

    Args:
        feature_names: features are stored as columns in a data frame,
            where ``feature_names`` defines the names of the columns.
        name: name of the feature set, e.g. ``'stft'``
        params: parameters that describe the feature set,
            e.g. ``{'win_size': 512, 'hop_size': 256, 'num_fft': 512}``.
            With the parameters you can differentiate different flavors of
            the same feature set
        process_func: segmentation with feature function,
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
            Must return a :class:`pandas.Series`
            with a :class:`pandas.MultiIndex` with two levels
            named `start` and `end` that hold start and end
            positions as :class:`pandas.Timedelta` objects,
            and with elements in the shape of
            ``(num_features)``.
        process_func_args: (keyword) arguments passed on to the processing
            function
        sampling_rate: sampling rate in Hz.
            If ``None`` it will call ``process_func`` with the actual
            sampling rate of the signal
        resample: if ``True`` enforces given sampling rate by resampling
        channels: channel selection, see :func:`audresample.remix`
        mixdown: apply mono mix-down on selection
        min_signal_dur: minimum signal length
            required by ``process_func``.
            If value is a float or integer
            it is treated as seconds.
            See :func:`audinterface.utils.to_timedelta` for further options.
            If provided signal is shorter,
            it will be zero padded at the end
        max_signal_dur: maximum signal length
            required by ``process_func``.
            If value is a float or integer
            it is treated as seconds.
            See :func:`audinterface.utils.to_timedelta` for further options.
            If provided signal is longer,
            it will be cut at the end
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
        ValueError: if ``resample = True``, but ``sampling_rate = None``

    Examples:
        >>> def segment_with_mean_std(signal, sampling_rate, *, win_size=1.0, hop_size=1.0):
        ...     size = signal.shape[1] / sampling_rate
        ...     starts = pd.to_timedelta(
        ...         np.arange(0, size - win_size + (1 / sampling_rate), hop_size),
        ...         unit="s",
        ...     )
        ...     ends = pd.to_timedelta(
        ...         np.arange(win_size, size + (1 / sampling_rate), hop_size), unit="s"
        ...     )
        ...     # Get windows of shape (channels, samples, frames)
        ...     frames = utils.sliding_window(signal, sampling_rate, win_size, hop_size)
        ...     means = frames.mean(axis=(0, 1))
        ...     stds = frames.std(axis=(0, 1))
        ...     index = pd.MultiIndex.from_tuples(zip(starts, ends), names=["start", "end"])
        ...     features = list(np.stack((means, stds), axis=-1))
        ...     return pd.Series(data=features, index=index)
        >>> interface = SegmentWithFeature(
        ...     feature_names=["mean", "std"], process_func=segment_with_mean_std
        ... )
        >>> signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> interface(signal, sampling_rate=2)
        start            end
        0 days 00:00:00  0 days 00:00:01    [1.5, 0.5]
        0 days 00:00:01  0 days 00:00:02    [3.5, 0.5]
        0 days 00:00:02  0 days 00:00:03    [5.5, 0.5]
        dtype: object
        >>> interface.process_signal(signal, sampling_rate=2)
                                        mean  std
        start           end
        0 days 00:00:00 0 days 00:00:01   1.5  0.5
        0 days 00:00:01 0 days 00:00:02   3.5  0.5
        0 days 00:00:02 0 days 00:00:03   5.5  0.5
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
        wav/03a01Fa.wav 0 days 0 days 00:00:01 -0.000329  0.098115

    """  # noqa: E501

    def __init__(
        self,
        feature_names: str | Sequence[str],
        *,
        name: str | None = None,
        params: dict[str, object] | None = None,
        process_func: Callable[..., pd.Series] | None = None,
        process_func_args: dict[str, object] | None = None,
        sampling_rate: int | None = None,
        resample: bool = False,
        channels: int | Sequence[int] = 0,
        mixdown: bool = False,
        min_signal_dur: Timestamp | None = None,
        max_signal_dur: Timestamp | None = None,
        keep_nat: bool = False,
        num_workers: int | None = 1,
        multiprocessing: bool = False,
        verbose: bool = False,
    ):
        # avoid cycling imports
        from audinterface.core.process import Process

        feature_names = audeer.to_list(feature_names)
        process_func_args = process_func_args or {}
        if process_func is None:
            process_func_args["feature_names"] = feature_names
            process_func = empty_series

        process = Process(
            process_func=process_func,
            process_func_args=process_func_args,
            sampling_rate=sampling_rate,
            resample=resample,
            channels=channels,
            mixdown=mixdown,
            min_signal_dur=min_signal_dur,
            max_signal_dur=max_signal_dur,
            keep_nat=keep_nat,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
            verbose=verbose,
        )

        self.feature_names = feature_names
        r"""Feature names."""

        self.name = name
        r"""Name of the feature set."""

        self.num_features = len(feature_names)
        r"""Number of features."""

        self.params = params
        r"""Dictionary of parameters describing the feature set."""

        self.process = process
        r"""Processing object."""

    def process_file(
        self,
        file: str,
        *,
        start: Timestamp | None = None,
        end: Timestamp | None = None,
        root: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        r"""Segment the content of an audio file and extract features.

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
                :attr:`audinterface.SegmentWithFeature.process.process_func_args`

        Returns:
            :class:`pandas.DataFrame` with segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            ValueError: if the process function doesn't return a :class:`pd.Series`
                with index conform to audformat_
                and elements of shape ``(num_features)``

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        if start is None or pd.isna(start):
            start = pd.to_timedelta(0)
        y = self.process.process_file(
            file,
            start=start,
            end=end,
            root=root,
            process_func_args=process_func_args,
        )
        return self._construct_frame_segmented_index(y)

    def process_files(
        self,
        files: Sequence[str],
        *,
        starts: Timestamps | None = None,
        ends: Timestamps | None = None,
        root: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        r"""Segment and extract features for a list of files.

        Args:
            files: list of file paths
            starts: segment start positions.
                Time values given as float or integers are treated as seconds.
                See :func:`audinterface.utils.to_timedelta`
                for further options.
                If a scalar is given, it is applied to all files
            ends: segment end positions.
                Time values given as float or integers are treated as seconds
                See :func:`audinterface.utils.to_timedelta`
                for further options.
                If a scalar is given, it is applied to all files
            root: root folder to expand relative file paths
            process_func_args: (keyword) arguments passed on
                to the processing function.
                They will temporarily overwrite
                the ones stored in
                :attr:`audinterface.SegmentWithFeature.process.process_func_args`

        Returns:
            :class:`pandas.DataFrame` with segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            ValueError: if the process function doesn't return a :class:`pd.Series`
                with index conform to audformat_
                and elements of shape ``(num_features)``

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        y = self.process.process_files(
            files,
            starts=starts,
            ends=ends,
            root=root,
            process_func_args=process_func_args,
        )
        return self._construct_frame_segmented_index(y)

    def process_folder(
        self,
        root: str,
        *,
        filetype: str = "wav",
        include_root: bool = True,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        r"""Segment and extract features for files in a folder.

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
                :attr:`audinterface.SegmentWithFeature.process.process_func_args`

        Returns:
            :class:`pandas.DataFrame` with segmented index conform to audformat_

        Raises:
            FileNotFoundError: if folder does not exist
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            ValueError: if the process function doesn't return a :class:`pd.Series`
                with index conform to audformat_
                and elements of shape ``(num_features)``

        .. _audformat: https://audeering.github.io/audformat/data-format.html

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
        root: str | None = None,
        cache_root: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        r"""Segment and extract features for files or segments from an index.

        If ``cache_root`` is not ``None``,
        a hash value is created from the index
        using :func:`audformat.utils.hash` and
        the result is stored as
        ``<cache_root>/<hash>.pkl``.
        When called again with the same index,
        results will be read from the cached file.

        Args:
            index: index conform to audformat_
            root: root folder to expand relative file paths
            cache_root: cache folder (see description)
            process_func_args: (keyword) arguments passed on
                to the processing function.
                They will temporarily overwrite
                the ones stored in
                :attr:`audinterface.SegmentWithFeature.process.process_func_args`

        Returns:
            :class:`pandas.DataFrame` with segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            ValueError: if the process function doesn't return a :class:`pd.Series`
                with index conform to audformat_
                and elements of shape ``(num_features)``

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        index = audformat.utils.to_segmented_index(index)
        utils.assert_index(index)

        y = self.process.process_index(
            index,
            preserve_index=False,
            root=root,
            cache_root=cache_root,
            process_func_args=process_func_args,
        )
        return self._construct_frame_segmented_index(y)

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
        r"""Segment and extract features for audio signal.

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
                :attr:`audinterface.SegmentWithFeature.process.process_func_args`

        Returns:
            :class:`pandas.DataFrame` with segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            ValueError: if the process function doesn't return a :class:`pd.Series`
                with index conform to audformat_
                and elements of shape ``(num_features)``

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        y = self.process.process_signal(
            signal,
            sampling_rate,
            file=file,
            start=start,
            end=end,
            process_func_args=process_func_args,
        )
        if file is None:
            return self._construct_frame_signal_index(y)
        else:
            return self._construct_frame_segmented_index(y)

    def process_signal_from_index(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        index: pd.Index,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        r"""Segment and extract features for parts of a signal.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz
            index: a segmented index conform to audformat_
                or a :class:`pandas.MultiIndex` with two levels
                named `start` and `end` that hold start and end
                positions as :class:`pandas.Timedelta` objects.
                See also :func:`audinterface.utils.signal_index`
            process_func_args: (keyword) arguments passed on
                to the processing function.
                They will temporarily overwrite
                the ones stored in
                :attr:`audinterface.SegmentWithFeature.process.process_func_args`

        Returns:
            :class:`pandas.DataFrame` with segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            ValueError: if index contains duplicates
            ValueError: if the process function doesn't return a :class:`pd.Series`
                with index conform to audformat_
                and elements of shape ``(num_features)``

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        utils.assert_index(index)

        if isinstance(index, pd.MultiIndex) and len(index.levels) == 2:
            has_file_level = False
            params = [
                (
                    (signal, sampling_rate),
                    {"start": start, "end": end},
                )
                for start, end in index
            ]
        else:
            has_file_level = True
            index = audformat.utils.to_segmented_index(index)
            params = [
                (
                    (signal, sampling_rate),
                    {
                        "file": file,
                        "start": start,
                        "end": end,
                        "process_func_args": process_func_args,
                    },
                )
                for file, start, end in index
            ]

        y = audeer.run_tasks(
            self.process_signal,
            params,
            num_workers=self.process.num_workers,
            multiprocessing=self.process.multiprocessing,
            progress_bar=self.process.verbose,
            task_description=f"Process {len(index)} segments",
            maximum_refresh_time=1,
        )
        files = []
        starts = []
        ends = []
        features = {col: [] for col in self.feature_names}
        for df in y:
            if df is None:
                continue
            if has_file_level:
                files.extend(df.index.get_level_values("file"))
            starts.extend(df.index.get_level_values("start"))
            ends.extend(df.index.get_level_values("end"))
            for col in self.feature_names:
                features[col].extend(df[col])
        if has_file_level:
            index = audformat.segmented_index(files, starts, ends)
        else:
            index = utils.signal_index(starts, ends)
        if len(index) == 0:
            # Pass no data to ensure consistent dtype for columns
            return pd.DataFrame(index=index, columns=self.feature_names)
        return pd.DataFrame(index=index, data=features, columns=self.feature_names)

    def process_table(
        self,
        table: pd.Series | pd.DataFrame,
        *,
        root: str | None = None,
        cache_root: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        r"""Segment and extract features for files or segments from a table.

        The labels of the table
        are reassigned to the new segments.
        The columns of the table may not overlap
        with the :attr:`audinterface.SegmentWithFeature.feature_names`.

        If ``cache_root`` is not ``None``,
        a hash value is created from the index
        using :func:`audformat.utils.hash` and
        the result is stored as
        ``<cache_root>/<hash>.pkl``.
        When called again with the same index,
        results will be read from the cached file.

        Args:
            table: :class:`pandas.Series` or :class:`pandas.DataFrame`
                with an index conform to audformat_
            root: root folder to expand relative file paths
            cache_root: cache folder (see description)
            process_func_args: (keyword) arguments passed on
                to the processing function.
                They will temporarily overwrite
                the ones stored in
                :attr:`audinterface.SegmentWithFeature.process.process_func_args`

        Returns:
            :class:`pandas.DataFrame` with segmented index conform to audformat_

        Raises:
            ValueError: if table is not a :class:`pandas.Series`
                or a :class:`pandas.DataFrame`
            ValueError: if the table columns and the extracted feature columns overlap
            ValueError: if the process function doesn't return a :class:`pd.Series`
                with index conform to audformat_
                and elements of shape ``(num_features)``
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        if not isinstance(table, pd.Series) and not isinstance(table, pd.DataFrame):
            raise ValueError("table has to be pd.Series or pd.DataFrame")
        if isinstance(table, pd.Series):
            if table.name in self.feature_names:
                raise ValueError(
                    "Name of table may not overlap with returned feature names."
                )
        else:
            if any([col in self.feature_names for col in table.columns]):
                raise ValueError(
                    "Column names in table may not overlap with returned feature names."
                )
        index = audformat.utils.to_segmented_index(table.index)
        utils.assert_index(index)

        y = self.process.process_index(
            index,
            preserve_index=False,
            root=root,
            cache_root=cache_root,
            process_func_args=process_func_args,
        )

        # Assign labels from the original table
        # to the newly created segments
        files = []
        starts = []
        ends = []
        labels = []
        features = {col: [] for col in self.feature_names}
        if isinstance(table, pd.Series):
            table = table.to_frame()
        for n, ((file, start, _), series) in enumerate(y.items()):
            self._check_return_format(series)
            if series.empty:
                continue
            data = self._reshape_numpy(series)
            data = np.stack(data).T
            index_data = series.index.to_numpy()
            files.extend([file] * len(series))
            starts.extend([t[0] + start for t in index_data])
            ends.extend([t[1] + start for t in index_data])
            for col, val in zip(self.feature_names, data):
                features[col].extend(val)
            labels.extend([[table.iloc[n].values] * len(series)])
        labels = np.vstack(labels) if labels else np.empty((0, table.shape[1]))
        index = audformat.segmented_index(files, starts, ends)
        if len(index) == 0:
            # Pass no data to ensure consistent dtype for columns
            result = pd.DataFrame(
                index=audformat.segmented_index(), columns=self.feature_names
            )
        else:
            result = pd.DataFrame(
                index=index,
                data=features,
                columns=self.feature_names,
            )

        dtypes = [table[col].dtype for col in table.columns]
        labels = {
            col: pd.Series(
                labels[:, ncol], index=index, dtype=dtypes[ncol]
            )  # supports also category
            for ncol, col in enumerate(table.columns)
        }
        table = pd.DataFrame(labels, index, columns=table.columns)

        result = result.join(table)
        return result

    def _check_return_format(self, x):
        if not isinstance(x, pd.Series):
            raise ValueError(f"A series must be returned, but got {type(x)}")
        utils.assert_index(x.index)
        # Disallow filewise index
        if audformat.is_filewise_index(x.index):
            raise ValueError(
                "The returned series must have a signal index or a segmented index"
            )

    def _construct_frame_segmented_index(self, y: pd.Series):
        r"""Construct dataframe with segmented index from process result."""
        files = []
        starts = []
        ends = []
        features = {col: [] for col in self.feature_names}
        for (file, start, _), series in y.items():
            self._check_return_format(series)
            if series.empty:
                continue
            data = self._reshape_numpy(series)
            data = np.stack(data).T
            index_data = series.index.to_numpy()
            files.extend([file] * len(series))
            starts.extend([t[0] + start for t in index_data])
            ends.extend([t[1] + start for t in index_data])
            for col, val in zip(self.feature_names, data):
                features[col].extend(val)
        if not files:
            # Pass no data to ensure consistent dtype for columns
            return pd.DataFrame(
                index=audformat.segmented_index(), columns=self.feature_names
            )
        return pd.DataFrame(
            index=audformat.segmented_index(files, starts, ends),
            data=features,
            columns=self.feature_names,
        )

    def _construct_frame_signal_index(self, y: pd.Series):
        r"""Construct dataframe with signal index from process result."""
        starts = []
        ends = []
        features = {col: [] for col in self.feature_names}
        for (start, _), series in y.items():
            self._check_return_format(series)
            if series.empty:
                continue
            data = self._reshape_numpy(series)
            data = np.stack(data).T
            index_data = series.index.to_numpy()
            starts.extend([t[0] + start for t in index_data])
            ends.extend([t[1] + start for t in index_data])
            for col, val in zip(self.feature_names, data):
                features[col].extend(val)
        if not starts:
            # Pass no data to ensure consistent dtype for columns
            return pd.DataFrame(index=utils.signal_index(), columns=self.feature_names)

        return pd.DataFrame(
            index=utils.signal_index(starts, ends),
            data=features,
            columns=self.feature_names,
        )

    def _reshape_numpy(
        self,
        features: pd.Series,
    ):
        r"""Reshape values in series to numpy array of shape [n_features]."""
        # Cover case that process function returned a series
        # with numpy arrays as elements
        if pd.api.types.is_object_dtype(features):
            data = [self._reshape_numpy_1d(values) for values in features]
        else:
            # Case that the process function returned a series
            # with elements that are scalar values.
            # Use to_numpy() directly to preserve dtype of series
            data = features.to_numpy()
            data = np.reshape(data, (len(features), self.num_features))
        return data

    def _reshape_numpy_1d(
        self,
        features: np.ndarray,
    ):
        r"""Reshape to [n_features]."""
        features = np.asarray(features)
        features = np.atleast_1d(features)
        if not features.shape == (self.num_features,):
            raise ValueError(
                f"The returned features must be reshapable to ({self.num_features})"
            )
        return features

    def __call__(
        self,
        signal: np.ndarray,
        sampling_rate: int,
    ) -> pd.Series:
        r"""Apply processing to signal.

        This function processes the signal **without** transforming the output
        into a :class:`pd.DataFrame`. Instead, it will return the raw
        processed signal. However, if channel selection, mixdown
        and/or resampling is enabled, the signal will be first remixed and
        resampled if the input sampling rate does not fit the expected
        sampling rate.

        Args:
            signal: signal values
            sampling_rate: sampling rate in Hz

        Returns:
            Processed signal

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            ValueError: if the process function doesn't return a :class:`pd.Series`
                with index conform to audformat_
                and elements of shape ``(num_features)``

        .. _audformat: https://audeering.github.io/audformat/data-format.html
        """
        y = self.process(
            signal,
            sampling_rate,
        )
        self._check_return_format(y)
        return y.apply(lambda x: self._reshape_numpy_1d(x))
