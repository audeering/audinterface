import errno
from functools import reduce
import os
import typing

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
    and also compute features for those segments at the same time.
    e.g. an speech recognition model that recognizes speech
    and also provides the time stamps of that speech.

    The features are returned as a :class:`pandas.DataFrame`.
    If the input signal has ``num_channels``,
    the returned object has ``num_channels * num_features`` columns
    and one row per detected segment.

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
            ``(num_features)``
            or ``(num_channels, num_features)``.
        process_func_args: (keyword) arguments passed on to the processing
            function
        process_func_is_mono: apply ``process_func`` to every channel
            individually
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
        ...     # Reshape to (frames, channels, samples)
        ...     frames = frames.transpose(2, 0, 1)
        ...     means = frames.mean(axis=2)
        ...     stds = frames.std(axis=2)
        ...     index = pd.MultiIndex.from_tuples(zip(starts, ends), names=["start", "end"])
        ...     # Pass list of arrays with shape (channels, features) to create series
        ...     features = list(np.stack((means, stds), axis=-1))
        ...     return pd.Series(data=features, index=index)
        >>> interface = SegmentWithFeature(
        ...     feature_names=["mean", "std"], process_func=segment_with_mean_std
        ... )
        >>> signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> interface(signal, sampling_rate=2)
        start            end
        0 days 00:00:00  0 days 00:00:01    [[1.5, 0.5]]
        0 days 00:00:01  0 days 00:00:02    [[3.5, 0.5]]
        0 days 00:00:02  0 days 00:00:03    [[5.5, 0.5]]
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
        >>> # Application on a multi-channel signal
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
        >>> interface = SegmentWithFeature(
        ...     feature_names=["mean", "std"],
        ...     process_func=segment_with_mean_std,
        ...     channels=[0, 1],
        ... )
        >>> interface.process_signal(
        ...     signal_multi_channel,
        ...     sampling_rate,
        ... )
                                    0                   1
                                    mean       std      mean       std
        start  end
        0 days 0 days 00:00:01 -0.500329  0.098115  0.499671  0.098115

    """  # noqa: E501

    def __init__(
        self,
        feature_names: typing.Union[str, typing.Sequence[str]],
        *,
        name: str = None,
        params: typing.Dict = None,
        process_func: typing.Callable[..., pd.Series] = None,
        process_func_args: typing.Dict[str, typing.Any] = None,
        process_func_is_mono: bool = False,
        sampling_rate: int = None,
        resample: bool = False,
        channels: typing.Union[int, typing.Sequence[int]] = 0,
        mixdown: bool = False,
        min_signal_dur: Timestamp = None,
        max_signal_dur: Timestamp = None,
        keep_nat: bool = False,
        num_workers: typing.Optional[int] = 1,
        multiprocessing: bool = False,
        verbose: bool = False,
    ):
        # avoid cycling imports
        from audinterface.core.process import Process

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
            process_func_args["feature_names"] = feature_names
            process_func = empty_series

        process = Process(
            process_func=process_func,
            process_func_args=process_func_args,
            process_func_is_mono=process_func_is_mono,
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
        self.column_names = column_names
        r"""Feature column names."""

        self.feature_names = feature_names
        r"""Feature names."""

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

    def process_file(
        self,
        file: str,
        *,
        start: Timestamp = None,
        end: Timestamp = None,
        root: str = None,
        process_func_args: typing.Dict[str, typing.Any] = None,
    ) -> pd.DataFrame:
        r"""Segment the content of an audio file and extract their features.

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

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        if start is None or pd.isna(start):
            start = pd.to_timedelta(0)
        series = self.process.process_file(
            file,
            start=start,
            end=end,
            root=root,
            process_func_args=process_func_args,
        ).values[0]
        df = self._series_to_frame(series)
        index = audformat.segmented_index(
            files=[file] * len(df),
            starts=df.index.get_level_values("start") + start,
            ends=df.index.get_level_values("end") + start,
        )
        df.index = index
        return df

    def process_files(
        self,
        files: typing.Sequence[str],
        *,
        starts: Timestamps = None,
        ends: Timestamps = None,
        root: str = None,
        process_func_args: typing.Dict[str, typing.Any] = None,
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

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        y = self.process.process_files(
            files,
            starts=starts,
            ends=ends,
            root=root,
            process_func_args=process_func_args,
        )
        files = []
        starts = []
        ends = []
        features = {col: [] for col in self.column_names}
        for (file, start, _), series_or_list in y.items():
            series = self._reshape_series_or_list(series_or_list)
            df = self._series_to_frame(series)
            files.extend([file] * len(df))
            starts.extend(df.index.get_level_values("start") + start)
            ends.extend(df.index.get_level_values("end") + start)
            for col in self.column_names:
                features[col].extend(df[col])
        if len(files) == 0:
            # Pass no data to ensure consistent dtype for columns
            return pd.DataFrame(
                index=audformat.segmented_index(), columns=self.column_names
            )
        return pd.DataFrame(
            index=audformat.segmented_index(files, starts, ends),
            data=features,
            columns=self.column_names,
        )

    def process_folder(
        self,
        root: str,
        *,
        filetype: str = "wav",
        include_root: bool = True,
        process_func_args: typing.Dict[str, typing.Any] = None,
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
        root: str = None,
        cache_root: str = None,
        process_func_args: typing.Dict[str, typing.Any] = None,
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

        files = []
        starts = []
        ends = []
        features = {col: [] for col in self.column_names}
        for (file, start, _), series_or_list in y.items():
            series = self._reshape_series_or_list(series_or_list)
            df = self._series_to_frame(series)
            files.extend([file] * len(df))
            starts.extend(df.index.get_level_values("start") + start)
            ends.extend(df.index.get_level_values("end") + start)
            for col in self.column_names:
                features[col].extend(df[col])
        if len(files) == 0:
            # Pass no data to ensure consistent dtype for columns
            return pd.DataFrame(
                index=audformat.segmented_index(), columns=self.column_names
            )
        return pd.DataFrame(
            index=audformat.segmented_index(files, starts, ends),
            data=features,
            columns=self.column_names,
        )

    def process_table(
        self,
        table: typing.Union[pd.Series, pd.DataFrame],
        *,
        root: str = None,
        cache_root: str = None,
        process_func_args: typing.Dict[str, typing.Any] = None,
        tablesuffix: str = "",
        featuresuffix: str = "",
    ) -> pd.DataFrame:
        r"""Segment and extract features for files or segments from a table.

        The labels of the table
        are reassigned to the new segments.
        If the columns of the table overlap
        with the :attr:`audinterface.SegmentWithFeature.column_names`,
        the ``tablesuffix`` or the ``featuresuffix`` must be specified.
        The provided ``tablesuffix`` is added
        to the table's overlapping column names
        and the provided ``featuresuffix`` is added
        to the extracted features' overlapping column names.
        In case the number of channels is greater than 1,
        the first level of the column names
        (corresponding to the channel ID)
        is renamed.

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
            tablesuffix: suffix to use for the table's overlapping columns
            featuresuffix: suffix to use for the features' overlapping columns

        Returns:
            :class:`pandas.DataFrame` with segmented index conform to audformat_

        Raises:
            ValueError: if table is not a :class:`pandas.Series`
                or a :class:`pandas.DataFrame`
            ValueError: if the table columns have more than 2 levels
            ValueError: if the table columns and the extract feature columns overlap
                and no suffix is specified
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        if not isinstance(table, pd.Series) and not isinstance(table, pd.DataFrame):
            raise ValueError("table has to be pd.Series or pd.DataFrame")
        if isinstance(table, pd.DataFrame) and table.columns.nlevels > 2:
            raise ValueError("Only 1 or 2 column levels are supported")
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
        features = {col: [] for col in self.column_names}
        if isinstance(table, pd.Series):
            for n, ((file, start, _), series_or_list) in enumerate(y.items()):
                series = self._reshape_series_or_list(series_or_list)
                df = self._series_to_frame(series)
                files.extend([file] * len(df))
                starts.extend(df.index.get_level_values("start") + start)
                ends.extend(df.index.get_level_values("end") + start)
                labels.extend([[table.iloc[n]] * len(df.index)])
                for col in self.column_names:
                    features[col].extend(df[col])
            if len(labels) > 0:
                labels = np.hstack(labels)
            else:
                labels = np.empty((0))
        else:
            for n, ((file, start, _), series_or_list) in enumerate(y.items()):
                series = self._reshape_series_or_list(series_or_list)
                df = self._series_to_frame(series)
                files.extend([file] * len(df))
                starts.extend(df.index.get_level_values("start") + start)
                ends.extend(df.index.get_level_values("end") + start)
                if len(df) > 0:  # avoid issues when stacking 0-length dataframes
                    labels.extend([[table.iloc[n].values] * len(df)])
                for col in self.column_names:
                    features[col].extend(df[col])
            if len(labels) > 0:
                labels = np.vstack(labels)
            else:
                labels = np.empty((0, table.shape[1]))  # avoid issue below
        index = audformat.segmented_index(files, starts, ends)
        if len(index) == 0:
            # Pass no data to ensure consistent dtype for columns
            result = pd.DataFrame(
                index=audformat.segmented_index(), columns=self.column_names
            )
        else:
            result = pd.DataFrame(
                index=index,
                data=features,
                columns=self.column_names,
            )

        if isinstance(table, pd.Series):
            dtype = table.dtype
            table = pd.Series(labels, index, name=table.name, dtype=dtype)
        else:
            dtypes = [table[col].dtype for col in table.columns]
            labels = {
                col: pd.Series(
                    labels[:, ncol], index=index, dtype=dtypes[ncol]
                )  # supports also category
                for ncol, col in enumerate(table.columns)
            }
            table = pd.DataFrame(labels, index, columns=table.columns)
        # In case result has two channels,
        # it has two column levels
        # so we have to ensure that the table columns have the same levels
        # as result.columns
        if isinstance(table, pd.Series):
            table = table.to_frame()
        if result.columns.nlevels > 1 and table.columns.nlevels == 1:
            # Alternative: add one copy per channel to table
            # channel_tables = []
            # for channel in range(self.num_channels):
            #     channel_table = table.copy()
            #     channel_table.columns = pd.MultiIndex.from_tuples(
            #         [(channel, col) for col in table.columns]
            #     )
            #     channel_tables.append(channel_table)
            # table = pd.concat(channel_tables, axis=1)

            # Add empty level to columns
            table.columns = pd.MultiIndex.from_tuples(
                [(col, "") for col in table.columns]
            )

        result = result.join(table, lsuffix=featuresuffix, rsuffix=tablesuffix)
        return result

    def process_signal(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        *,
        file: str = None,
        start: Timestamp = None,
        end: Timestamp = None,
        process_func_args: typing.Dict[str, typing.Any] = None,
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

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        series_or_list = self.process.process_signal(
            signal,
            sampling_rate,
            file=file,
            start=start,
            end=end,
            process_func_args=process_func_args,
        ).values[0]
        series = self._reshape_series_or_list(series_or_list)
        index = series.index
        df = self._series_to_frame(series)
        if start is not None:
            start = utils.to_timedelta(start)
            # Here we change directly the levels,
            # so we need to use
            # `index.levels[0]`
            # instead of
            # `index.get_level_values('start')`
            index = index.set_levels(
                [
                    index.levels[0] + start,
                    index.levels[1] + start,
                ],
                level=[0, 1],
            )
        if file is not None:
            index = audformat.segmented_index(
                files=[file] * len(index),
                starts=index.get_level_values("start"),
                ends=index.get_level_values("end"),
            )
        df.index = index
        return df

    def process_signal_from_index(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        index: pd.Index,
        process_func_args: typing.Dict[str, typing.Any] = None,
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
        features = {col: [] for col in self.column_names}
        for df in y:
            if has_file_level:
                files.extend(df.index.get_level_values("file"))
            starts.extend(df.index.get_level_values("start"))
            ends.extend(df.index.get_level_values("end"))
            for col in self.column_names:
                features[col].extend(df[col])
        if has_file_level:
            index = audformat.segmented_index(files, starts, ends)
        else:
            index = utils.signal_index(starts, ends)
        if len(index) == 0:
            # Pass no data to ensure consistent dtype for columns
            return pd.DataFrame(index=index, columns=self.column_names)
        df = pd.DataFrame(index=index, data=features, columns=self.column_names)
        return df

    def _reshape_series_or_list(self, features):
        # If features is a list, each list element corresponds to a channel
        if isinstance(features, list):
            # Convert all channels' feature values to np.array of correct shape
            features = [
                feature.apply(lambda x: self._reshape_numpy_2d(x))
                for feature in features
            ]

            # Create combined index of segments
            combined_index = reduce(
                lambda x, y: x.union(y), [feature.index for feature in features]
            ).sort_values()

            # Extend all features to shared index
            features = [
                feature.reindex(
                    combined_index, fill_value=np.full(self.num_features, np.nan)
                )
                for feature in features
            ]

            # Combine different channels' features into a single series
            values = [np.vstack(feature.to_list()) for feature in features]
            values = np.stack(values, axis=1)
            result = pd.Series(index=combined_index, data=list(values))
            return result
        # Otherwise, features is a series
        else:
            # Each feature value is converted to np.array of correct shape
            return features.apply(lambda x: self._reshape_numpy_2d(x))

    def _reshape_numpy_2d(
        self,
        features: np.ndarray,
    ):
        r"""Reshape to [n_channels, n_features]."""
        features = np.asarray(features)
        features = np.atleast_1d(features)

        if len(features.shape) == 1:
            features = features.reshape(1, features.shape[0])
        return features

    def _series_to_frame(
        self,
        y: pd.Series,
    ) -> pd.DataFrame:
        if y.empty:
            return pd.DataFrame(
                index=audformat.segmented_index(),
                columns=self.column_names,
                dtype=object,
            )
        index = y.index
        data = [self._values_to_frame_order(values) for values in y]
        data = np.stack(data)
        df = pd.DataFrame(
            data,
            index=index,
            columns=self.column_names,
        )
        return df

    def _values_to_frame_order(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        r"""Reshape features to the expected column order of output dataframe."""
        # Assumed formats are:
        # [n_channels, n_features]
        # [n_features]
        features = self._reshape_numpy_2d(features)
        n_channels, n_features = features.shape
        # Combine features and channels into one dimension
        # f1-c1, f2-c1, ..., fN-c1, ..., f1-cM, f2-cM, ..., fN-cM
        new_shape = n_channels * n_features
        features = features.reshape(new_shape).T
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

        """
        y = self.process(
            signal,
            sampling_rate,
        )

        return self._reshape_series_or_list(y)
