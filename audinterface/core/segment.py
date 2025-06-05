from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import errno
import os

import numpy as np
import pandas as pd

import audeer
import audformat

from audinterface.core import utils
from audinterface.core.typing import Timestamp
from audinterface.core.typing import Timestamps


def signal_index(signal, sampling_rate, **kwargs) -> pd.MultiIndex:
    r"""Default segment process function.

    This function is used,
    when ``Segment`` is instantiated
    with ``process_func=None``.
    It returns an empty multi-index,
    with levels ``start`` and ``end``.

    Args:
        signal: signal
        sampling_rate: sampling rate in Hz
        **kwargs: additional keyword arguments of the processing function

    Returns:
        index with segments

    """
    return utils.signal_index()


def inverted_process_func(
    signal,
    sampling_rate,
    *,
    __process_func,
    **kwargs,
) -> pd.MultiIndex:
    r"""Inverted segment process function.

    This function is used,
    when ``Segment`` is instantiated
    with ``invert=True``.

    Args:
        signal: signal
        sampling_rate: sampling rate in Hz
        __process_func: process func to invert.
            Note, ``__process_func`` needs to be added to ``process_func_args``
            before calling this function.
            This means, a user cannot use ``__process_func``
            as argument name
            in ``process_func``
        **kwargs: additional keyword arguments of the processing function

    Returns:
        index with segments

    """
    index = __process_func(signal, sampling_rate, **kwargs)
    duration = pd.to_timedelta(signal.shape[-1] / sampling_rate, unit="s")
    index = index.sortlevel("start")[0]
    index = merge_index(index)
    index = invert_index(index, duration)
    return index


def invert_index(
    index: pd.MultiIndex,
    dur: pd.Timedelta,
) -> pd.MultiIndex:
    r"""Invert index.

    Assumes that index is sorted by 'start' level.

    """
    if index.empty:
        return utils.signal_index(0, dur)

    starts = index.get_level_values("start")
    ends = index.get_level_values("end")
    new_starts = ends[:-1]
    new_ends = starts[1:]
    if starts[0] != pd.to_timedelta(0):
        new_starts = new_starts.insert(0, pd.to_timedelta(0))
        new_ends = new_ends.insert(0, starts[0])
    if ends[-1] != dur:
        new_starts = new_starts.insert(len(new_starts), ends[-1])
        new_ends = new_ends.insert(len(new_ends), dur)
    return utils.signal_index(new_starts, new_ends)


def merge_index(
    index: pd.MultiIndex,
) -> pd.MultiIndex:
    r"""Merge overlapping segments.

    Assumes that index is sorted by 'start' level.

    """
    if index.empty:
        return index

    starts = index.get_level_values("start")
    ends = index.get_level_values("end")
    new_starts = []
    new_ends = []
    new_start = starts[0]
    new_end = ends[0]
    for start, end in zip(starts[1:], ends[1:]):
        if start > new_end:
            new_starts.append(new_start)
            new_ends.append(new_end)
            new_start = start
            new_end = end
        elif end > new_end:
            new_end = end
    new_starts.append(new_start)
    new_ends.append(new_end)

    return utils.signal_index(new_starts, new_ends)


class Segment:
    r"""Segmentation interface.

    Interface for models that apply a segmentation to the input signal,
    e.g. a voice activity model that detects speech regions.

    Args:
        process_func: segmentation function,
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
            Must return a :class:`pandas.MultiIndex` with two levels
            named `start` and `end` that hold start and end
            positions as :class:`pandas.Timedelta` objects
        process_func_args: (keyword) arguments passed on to the processing
            function
        invert: Invert the segmentation
        sampling_rate: sampling rate in Hz
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
        >>> def segment(signal, sampling_rate, *, win_size=0.2, hop_size=0.1):
        ...     size = signal.shape[1] / sampling_rate
        ...     starts = pd.to_timedelta(np.arange(0, size - win_size, hop_size), unit="s")
        ...     ends = pd.to_timedelta(np.arange(win_size, size, hop_size), unit="s")
        ...     return pd.MultiIndex.from_tuples(zip(starts, ends), names=["start", "end"])
        >>> interface = Segment(process_func=segment)
        >>> signal = np.array([1.0, 2.0, 3.0])
        >>> interface(signal, sampling_rate=3)
        MultiIndex([(       '0 days 00:00:00', '0 days 00:00:00.200000'),
                    ('0 days 00:00:00.100000', '0 days 00:00:00.300000'),
                    ('0 days 00:00:00.200000', '0 days 00:00:00.400000'),
                    ('0 days 00:00:00.300000', '0 days 00:00:00.500000'),
                    ('0 days 00:00:00.400000', '0 days 00:00:00.600000'),
                    ('0 days 00:00:00.500000', '0 days 00:00:00.700000'),
                    ('0 days 00:00:00.600000', '0 days 00:00:00.800000'),
                    ('0 days 00:00:00.700000', '0 days 00:00:00.900000')],
                   names=['start', 'end'])
        >>> # Apply interface on an audformat conform index of a dataframe
        >>> import audb
        >>> db = audb.load(
        ...     "emodb",
        ...     version="1.3.0",
        ...     media="wav/03a01Fa.wav",
        ...     full_path=False,
        ...     verbose=False,
        ... )
        >>> interface = Segment(
        ...     process_func=segment,
        ...     process_func_args={"win_size": 0.5, "hop_size": 0.25},
        ... )
        >>> interface.process_index(db["emotion"].index, root=db.root)
        MultiIndex([('wav/03a01Fa.wav',        '0 days 00:00:00', ...),
                    ('wav/03a01Fa.wav', '0 days 00:00:00.250000', ...),
                    ('wav/03a01Fa.wav', '0 days 00:00:00.500000', ...),
                    ('wav/03a01Fa.wav', '0 days 00:00:00.750000', ...),
                    ('wav/03a01Fa.wav',        '0 days 00:00:01', ...),
                    ('wav/03a01Fa.wav', '0 days 00:00:01.250000', ...)],
                   names=['file', 'start', 'end'])

    """  # noqa: E501

    def __init__(
        self,
        *,
        process_func: Callable[..., pd.MultiIndex] | None = None,
        process_func_args: dict[str, object] | None = None,
        invert: bool = False,
        sampling_rate: int | None = None,
        resample: bool = False,
        channels: int | Sequence[int] | None = None,
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

        if process_func is None:
            process_func = signal_index

        if invert:
            process_func_args = process_func_args or {}
            process_func_args["__process_func"] = process_func
            process_func = inverted_process_func

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

        self.process = process
        r"""Processing object."""

        self.invert = invert
        r"""Invert segmentation."""

    def process_file(
        self,
        file: str,
        *,
        start: Timestamp | None = None,
        end: Timestamp | None = None,
        root: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.Index:
        r"""Segment the content of an audio file.

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
                :attr:`audinterface.Segment.process.process_func_args`

        Returns:
            Segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        if start is None or pd.isna(start):
            start = pd.to_timedelta(0)
        index = self.process.process_file(
            file,
            start=start,
            end=end,
            root=root,
            process_func_args=process_func_args,
        ).values[0]
        return audformat.segmented_index(
            files=[file] * len(index),
            starts=index.get_level_values("start") + start,
            ends=index.get_level_values("end") + start,
        )

    def process_files(
        self,
        files: Sequence[str],
        *,
        starts: Timestamps | None = None,
        ends: Timestamps | None = None,
        root: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.Index:
        r"""Segment a list of files.

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
                :attr:`audinterface.Segment.process.process_func_args`

        Returns:
            Segmented index conform to audformat_

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
        if len(y) == 0:
            return audformat.filewise_index()

        files = []
        starts = []
        ends = []
        for (file, start, _), index in y.items():
            files.extend([file] * len(index))
            starts.extend(index.get_level_values("start") + start)
            ends.extend(index.get_level_values("end") + start)

        return audformat.segmented_index(files, starts, ends)

    def process_folder(
        self,
        root: str,
        *,
        filetype: str = "wav",
        include_root: bool = True,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.Index:
        r"""Segment files in a folder.

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
                :attr:`audinterface.Segment.process.process_func_args`

        Returns:
            Segmented index conform to audformat_

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
        root: str | None = None,
        cache_root: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.Index:
        r"""Segment files or segments from an index.

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
                :attr:`audinterface.Segment.process.process_func_args`

        Returns:
            Segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        index = audformat.utils.to_segmented_index(index)
        utils.assert_index(index)

        if index.empty:
            return index

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
        for (file, start, _), index in y.items():
            files.extend([file] * len(index))
            starts.extend(index.get_level_values("start") + start)
            ends.extend(index.get_level_values("end") + start)

        return audformat.segmented_index(files, starts, ends)

    def process_table(
        self,
        table: pd.Series | pd.DataFrame,
        *,
        root: str | None = None,
        cache_root: str | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.Series | pd.DataFrame:
        r"""Segment files or segments from a table.

        The labels of the table
        are reassigned to the new segments.

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
                :attr:`audinterface.Segment.process.process_func_args`

        Returns:
            Segmented table with an index conform to audformat_

        Raises:
            ValueError: if table is not a :class:`pandas.Series`
                or a :class:`pandas.DataFrame`
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        if not isinstance(table, pd.Series) and not isinstance(table, pd.DataFrame):
            raise ValueError("table has to be pd.Series or pd.DataFrame")

        index = audformat.utils.to_segmented_index(table.index)
        utils.assert_index(index)

        if index.empty:
            return table

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
        if isinstance(table, pd.Series):
            for n, ((file, start, _), index) in enumerate(y.items()):
                files.extend([file] * len(index))
                starts.extend(index.get_level_values("start") + start)
                ends.extend(index.get_level_values("end") + start)
                labels.extend([[table.iloc[n]] * len(index)])
            labels = np.hstack(labels)
        else:
            for n, ((file, start, _), index) in enumerate(y.items()):
                files.extend([file] * len(index))
                starts.extend(index.get_level_values("start") + start)
                ends.extend(index.get_level_values("end") + start)
                if len(index) > 0:  # avoid issues when stacking 0-length dataframes
                    labels.extend([[table.iloc[n].values] * len(index)])
            if len(labels) > 0:
                labels = np.vstack(labels)
            else:
                labels = np.empty((0, table.shape[1]))  # avoid issue below

        index = audformat.segmented_index(files, starts, ends)

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
            table = pd.DataFrame(labels, index)

        return table

    def process_signal(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        *,
        file: str | None = None,
        start: Timestamp | None = None,
        end: Timestamp | None = None,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.Index:
        r"""Segment audio signal.

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
                :attr:`audinterface.Segment.process.process_func_args`

        Returns:
            Segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        index = self.process.process_signal(
            signal,
            sampling_rate,
            file=file,
            start=start,
            end=end,
            process_func_args=process_func_args,
        ).values[0]
        utils.assert_index(index)
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

        return index

    def process_signal_from_index(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        index: pd.Index,
        process_func_args: dict[str, object] | None = None,
    ) -> pd.Index:
        r"""Segment parts of a signal.

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
                :attr:`audinterface.Segment.process.process_func_args`

        Returns:
            Segmented index conform to audformat_

        Raises:
            RuntimeError: if sampling rates do not match
            RuntimeError: if channel selection is invalid
            ValueError: if index contains duplicates

        .. _audformat: https://audeering.github.io/audformat/data-format.html

        """
        utils.assert_index(index)

        if index.empty:
            return index

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

        for idx in y:
            if has_file_level:
                files.extend(idx.get_level_values("file"))
            starts.extend(idx.get_level_values("start"))
            ends.extend(idx.get_level_values("end"))

        if has_file_level:
            index = audformat.segmented_index(files, starts, ends)
        else:
            index = utils.signal_index(starts, ends)

        return index

    def __call__(
        self,
        signal: np.ndarray,
        sampling_rate: int,
    ) -> pd.Index:
        r"""Apply processing to signal.

        This function processes the signal **without** transforming the output
        into a :class:`pd.MultiIndex`. Instead, it will return the raw
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
        return self.process(
            signal,
            sampling_rate,
        )
