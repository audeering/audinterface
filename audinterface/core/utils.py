import collections
import os
import typing

import numpy as np
import pandas as pd

import audeer
import audformat
import audresample
import audiofile as af

from audinterface.core.typing import (
    Timestamp,
    Timestamps,
)


def assert_index(obj: pd.Index):
    r"""Check if index is conform to audformat."""

    if isinstance(obj, pd.MultiIndex) and len(obj.levels) == 2:

        if obj.has_duplicates:
            max_display = 10
            duplicates = obj[obj.duplicated()]
            msg_tail = '\n...' if len(duplicates) > max_display else ''
            msg_duplicates = '\n'.join(
                [
                    str(duplicate) for duplicate
                    in duplicates[:max_display].tolist()
                ]
            )
            raise ValueError(
                'Found duplicates:\n'
                f'{msg_duplicates}{msg_tail}'
            )

        if not (
                obj.names[0] == audformat.define.IndexField.START
                and obj.names[1] == audformat.define.IndexField.END
        ):
            expected_names = [
                audformat.define.IndexField.START,
                audformat.define.IndexField.END,
            ]
            raise ValueError(
                'Found two levels with names '
                f'{obj.names}, '
                f'but expected names '
                f'{expected_names}.'
            )
        if not pd.api.types.is_timedelta64_dtype(obj.levels[0].dtype):
            raise ValueError(
                "Level 'start' must contain values of type 'timedelta64[ns]'."
            )
        if not pd.api.types.is_timedelta64_dtype(obj.levels[1].dtype):
            raise ValueError(
                "Level 'end' must contain values of type 'timedelta64[ns]'."
            )
    else:
        audformat.assert_index(obj)


def is_scalar(value: typing.Any) -> bool:
    r"""Check if value is scalar"""
    return (value is not None) and \
           (isinstance(value, str) or not hasattr(value, '__len__'))


kwargs_deprecation_warning = (
    "The use of **kwargs is deprecated "
    "and will be removed with version 1.0.0. "
    "Use 'process_func_args' instead."
)


def preprocess_signal(
        signal: np.ndarray,
        sampling_rate: int,
        expected_rate: int,
        resample: bool,
        channels: typing.Union[int, typing.Sequence[int]],
        mixdown: bool,
) -> (np.ndarray, int):
    r"""Pre-process signal."""

    signal = np.atleast_2d(signal)

    if channels is not None or mixdown:
        signal = audresample.remix(signal, channels, mixdown)

    if expected_rate is not None and sampling_rate != expected_rate:
        if resample:
            signal = audresample.resample(
                signal, sampling_rate, expected_rate,
            )
            sampling_rate = expected_rate
        else:
            raise RuntimeError(
                f'Sampling rate of input signal is '
                f'{sampling_rate} '
                f'but the expected sampling rate is '
                f'{expected_rate} Hz. '
                f'Enable resampling to avoid this error.'
            )

    return signal, sampling_rate


def read_audio(
        file: str,
        *,
        start: pd.Timedelta = None,
        end: pd.Timedelta = None,
        root: str = None,
) -> typing.Tuple[np.ndarray, int]:
    """Reads (segment of an) audio file.

    Args:
        file: path to audio file
        start: read from this position
        end: read until this position
        root: root folder

    Returns:
        * array with signal values in shape ``(channels, samples)``
        * sampling rate in Hz

    Examples:
        >>> import audb
        >>> media = audb.load_media(
        ...     'emodb',
        ...     'wav/03a01Fa.wav',
        ...     version='1.2.0',
        ...     verbose=False,
        ... )
        >>> signal, sampling_rate = read_audio(media[0], end=pd.Timedelta(0.01, unit='s'))
        >>> signal.shape
        (1, 160)

    """  # noqa: E501
    if root is not None and not os.path.isabs(file):
        file = os.path.join(root, file)

    if start is None or pd.isna(start):
        offset = 0
    else:
        offset = start.total_seconds()

    if end is None or pd.isna(end):
        duration = None
    else:
        duration = end.total_seconds() - offset

    signal, sampling_rate = af.read(
        audeer.safe_path(file),
        always_2d=True,
        offset=offset,
        duration=duration,
    )

    return signal, sampling_rate


def segment_to_indices(
        signal: np.ndarray,
        sampling_rate: int,
        start: pd.Timedelta,
        end: pd.Timedelta,
) -> typing.Tuple[int, int]:
    if pd.isna(end):
        end = pd.to_timedelta(signal.shape[-1] / sampling_rate, unit='s')
    max_i = signal.shape[-1]
    start_i = int(round(start.total_seconds() * sampling_rate))
    start_i = min(start_i, max_i)
    end_i = int(round(end.total_seconds() * sampling_rate))
    end_i = min(end_i, max_i)
    return start_i, end_i


def segments_to_indices(
        signal: np.ndarray,
        sampling_rate: int,
        index: pd.MultiIndex,
) -> typing.Tuple[typing.Sequence[int], typing.Sequence[int]]:
    starts_i = [0] * len(index)
    ends_i = [0] * len(index)
    for idx, (start, end) in enumerate(index):
        start_i, end_i = segment_to_indices(signal, sampling_rate, start, end)
        starts_i[idx] = start_i
        ends_i[idx] = end_i
    return starts_i, ends_i


def signal_index(
        starts: Timestamps = None,
        ends: Timestamps = None,
) -> pd.MultiIndex:
    r"""Create signal index.

    Returns a segmented index like
    :func:`audformat.segmented_index`,
    but without the ``'file'`` level.
    Can be used with the following methods:

    * :meth:`audinterface.Feature.process_signal_from_index`
    * :meth:`audinterface.Process.process_signal_from_index`
    * :meth:`audinterface.ProcessWithContext.process_signal_from_index`
    * :meth:`audinterface.Segment.process_signal_from_index`

    Args:
        starts: segment start positions.
            Time values given as float or integers are treated as seconds
        ends: segment end positions.
            Time values given as float or integers are treated as seconds

    Returns:
        index with start and end times

    Raises:
        ValueError: if ``start`` and ``ends`` differ in size

    Examples:
        >>> signal_index(0, 1.1)
        MultiIndex([('0 days', '0 days 00:00:01.100000')],
                   names=['start', 'end'])
        >>> signal_index('0ms', '1ms')
        MultiIndex([('0 days', '0 days 00:00:00.001000')],
                   names=['start', 'end'])
        >>> signal_index([None, 1], [1, None])
        MultiIndex([(              NaT, '0 days 00:00:01'),
                    ('0 days 00:00:01',              NaT)],
                   names=['start', 'end'])
        >>> signal_index(
        ...     starts=[0, 1],
        ...     ends=pd.to_timedelta([1000, 2000], unit='ms'),
        ... )
        MultiIndex([('0 days 00:00:00', '0 days 00:00:01'),
                    ('0 days 00:00:01', '0 days 00:00:02')],
                   names=['start', 'end'])
        >>> signal_index([0, 1])
        MultiIndex([('0 days 00:00:00', NaT),
                    ('0 days 00:00:01', NaT)],
                   names=['start', 'end'])
        >>> signal_index(ends=[1, 2])
        MultiIndex([('0 days', '0 days 00:00:01'),
                    ('0 days', '0 days 00:00:02')],
                   names=['start', 'end'])

    """
    starts = to_array(starts)
    ends = to_array(ends)

    if starts is None:
        if ends is not None:
            starts = [0] * len(ends)
        else:
            starts = []

    if ends is None:
        ends = [pd.NaT] * len(starts)

    if len(starts) != len(ends):
        raise ValueError(
            f"Cannot create index,"
            f"'starts' and 'ends' differ in length: "
            f"{len(starts)} != {len(ends)}.",
        )

    index = pd.MultiIndex.from_arrays(
        [
            pd.TimedeltaIndex(to_timedelta(starts)),
            pd.TimedeltaIndex(to_timedelta(ends)),
        ],
        names=[
            audformat.define.IndexField.START,
            audformat.define.IndexField.END,
        ],
    )
    assert_index(index)

    return index


def sliding_window(
        signal: np.ndarray,
        sampling_rate: int,
        win_dur: Timestamp,
        hop_dur: Timestamp,
) -> np.ndarray:
    r"""Reshape signal by applying a sliding window.

    Windows that do not match the specified duration
    at the end of the signals will be dropped.

    Args:
        signal: input signal in shape
            ``(samples,)``
            or ``(channels, samples)``
        sampling_rate: sampling rate in Hz
        win_dur: window duration,
            if value is as a float or integer
            it is treated as seconds.
            See :func:`audinterface.utils.to_timedelta` for further options
        hop_dur: hop duration,
            if value is as a float or integer
            it is treated as seconds.
            See :func:`audinterface.utils.to_timedelta` for further options

    Returns:
        view of signal with shape ``(channels, samples, frames)``

    Raises:
        ValueError: if ``win_dur`` or ``hop_dur``
            is smaller than ``1/sampling_rate``

    Examples:
        >>> signal = np.array(
        ...     [
        ...         [0, 1, 2, 3, 4, 5],
        ...         [0, 10, 20, 30, 40, 50],
        ...     ],
        ... )
        >>> signal
        array([[ 0,  1,  2,  3,  4,  5],
               [ 0, 10, 20, 30, 40, 50]])
        >>> frames = sliding_window(
        ...     signal,
        ...     sampling_rate=1,
        ...     win_dur=3,
        ...     hop_dur=2,
        ... )
        >>> # First frame
        >>> frames[..., 0]
        array([[ 0,  1,  2],
               [ 0, 10, 20]])
        >>> # Last frame
        >>> frames[..., -1]
        array([[ 2,  3,  4],
               [20, 30, 40]])
        >>> # Mean per frame
        >>> frames.mean(axis=1)
        array([[ 1.,  3.],
               [10., 30.]])

    """
    signal = np.atleast_2d(signal)

    win_dur = to_timedelta(win_dur, sampling_rate)
    hop_dur = to_timedelta(hop_dur, sampling_rate)
    win_length = int(win_dur.total_seconds() * sampling_rate)
    hop_length = int(hop_dur.total_seconds() * sampling_rate)

    if win_length <= 0:
        raise ValueError(
            f'When the sampling rate is '
            f'{sampling_rate} '
            f'Hz the window duration must be at least '
            f'{1.0/sampling_rate}s, '
            f'but got '
            f'{win_dur.total_seconds()}s.'
        )

    if hop_length <= 0:
        raise ValueError(
            f'When the sampling rate is '
            f'{sampling_rate} '
            f'Hz the hop duration must be at least '
            f'{1.0/sampling_rate}s, '
            f'but got '
            f'{hop_dur.total_seconds()}s.'
        )

    if signal.shape[1] < win_length:  # signal too short
        return np.array([], dtype=signal.dtype)

    shape = (signal.shape[0], signal.shape[1] - win_length + 1, win_length)
    strides = (signal.strides[0], signal.strides[1], signal.strides[1])
    frames = np.lib.stride_tricks.as_strided(
        signal,
        strides=strides,
        shape=shape,
    )[:, 0::hop_length]
    frames = frames.swapaxes(1, 2)  # make frames last axis

    return frames


def to_array(value: typing.Any) -> np.ndarray:
    r"""Convert value to numpy array."""
    if value is not None:
        if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
            value = value.to_numpy()
        elif is_scalar(value):
            value = np.array([value])
    return value


def to_timedelta(
        durations: Timestamps,
        sampling_rate: int = None,
) -> typing.Union[pd.Timedelta, typing.List[pd.Timedelta]]:
    r"""Convert duration value(s) to :class:`pandas.Timedelta`.

    If duration is given as string without unit,
    it is treated as samples
    and requires that ``'sampling_rate'`` is not ``None``.

    Args:
        durations: duration value(s).
            If value is a float or integer
            it is treated as seconds.
            To specify a unit provide as string,
            e.g. ``'2ms'``.
            To specify in samples provide as string without unit,
            e.g. ``'2000'``
        sampling_rate: sampling rate in Hz.
            Needs to be provided
            if any duration value is provided in samples

    Returns:
        duration value(s) as :class:`pandas.Timedelta` objects

    Raises:
        ValueError: if a duration value is given in samples,
            but ``sampling_rate`` is ``None``

    Examples:
        >>> to_timedelta(2)
        Timedelta('0 days 00:00:02')
        >>> to_timedelta(2.0)
        Timedelta('0 days 00:00:02')
        >>> to_timedelta('2ms')
        Timedelta('0 days 00:00:00.002000')
        >>> to_timedelta('200milliseconds')
        Timedelta('0 days 00:00:00.200000')
        >>> to_timedelta([1, '2000'], 1000)
        [Timedelta('0 days 00:00:01'), Timedelta('0 days 00:00:02')]

    """  # noqa: E501

    def convert_samples_to_seconds(time):
        if isinstance(time, str):
            # ensure we have a str and not numpy.str_
            time = str(time)
            # string without unit represents samples
            if all(t.isdigit() for t in time):
                if sampling_rate is None:
                    raise ValueError(
                        "You have to provide 'sampling_rate' "
                        "when specifying the duration in samples "
                        f"as you did with '{time}'."
                    )
                time = int(time) / sampling_rate
        return time

    if (
            not isinstance(durations, str)
            and isinstance(durations, collections.abc.Iterable)
    ):
        # sequence of duration entries
        durations = [convert_samples_to_seconds(dur) for dur in durations]
    else:
        # single duration entry
        durations = convert_samples_to_seconds(durations)

    try:
        durations = pd.to_timedelta(durations, unit='s')
    except ValueError:  # catches values like '1s'
        durations = pd.to_timedelta(durations)

    if isinstance(durations, pd.TimedeltaIndex):
        durations = list(durations)

    return durations
