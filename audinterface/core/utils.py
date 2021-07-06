import os
import typing

import numpy as np
import pandas as pd

import audeer
import audformat
import audresample
import audiofile as af

from audinterface.core.typing import Timestamps


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
        signal: array with signal values in shape ``(channels, samples)``
        sampling_rate: sampling rate in Hz

    """
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

    Example:
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
            to_timedelta(starts),
            to_timedelta(ends),
        ],
        names=[
            audformat.define.IndexField.START,
            audformat.define.IndexField.END,
        ],
    )
    assert_index(index)

    return index


def to_array(value: typing.Any) -> np.ndarray:
    r"""Convert value to numpy array."""
    if value is not None:
        if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
            value = value.to_numpy()
        elif is_scalar(value):
            value = np.array([value])
    return value


def to_timedelta(times: Timestamps):
    r"""Convert time value to pd.Timedelta."""
    try:
        return pd.to_timedelta(times, unit='s')
    except ValueError:  # catches values like '1s'
        return pd.to_timedelta(times)
