import typing

import numpy as np
import pandas as pd

import audiofile as af


def check_index(
        index: pd.MultiIndex
):
    if not len(index.levels) == 2:
        raise ValueError(f'Index has {len(index.levels)} levels, '
                         f'expected 2.')
    if not index.levels[0].dtype == 'timedelta64[ns]':
        raise ValueError(f'Level 0 has type {type(index.levels[0].dtype)}'
                         f', expected timedelta64[ns].')
    if not index.levels[1].dtype == 'timedelta64[ns]':
        raise ValueError(f'Level 0 has type {type(index.levels[0].dtype)}'
                         f', expected timedelta64[ns].')


def read_audio(
        path: str,
        start: pd.Timedelta = None,
        end: pd.Timedelta = None,
        channel: int = None,
) -> typing.Tuple[np.ndarray, int]:
    """Load audio using audiofile."""

    if start is not None:
        offset = start.total_seconds()
    else:
        offset = 0

    if end is not None:
        duration = None if pd.isna(end) else end.total_seconds() - offset
    else:
        duration = None

    # load raw audio
    signal, sampling_rate = af.read(
        path,
        always_2d=True,
        offset=offset,
        duration=duration,
    )

    # mix down
    if channel is not None:
        if channel < 0 or channel >= signal.shape[0]:
            raise ValueError(
                f'We need 0<=channel<{signal.shape[0]}, '
                f'but we have channel={channel}.'
            )
        signal = signal[channel, :]

    return signal, sampling_rate


def segment_to_indices(
        signal: np.ndarray,
        sampling_rate: int,
        start: pd.Timedelta,
        end: pd.Timedelta,
) -> typing.Tuple[int, int]:
    max_i = signal.shape[-1]
    if start is not None:
        start_i = int(round(start.total_seconds() * sampling_rate))
        start_i = min(start_i, max_i)
    else:
        start_i = 0
    if end is not None and not pd.isna(end):
        end_i = int(round(end.total_seconds() * sampling_rate))
        end_i = min(end_i, max_i)
    else:
        end_i = max_i
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
