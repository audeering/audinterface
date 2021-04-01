import typing

import numpy as np
import pandas as pd

import audeer
import audformat
import audresample
import audiofile as af

from audinterface.core.typing import Timestamps


def check_index(index: pd.Index):
    r"""Check if index is conform to audformat."""
    if isinstance(index, pd.MultiIndex) and len(index.levels) == 2:
        if not index.empty:
            if not pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
                    index.levels[0]
            ):
                raise ValueError(f'Level 0 has type '
                                 f'{type(index.levels[0].dtype)}'
                                 f', expected timedelta64[ns].')
            if not pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
                    index.levels[1]
            ):
                raise ValueError(f'Level 1 has type '
                                 f'{type(index.levels[1].dtype)}'
                                 f', expected timedelta64[ns].')
    else:
        audformat.assert_index(index)


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
        path: str,
        start: pd.Timedelta = None,
        end: pd.Timedelta = None,
) -> typing.Tuple[np.ndarray, int]:
    """Reads (segment of an) audio file.

    Args:
        path: path to audio file
        start: read from this position
        end: read until this position

    Returns:
        signal: array with signal values in shape ``(channels, samples)``
        sampling_rate: sampling rate in Hz

    """
    if start is None or pd.isna(start):
        offset = 0
    else:
        offset = start.total_seconds()

    if end is None or pd.isna(end):
        duration = None
    else:
        duration = end.total_seconds() - offset

    signal, sampling_rate = af.read(
        audeer.safe_path(path),
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
        end = pd.to_timedelta(
            signal.shape[-1] / sampling_rate, unit='sec'
        )
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


def to_timedelta(times: Timestamps):
    r"""Convert time value to pd.Timedelta."""
    try:
        return pd.to_timedelta(times, unit='s')
    except ValueError:  # catches values like '1s'
        return pd.to_timedelta(times)
