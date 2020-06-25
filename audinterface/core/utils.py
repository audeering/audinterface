import concurrent.futures
import typing

import numpy as np
import pandas as pd

import audeer
import audiofile as af


def check_index(
        index: pd.MultiIndex
):
    if len(index.levels) == 2:
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
    elif len(index.levels) == 3:
        if not index.names == ('file', 'start', 'end'):
            raise ValueError('Not a segmented index conform to Unified Format')
        if not index.empty:
            if not pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
                    index.levels[1]
            ):
                raise ValueError(f'Level 1 has type '
                                 f'{type(index.levels[1].dtype)}'
                                 f', expected timedelta64[ns].')
            if not pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
                    index.levels[2]
            ):
                raise ValueError(f'Level 2 has type '
                                 f'{type(index.levels[2].dtype)}'
                                 f', expected timedelta64[ns].')
    else:
        raise ValueError(f'Index has {len(index.levels)} levels, '
                         f'expected 2 or 3.')


def read_audio(
        path: str,
        start: pd.Timedelta = None,
        end: pd.Timedelta = None,
        channel: int = None,
) -> typing.Tuple[np.ndarray, int]:
    """Reads (segment of an) audio file.

    Args:
        path: path to audio file
        start: read from this position
        end: read until this position
        channel: channel number

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

    # load raw audio
    signal, sampling_rate = af.read(
        audeer.safe_path(path),
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


def run_tasks(
        task_func: typing.Callable,
        params: typing.Sequence[
            typing.Tuple[
                typing.Sequence[typing.Any],
                typing.Dict[str, typing.Any],
            ]
        ],
        *,
        num_workers: int = None,
        multiprocessing: bool = False,
        progress_bar: bool = False,
        task_description: str = None
) -> typing.Sequence[typing.Any]:
    r"""Run parallel tasks using multprocessing.

    .. note:: Result values are returned in order of ``params``.

    Args:
        task_func: task function with one or more
            parameters, e.g. ``x, y, z``, and optionally returning a value
        params: sequence of tuples holding parameters for each task.
            Each tuple contains a sequence of positional arguments and a
            dictionary with keyword arguments, e.g.:
            ``[((x1, y1), {'z': z1}), ((x2, y2), {'z': z2}), ...]``
        num_workers: number of parallel jobs or 1 for sequential
            processing. If ``None`` will be set to the number of
            processors on the machine multiplied by 5 in case of
            multithreading and number of processors in case of
            multiprocessing
        multiprocessing: use multiprocessing instead of multithreading
        progress_bar: show a progress bar
        task_description: task description
            that will be displayed next to progress bar

    Example:
        >>> power = lambda x, n: x ** n
        >>> params = [([2, n], {}) for n in range(10)]
        >>> run_tasks(power, params, num_workers=3)
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    """
    num_tasks = max(1, len(params))
    results = [None] * num_tasks

    if num_workers == 1:  # sequential

        with audeer.progress_bar(
            params,
            total=len(params),
            desc=task_description,
            disable=not progress_bar,
        ) as pbar:
            for index, param in enumerate(pbar):
                results[index] = task_func(*param[0], **param[1])

    else:  # parallel

        if multiprocessing:
            executor = concurrent.futures.ProcessPoolExecutor
        else:
            executor = concurrent.futures.ThreadPoolExecutor
        with executor(max_workers=num_workers) as pool:
            with audeer.progress_bar(
                    total=len(params),
                    desc=task_description,
                    disable=not progress_bar,
            ) as pbar:
                futures = []
                for param in params:
                    future = pool.submit(task_func, *param[0], **param[1])
                    future.add_done_callback(lambda p: pbar.update())
                    futures.append(future)
                for idx, future in enumerate(futures):
                    result = future.result()
                    results[idx] = result

    return results


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
