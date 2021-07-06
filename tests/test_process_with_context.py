
import os

import numpy as np
import pandas as pd
import pytest

import audformat
import audinterface
import audiofile as af


def signal_max(signal, sampling_rate):
    return np.max(signal)


def signal_max_with_context(signal, sampling_rate, starts, ends):
    result = np.zeros(len(starts))
    for idx, (start, end) in enumerate(zip(starts, ends)):
        result[idx] = signal_max(signal[:, start:end], sampling_rate)
    return result


def test_process_index(tmpdir):

    process = audinterface.ProcessWithContext(
        process_func=None,
        sampling_rate=None,
        resample=False,
        verbose=False,
    )

    # empty

    index = audformat.segmented_index()
    result = process.process_index(index)
    assert result.empty

    # non-empty

    # create file
    sampling_rate = 8000
    signal = np.random.uniform(-1.0, 1.0, (1, 3 * sampling_rate))
    root = str(tmpdir.mkdir('wav'))
    file = 'file.wav'
    path = os.path.join(root, file)
    af.write(path, signal, sampling_rate)

    # absolute paths
    index = audformat.segmented_index(
        [path] * 3,
        pd.timedelta_range('0s', '2s', 3),
        pd.timedelta_range('1s', '3s', 3),
    )
    result = process.process_index(index)
    for (path, start, end), value in result.items():
        signal, sampling_rate = audinterface.utils.read_audio(
            path, start=start, end=end
        )
        np.testing.assert_equal(signal, value)

    # relative paths
    index = audformat.segmented_index(
        [file] * 3,
        pd.timedelta_range('0s', '2s', 3),
        pd.timedelta_range('1s', '3s', 3),
    )
    result = process.process_index(index, root=root)
    for (file, start, end), value in result.items():
        signal, sampling_rate = audinterface.utils.read_audio(
            file, start=start, end=end, root=root
        )
        np.testing.assert_equal(signal, value)


@pytest.mark.parametrize(
    'process_func,process_func_with_context,signal,sampling_rate,index',
    [
        (
            None,
            None,
            np.random.random(5 * 44100),
            44100,
            audinterface.utils.signal_index(),
        ),
        (
            None,
            None,
            np.random.random(5 * 44100),
            44100,
            audinterface.utils.signal_index(
                pd.timedelta_range('0s', '3s', 3),
                pd.timedelta_range('1s', '4s', 3),
            ),
        ),
        (
            signal_max,
            signal_max_with_context,
            np.random.random(5 * 44100),
            44100,
            audinterface.utils.signal_index(
                pd.timedelta_range('0s', '3s', 3),
                pd.timedelta_range('1s', '4s', 3),
            ),
        ),
        (
            signal_max,
            signal_max_with_context,
            np.random.random(5 * 44100),
            44100,
            audinterface.utils.signal_index(),
        ),
        pytest.param(
            signal_max,
            signal_max_with_context,
            np.random.random(5 * 44100),
            44100,
            pd.MultiIndex.from_arrays(
                [
                    pd.timedelta_range('0s', '3s', 3),
                ],
            ),
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        pytest.param(
            signal_max,
            signal_max_with_context,
            np.random.random(5 * 44100),
            44100,
            pd.MultiIndex.from_arrays(
                [
                    ['wrong', 'data', 'type'],
                    pd.timedelta_range('1s', '4s', 3),
                ],
            ),
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        pytest.param(
            signal_max,
            signal_max_with_context,
            np.random.random(5 * 44100),
            44100,
            pd.MultiIndex.from_arrays(
                [
                    pd.timedelta_range('0s', '3s', 3),
                    ['wrong', 'data', 'type'],
                ],
            ),
            marks=pytest.mark.xfail(raises=ValueError)
        ),
    ],
)
def test_process_signal_from_index(
        process_func,
        process_func_with_context,
        signal,
        sampling_rate,
        index,
):
    model = audinterface.Process(
        process_func=process_func,
        sampling_rate=None,
        resample=False,
        verbose=False,
    )
    model_with_context = audinterface.ProcessWithContext(
        process_func=process_func_with_context,
        sampling_rate=None,
        resample=False,
        verbose=False,
    )
    result = model_with_context.process_signal_from_index(
        signal, sampling_rate, index,
    )

    expected = []
    for start, end in index:
        expected.append(
            model.process_signal(signal, sampling_rate, start=start, end=end)
        )
    if not expected:
        pd.testing.assert_series_equal(
            result,
            pd.Series([], index, dtype=float),
        )
    else:
        pd.testing.assert_series_equal(
            result,
            pd.concat(expected, names=['start', 'end']),
        )


@pytest.mark.parametrize(
    'signal_sampling_rate,target_rate,resample',
    [
        pytest.param(
            44100,
            None,
            True,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            44100,
            44100,
            True,
        ),
        (
            44100,
            44100,
            False,
        ),
        pytest.param(
            48000,
            44100,
            False,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        (
            4,
            3,
            True,
        ),
        (
            41000,
            None,
            False,
        ),
    ],
)
def test_sampling_rate_mismatch(
        signal_sampling_rate,
        target_rate,
        resample,
):
    model = audinterface.ProcessWithContext(
        process_func=None,
        sampling_rate=target_rate,
        resample=resample,
        verbose=False,
    )
    signal = np.random.random(5 * 44100)
    index = audinterface.utils.signal_index(
        pd.timedelta_range('0s', '3s', 3),
        pd.timedelta_range('1s', '4s', 3),
    )
    model.process_signal_from_index(signal, signal_sampling_rate, index)
