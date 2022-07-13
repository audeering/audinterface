import os

import numpy as np
import pandas as pd
import pytest

import audformat
import audinterface
import audiofile


def signal_max(signal, sampling_rate):
    return np.max(signal)


def signal_max_with_context(signal, sampling_rate, starts, ends):
    result = np.zeros(len(starts))
    for idx, (start, end) in enumerate(zip(starts, ends)):
        result[idx] = signal_max(signal[:, start:end], sampling_rate)
    return result


def test_process_func_args():
    def process_func(s, sr, starts, ends, arg1, arg2):
        assert arg1 == 'foo'
        assert arg2 == 'bar'
    audinterface.ProcessWithContext(
        process_func=process_func,
        process_func_args={
            'arg1': 'foo',
            'arg2': 'bar',
        }
    )
    with pytest.warns(UserWarning):
        audinterface.ProcessWithContext(
            feature_names=('o1', 'o2', 'o3'),
            process_func=process_func,
            arg1='foo',
            arg2='bar',
        )


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
    audiofile.write(path, signal, sampling_rate)

    # absolute paths
    index = audformat.segmented_index(
        [path] * 3,
        pd.timedelta_range('0s', '2s', 3),
        pd.timedelta_range('1s', '3s', 3),
    )
    result = process.process_index(index)
    for (path, start, end), value in result.items():
        x, sampling_rate = audinterface.utils.read_audio(
            path, start=start, end=end
        )
        np.testing.assert_equal(x, value)

    # relative paths
    index = audformat.segmented_index(
        [file] * 3,
        pd.timedelta_range('0s', '2s', 3),
        pd.timedelta_range('1s', '3s', 3),
    )
    result = process.process_index(index, root=root)
    for (file, start, end), value in result.items():
        x, sampling_rate = audinterface.utils.read_audio(
            file, start=start, end=end, root=root
        )
        np.testing.assert_equal(x, value)

    # multiple channels
    signal = np.concatenate([signal] * 3)
    audiofile.write(path, signal, sampling_rate)
    result = process.process_index(index, root=root)
    for (file, start, end), value in result.items():
        x, sampling_rate = audinterface.utils.read_audio(
            file, start=start, end=end, root=root
        )
        np.testing.assert_equal(x, value)

    # specific channels
    channels = [0, 2]
    process = audinterface.ProcessWithContext(
        process_func=None,
        sampling_rate=None,
        resample=False,
        channels=channels,
        verbose=False,
    )
    result = process.process_index(index, root=root)
    for (file, start, end), value in result.items():
        x, sampling_rate = audinterface.utils.read_audio(
            file, start=start, end=end, root=root
        )
        np.testing.assert_equal(x[channels, :], value)

    # specific channel
    channel = 1
    process = audinterface.ProcessWithContext(
        process_func=None,
        sampling_rate=None,
        resample=False,
        channels=channel,
        verbose=False,
    )
    result = process.process_index(index, root=root)
    for (file, start, end), value in result.items():
        x, sampling_rate = audinterface.utils.read_audio(
            file, start=start, end=end, root=root
        )
        np.testing.assert_equal(np.atleast_2d(x[channel]), value)


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
        pytest.param(  # process_func returns None
            lambda signal, sampling_rate: None,
            lambda signal, sampling_rate, starts, ends: None,
            np.random.random(5 * 44100),
            44100,
            audinterface.utils.signal_index(
                pd.timedelta_range('0s', '3s', 3),
                pd.timedelta_range('1s', '4s', 3),
            ),
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        pytest.param(  # process_func returns single value
            lambda signal, sampling_rate: 0,
            lambda signal, sampling_rate, starts, ends: 0,
            np.random.random(5 * 44100),
            44100,
            audinterface.utils.signal_index(
                pd.timedelta_range('0s', '3s', 3),
                pd.timedelta_range('1s', '4s', 3),
            ),
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        pytest.param(  # process_func returns array with wrong length
            lambda signal, sampling_rate: [0, 1],
            lambda signal, sampling_rate, starts, ends: [0, 1],
            np.random.random(5 * 44100),
            44100,
            audinterface.utils.signal_index(
                pd.timedelta_range('0s', '3s', 3),
                pd.timedelta_range('1s', '4s', 3),
            ),
            marks=pytest.mark.xfail(raises=RuntimeError),
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
            pd.Series([], index, dtype=object),
        )
    else:
        pd.testing.assert_series_equal(
            result,
            pd.concat(expected, names=['start', 'end']),
        )


def test_process_with_special_args(tmpdir):

    duration = 3
    sampling_rate = 1
    signal = np.zeros((2, duration), np.float32)
    num_files = 10
    win_dur = 1
    num_frames = duration // win_dur

    # create files
    root = tmpdir
    files = [f'f{idx}.wav' for idx in range(num_files)]
    for file in files:
        path = os.path.join(root, file)
        audiofile.write(path, signal, sampling_rate, bit_depth=32)

    # create interface
    def process_func(signal, sampling_rate, starts, ends, idx, file, root):
        return [(idx, file, root)] * len(starts)

    process = audinterface.ProcessWithContext(process_func=process_func)

    # process signal from index
    index = audinterface.utils.signal_index(
        range(num_frames),
        range(1, num_frames + 1),
    )
    y = process.process_signal_from_index(
        signal,
        sampling_rate,
        index,
    )
    expected = pd.Series(
        [(0, None, None)] * len(index),
        index,
    )
    pd.testing.assert_series_equal(y, expected)

    # process index
    index = audformat.segmented_index(
        np.repeat(files, num_frames),
        np.tile(range(num_frames), num_files),
        np.tile(range(1, num_frames + 1), num_files),
    )
    y = process.process_index(index, root=root)
    values = []
    for idx in range(num_files):
        file = files[idx]
        for _ in range(num_frames):
            values.append((idx, file, root))
    expected = pd.Series(values, index)
    pd.testing.assert_series_equal(y, expected)

    # explicitely pass special arguments

    process = audinterface.ProcessWithContext(
        process_func=process_func,
        process_func_args={'idx': 99, 'file': 'my/file', 'root': None},
    )
    y = process.process_index(index, root=root)
    expected = pd.Series([(99, 'my/file', None)] * len(index), index)
    pd.testing.assert_series_equal(y, expected)


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
    signal = np.random.random(5 * 44100).astype('float32')
    index = audinterface.utils.signal_index(
        pd.timedelta_range('0s', '3s', 3),
        pd.timedelta_range('1s', '4s', 3),
    )
    model.process_signal_from_index(signal, signal_sampling_rate, index)
