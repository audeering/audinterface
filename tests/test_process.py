import os

import audiofile
import audiofile as af
import numpy as np
import pandas as pd
import pytest

import audinterface
import audformat
import audobject


def signal_duration(signal, sampling_rate):
    return signal.shape[1] / sampling_rate


def signal_max(signal, sampling_rate):
    return np.max(signal)


class SignalObject(audobject.Object):

    def __call__(self, signal, sampling_rate):
        return np.max(signal)


SEGMENT = audinterface.Segment(
    process_func=lambda x, sr:
        audinterface.utils.signal_index(
            pd.to_timedelta(0),
            pd.to_timedelta(x.shape[1] / sr, unit='s') / 2,
        )
)


def signal_modification(signal, sampling_rate, subtract=False):
    if subtract:
        signal -= 0.1 * signal
    else:
        signal += 0.1 * signal
    return signal


@pytest.mark.parametrize(
    'process_func, segment, signal, sampling_rate, start, end, keep_nat, '
    'channels, mixdown, expected_output',
    [
        (
            signal_max,
            None,
            np.ones((1, 3)),
            8000,
            None,
            None,
            False,
            None,
            False,
            1,
        ),
        (
            signal_max,
            SEGMENT,
            np.ones((1, 8000)),
            8000,
            None,
            None,
            False,
            None,
            False,
            1,
        ),
        (
            signal_max,
            None,
            np.ones(3),
            8000,
            None,
            None,
            False,
            0,
            False,
            1,
        ),
        (
            signal_max,
            None,
            np.array([[0., 0., 0.], [1., 1., 1.]]),
            8000,
            None,
            None,
            False,
            0,
            False,
            0,
        ),
        (
            signal_max,
            None,
            np.array([[0., 0., 0.], [1., 1., 1.]]),
            8000,
            None,
            None,
            False,
            0,
            False,
            0,
        ),
        (
            signal_max,
            None,
            np.array([[0., 0., 0.], [1., 1., 1.]]),
            8000,
            None,
            None,
            False,
            1,
            False,
            1,
        ),
        (
            signal_max,
            None,
            np.array([[0., 0., 0.], [1., 1., 1.]]),
            8000,
            None,
            None,
            False,
            None,
            True,
            0.5,
        ),
        (
            signal_max,
            None,
            np.array([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]),
            8000,
            None,
            None,
            False,
            [1, 2],
            True,
            0.5,
        ),
        # invalid channel selection
        pytest.param(
            signal_max,
            None,
            np.ones((1, 3)),
            8000,
            None,
            None,
            False,
            1,
            False,
            1,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            signal_duration,
            None,
            np.zeros((1, 24000)),
            8000,
            None,
            None,
            False,
            None,
            False,
            3.0,
        ),
        (
            signal_duration,
            None,
            np.zeros((1, 24000)),
            8000,
            pd.NaT,
            pd.NaT,
            False,
            None,
            False,
            3.0,
        ),
        (
            signal_duration,
            None,
            np.zeros((1, 24000)),
            8000,
            pd.NaT,
            pd.NaT,
            True,
            None,
            False,
            3.0,
        ),
        (
            signal_duration,
            None,
            np.zeros((1, 24000)),
            8000,
            pd.to_timedelta('1s'),
            None,
            False,
            None,
            False,
            2.0,
        ),
        (
            signal_duration,
            None,
            np.zeros((1, 24000)),
            8000,
            pd.to_timedelta('1s'),
            pd.NaT,
            False,
            None,
            False,
            2.0,
        ),
        (
            signal_duration,
            None,
            np.zeros((1, 24000)),
            8000,
            None,
            pd.to_timedelta('2s'),
            False,
            None,
            False,
            2.0,
        ),
        (
            signal_duration,
            None,
            np.zeros((1, 24000)),
            8000,
            pd.NaT,
            pd.to_timedelta('2s'),
            False,
            None,
            False,
            2.0,
        ),
        (
            signal_duration,
            None,
            np.zeros((1, 24000)),
            8000,
            pd.to_timedelta('1s'),
            pd.to_timedelta('2s'),
            False,
            None,
            False,
            1.0,
        ),
        (
            signal_duration,
            None,
            np.zeros((1, 24000)),
            8000,
            '1s',
            '2s',
            False,
            None,
            False,
            1.0,
        ),
        (
            signal_duration,
            None,
            np.zeros((1, 24000)),
            8000,
            '1000ms',
            '2000ms',
            False,
            None,
            False,
            1.0,
        ),
        (
            signal_duration,
            None,
            np.zeros((1, 24000)),
            8000,
            1,
            2,
            False,
            None,
            False,
            1.0,
        ),
        (
            signal_duration,
            None,
            np.zeros((1, 24000)),
            8000,
            1.0,
            2.0,
            False,
            None,
            False,
            1.0,
        ),
    ],
)
def test_process_file(
    tmpdir,
    process_func,
    segment,
    signal,
    sampling_rate,
    start,
    end,
    keep_nat,
    channels,
    mixdown,
    expected_output,
):
    process = audinterface.Process(
        process_func=process_func,
        sampling_rate=sampling_rate,
        resample=False,
        channels=channels,
        mixdown=mixdown,
        segment=segment,
        keep_nat=keep_nat,
        verbose=False,
    )

    # create test file
    root = str(tmpdir.mkdir('wav'))
    file = 'file.wav'
    path = os.path.join(root, file)
    af.write(path, signal, sampling_rate, bit_depth=32)

    # test absolute path
    y = process.process_file(
        path,
        start=start,
        end=end,
    )
    np.testing.assert_almost_equal(
        y.values, expected_output, decimal=4,
    )

    # test relative path
    y = process.process_file(
        file,
        start=start,
        end=end,
        root=root,
    )
    np.testing.assert_almost_equal(
        y.values, expected_output, decimal=4,
    )


@pytest.mark.parametrize(
    'process_func, num_files, signal, sampling_rate, starts, ends, '
    'expected_output',
    [
        (
            signal_duration,
            2,
            np.zeros((1, 24000)),
            8000,
            None,
            None,
            [3.0] * 2,
        ),
        (
            signal_duration,
            2,
            np.zeros((1, 24000)),
            8000,
            '1s',
            '2s',
            [1.0] * 2,
        ),
        (
            signal_duration,
            2,
            np.zeros((1, 24000)),
            8000,
            1,
            2,
            [1.0] * 2,
        ),
        (
            signal_duration,
            2,
            np.zeros((1, 24000)),
            8000,
            [None, 1],
            [None, 2],
            [3.0, 1.0],
        ),
        (
            signal_duration,
            2,
            np.zeros((1, 24000)),
            8000,
            [None, '1s'],
            [None, '2s'],
            [3.0, 1.0],
        ),
        (
            signal_duration,
            3,
            np.zeros((1, 24000)),
            8000,
            [None, '1s'],
            [None, '2s', None],
            [3.0, 1.0],
        ),
        (
            signal_duration,
            1,
            np.zeros((1, 24000)),
            8000,
            [None],
            [None, '2s'],
            [3.0],
        ),
    ],
)
def test_process_files(
    tmpdir,
    process_func,
    num_files,
    signal,
    sampling_rate,
    starts,
    ends,
    expected_output,
):
    process = audinterface.Process(
        process_func=process_func,
        sampling_rate=sampling_rate,
        resample=False,
        verbose=False,
    )

    # create files
    files = []
    paths = []
    root = tmpdir
    for idx in range(num_files):
        file = f'file{idx}.wav'
        path = os.path.join(root, file)
        af.write(path, signal, sampling_rate)
        files.append(file)
        paths.append(path)

    # test absolute paths
    output = process.process_files(
        paths,
        starts=starts,
        ends=ends,
    )
    np.testing.assert_almost_equal(
        output.values,
        expected_output,
        decimal=4,
    )

    # test relative paths
    output = process.process_files(
        files,
        starts=starts,
        ends=ends,
        root=root,
    )
    np.testing.assert_almost_equal(
        output.values,
        expected_output,
        decimal=4,
    )


@pytest.mark.parametrize(
    'num_files, segment, num_workers, multiprocessing',
    [
        (3, None, 1, False, ),
        (3, None, 2, False, ),
        (3, None, None, False, ),
        (3, SEGMENT, 1, False, ),
    ]
)
def test_process_folder(
        tmpdir,
        num_files,
        segment,
        num_workers,
        multiprocessing,
):
    process = audinterface.Process(
        process_func=None,
        sampling_rate=None,
        resample=False,
        segment=segment,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
        verbose=False,
    )
    sampling_rate = 8000
    root = str(tmpdir.mkdir('wav'))
    files = [
        os.path.join(root, f'file{n}.wav') for n in range(num_files)
    ]
    for file in files:
        signal = np.random.uniform(-1.0, 1.0, (1, sampling_rate))
        af.write(file, signal, sampling_rate)
    y = process.process_folder(root)
    pd.testing.assert_series_equal(
        y,
        process.process_files(files),
    )

    # non-existing folder
    with pytest.raises(FileNotFoundError):
        process.process_folder('bad-folder')

    # empty folder
    root = str(tmpdir.mkdir('empty'))
    y = process.process_folder(root)
    pd.testing.assert_series_equal(y, pd.Series(dtype=object))


def test_process_func_args():
    def process_func(s, sr, arg1, arg2):
        assert arg1 == 'foo'
        assert arg2 == 'bar'
    audinterface.Process(
        process_func=process_func,
        process_func_args={
            'arg1': 'foo',
            'arg2': 'bar',
        }
    )
    with pytest.warns(UserWarning):
        audinterface.Process(
            feature_names=('o1', 'o2', 'o3'),
            process_func=process_func,
            arg1='foo',
            arg2='bar',
        )


@pytest.mark.parametrize(
    'num_workers, multiprocessing',
    [
        (1, False, ),
        (2, False, ),
        (None, False, ),
    ]
)
def test_process_index(tmpdir, num_workers, multiprocessing):

    cache_root = os.path.join(tmpdir, 'cache')

    process = audinterface.Process(
        process_func=None,
        sampling_rate=None,
        resample=False,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
        verbose=False,
    )
    sampling_rate = 8000
    signal = np.random.uniform(-1.0, 1.0, (1, 3 * sampling_rate))

    # create file
    root = str(tmpdir.mkdir('wav'))
    file = 'file.wav'
    path = os.path.join(root, file)
    af.write(path, signal, sampling_rate)

    # empty index
    index = audformat.segmented_index()
    y = process.process_index(index)
    assert y.empty

    # segmented index with absolute paths
    index = audformat.segmented_index(
        [path] * 3,
        pd.timedelta_range('0s', '2s', 3),
        pd.timedelta_range('1s', '3s', 3),
    )
    y = process.process_index(index)
    for (path, start, end), value in y.items():
        signal, sampling_rate = audinterface.utils.read_audio(
            path, start=start, end=end
        )
        np.testing.assert_equal(signal, value)

    # filewise index with absolute paths
    index = audformat.filewise_index(path)
    y = process.process_index(index)
    for (path, start, end), value in y.items():
        signal, sampling_rate = audinterface.utils.read_audio(
            path, start=start, end=end
        )
        np.testing.assert_equal(signal, value)

    # segmented index with relative paths
    index = audformat.segmented_index(
        [file] * 3,
        pd.timedelta_range('0s', '2s', 3),
        pd.timedelta_range('1s', '3s', 3),
    )
    y = process.process_index(index, root=root)
    for (file, start, end), value in y.items():
        signal, sampling_rate = audinterface.utils.read_audio(
            file, start=start, end=end, root=root
        )
        np.testing.assert_equal(signal, value)

    # filewise index with relative paths
    index = audformat.filewise_index(path)
    y = process.process_index(index, root=root)
    for (file, start, end), value in y.items():
        signal, sampling_rate = audinterface.utils.read_audio(
            file, start=start, end=end, root=root
        )
        np.testing.assert_equal(signal, value)

    # cache result
    y = process.process_index(
        index,
        root=root,
        cache_root=cache_root,
    )
    os.remove(path)

    # fails because second file does not exist
    with pytest.raises(RuntimeError):
        process.process_index(
            index,
            root=root,
        )

    # loading from cache still works
    y_cached = process.process_index(
        index,
        root=root,
        cache_root=cache_root,
    )
    pd.testing.assert_series_equal(y, y_cached)


@pytest.mark.parametrize(
    'process_func, process_func_args, segment, signal, '
    'sampling_rate, file, start, end, keep_nat, expected_signal',
    [
        (
            None,
            {},
            None,
            np.array([1., 2., 3.]),
            44100,
            None,
            None,
            None,
            False,
            np.array([1., 2., 3.]),
        ),
        (
            None,
            {},
            None,
            np.array([1., 2., 3.]),
            44100,
            'file',
            None,
            None,
            False,
            np.array([1., 2., 3.]),
        ),
        (
            None,
            {},
            SEGMENT,
            np.array([1., 2., 3., 4.]),
            44100,
            None,
            None,
            None,
            False,
            np.array([1., 2.]),
        ),
        (
            None,
            {},
            SEGMENT,
            np.array([1., 2., 3., 4.]),
            44100,
            'file',
            None,
            None,
            False,
            np.array([1., 2.]),
        ),
        (
            signal_max,
            {},
            None,
            np.array([1., 2., 3.]),
            44100,
            None,
            None,
            None,
            False,
            3.0,
        ),
        (
            signal_duration,
            {},
            None,
            np.array([1., 2., 3.]),
            3,
            None,
            None,
            None,
            False,
            1.0,
        ),
        (
            signal_duration,
            {},
            None,
            np.array([1., 2., 3.]),
            1,
            None,
            pd.to_timedelta('2s'),
            None,
            False,
            1.0,
        ),
        (
            signal_duration,
            {},
            None,
            np.array([1., 2., 3.]),
            1,
            None,
            None,
            pd.to_timedelta('1s'),
            False,
            1.0,
        ),
        (
            signal_duration,
            {},
            None,
            np.array([1., 2., 3.]),
            1,
            None,
            None,
            pd.NaT,
            False,
            3.0,
        ),
        (
            signal_duration,
            {},
            None,
            np.array([1., 2., 3.]),
            1,
            None,
            None,
            pd.NaT,
            True,
            3.0,
        ),
        (
            signal_duration,
            {},
            None,
            np.array([1., 2., 3.]),
            1,
            None,
            pd.to_timedelta('1s'),
            pd.to_timedelta('2s'),
            False,
            1.0,
        ),
        (
            signal_duration,
            {},
            None,
            np.array([1., 2., 3.]),
            1,
            None,
            1,
            2,
            False,
            1.0,
        ),
        (
            signal_duration,
            {},
            None,
            np.array([1., 2., 3.]),
            1,
            None,
            1.0,
            2.0,
            False,
            1.0,
        ),
        (
            signal_duration,
            {},
            None,
            np.array([1., 2., 3.]),
            1,
            None,
            '1s',
            '2s',
            False,
            1.0,
        ),
        (
            signal_duration,
            {},
            None,
            np.array([1., 2., 3.]),
            1,
            'file',
            pd.to_timedelta('1s'),
            pd.to_timedelta('2s'),
            False,
            1.0,
        ),
        (
            signal_modification,
            {},
            None,
            np.array([1., 1., 1.]),
            44100,
            None,
            None,
            None,
            False,
            np.array([[1.1, 1.1, 1.1]]),
        ),
        (
            signal_modification,
            {'subtract': False},
            None,
            np.array([1., 1., 1.]),
            44100,
            None,
            None,
            None,
            False,
            np.array([[1.1, 1.1, 1.1]]),
        ),
        (
            signal_modification,
            {'subtract': True},
            None,
            np.array([1., 1., 1.]),
            44100,
            None,
            None,
            None,
            False,
            np.array([[0.9, 0.9, 0.9]]),
        ),
    ],
)
def test_process_signal(
        process_func,
        process_func_args,
        segment,
        signal,
        sampling_rate,
        file,
        start,
        end,
        keep_nat,
        expected_signal,
):
    process = audinterface.Process(
        process_func=process_func,
        sampling_rate=None,
        resample=False,
        segment=segment,
        keep_nat=keep_nat,
        verbose=False,
        process_func_args=process_func_args,
    )
    x = process.process_signal(
        signal,
        sampling_rate,
        file=file,
        start=start,
        end=end,
    )
    signal = np.atleast_2d(signal)
    if start is None or pd.isna(start):
        start = pd.to_timedelta(0)
    elif isinstance(start, (int, float)):
        start = pd.to_timedelta(start, 's')
    elif isinstance(start, str):
        start = pd.to_timedelta(start)
    if end is None or (pd.isna(end) and not keep_nat):
        end = pd.to_timedelta(
            signal.shape[1] / sampling_rate,
            unit='s',
        )
    elif isinstance(end, (int, float)):
        end = pd.to_timedelta(end, 's')
    elif isinstance(end, str):
        end = pd.to_timedelta(end)

    if segment is not None:
        index = segment.process_signal(
            signal,
            sampling_rate,
            start=start,
            end=end,
        )
        start = index[0][0]
        end = index[0][1]

    if file is None:
        y = pd.Series(
            [expected_signal],
            index=audinterface.utils.signal_index(start, end),
        )
    else:
        y = pd.Series(
            [expected_signal],
            index=audformat.segmented_index(file, start, end),
        )
    pd.testing.assert_series_equal(x, y)


@pytest.mark.parametrize(
    'num_workers, multiprocessing',
    [
        (1, False, ),
        (2, False, ),
        (None, False, ),
    ]
)
@pytest.mark.parametrize(
    'process_func, signal, sampling_rate, index',
    [
        (
            None,
            np.random.random(5 * 44100),
            44100,
            audinterface.utils.signal_index(),
        ),
        (
            None,
            np.random.random(5 * 44100),
            44100,
            audinterface.utils.signal_index(
                pd.timedelta_range('0s', '3s', 3),
                pd.timedelta_range('1s', '4s', 3)
            ),
        ),
        (
            signal_max,
            np.random.random(5 * 44100),
            44100,
            audinterface.utils.signal_index(
                pd.timedelta_range('0s', '3s', 3),
                pd.timedelta_range('1s', '4s', 3),
            ),
        ),
        (
            signal_max,
            np.random.random(5 * 44100),
            44100,
            audinterface.utils.signal_index(),
        ),
        (
            SignalObject(),
            np.random.random(5 * 44100),
            44100,
            audinterface.utils.signal_index(),
        ),
        pytest.param(
            signal_max,
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
        num_workers,
        multiprocessing,
        process_func,
        signal,
        sampling_rate,
        index,
):
    process = audinterface.Process(
        process_func=process_func,
        sampling_rate=None,
        resample=False,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
        verbose=False,
    )
    result = process.process_signal_from_index(signal, sampling_rate, index)
    expected = []
    for start, end in index:
        expected.append(
            process.process_signal(signal, sampling_rate, start=start, end=end)
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


@pytest.mark.parametrize(
    'process_func, signal, sampling_rate, min_signal_dur, '
    'max_signal_dur, expected',
    [
        (
            None,
            np.ones((1, 44100)),
            44100,
            None,
            None,
            np.ones((1, 44100)),
        ),
        (
            None,
            np.ones((1, 44100)),
            44100,
            2,
            None,
            np.concatenate(
                [
                    np.ones((1, 44100)),
                    np.zeros((1, 44100)),
                ],
                axis=1,
            ),
        ),
        (
            None,
            np.ones((1, 44100)),
            44100,
            None,
            0.01,
            np.ones((1, 441)),
        ),
    ]
)
def test_process_signal_min_max(
        process_func,
        signal,
        sampling_rate,
        min_signal_dur,
        max_signal_dur,
        expected,
):
    process = audinterface.Process(
        process_func=process_func,
        sampling_rate=None,
        resample=False,
        min_signal_dur=min_signal_dur,
        max_signal_dur=max_signal_dur,
        verbose=False,
    )
    result = process.process_signal(signal, sampling_rate)
    expected = pd.Series(
        [expected],
        index=audinterface.utils.signal_index(
            pd.to_timedelta(0),
            pd.to_timedelta(expected.shape[1] / sampling_rate, unit='s'),
        )
    )
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'process_func, signal, sampling_rate',
    [
        (
            lambda x, sr: x.mean(),
            np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32),
            1,
        ),
    ]
)
@pytest.mark.parametrize(
    'start, end, win_dur, hop_dur, expected',
    [
        (
            None, None, 4, None,
            pd.Series(
                [0, 0.5, 1],
                audinterface.utils.signal_index(
                    [0, 2, 4],
                    [4, 6, 8],
                ),
                dtype=np.float32,
            ),
        ),
        (
            None, None, 4, 2,
            pd.Series(
                [0, 0.5, 1],
                audinterface.utils.signal_index(
                    [0, 2, 4],
                    [4, 6, 8],
                ),
                dtype=np.float32,
            ),
        ),
        (
            None, None, 4, 3,
            pd.Series(
                [0, 0.75],
                audinterface.utils.signal_index(
                    [0, 3],
                    [4, 7],
                ),
                dtype=np.float32,
            ),
        ),
        (
            None, None, 4, 4,
            pd.Series(
                [0, 1],
                audinterface.utils.signal_index(
                    [0, 4],
                    [4, 8],
                ),
                dtype=np.float32,
            ),
        ),
        (
            None, None, 2, 4,
            pd.Series(
                [0, 1.0],
                audinterface.utils.signal_index(
                    [0, 4],
                    [2, 6],
                ),
                dtype=np.float32,
            ),
        ),
        (
            1.0, None, 4, 2,
            pd.Series(
                [0.25, 0.75],
                audinterface.utils.signal_index(
                    [1, 3],
                    [5, 7],
                ),
                dtype=np.float32,
            ),
        ),
        (
            1.0, 5.0, 4, 2,
            pd.Series(
                [0.25],
                audinterface.utils.signal_index(1, 5),
                dtype=np.float32,
            ),
        ),
        (
            1.0, 2.0, 4, 2,
            pd.Series(
                [],
                audinterface.utils.signal_index(),
                dtype=object,
            ),
        ),
        (
            9.0, 15.0, 4, 2,
            pd.Series(
                [],
                audinterface.utils.signal_index(),
                dtype=object,
            ),
        ),
        # missing win duration
        pytest.param(
            None, None, None, 2, None,
            marks=pytest.mark.xfail(raises=ValueError),
        )
    ]
)
def test_process_with_sliding_window(
        tmpdir,
        process_func,
        signal,
        sampling_rate,
        start,
        end,
        win_dur,
        hop_dur,
        expected,
):
    # save signal to file
    root = tmpdir
    file = 'file.wav'
    path = os.path.join(root, file)
    audiofile.write(path, signal, sampling_rate, bit_depth=32)

    # create interface
    process = audinterface.Process(
        process_func=process_func,
        hop_dur=hop_dur,
        win_dur=win_dur,
    )

    # process signal
    y = process.process_signal(
        signal,
        sampling_rate,
        start=start,
        end=end,
    )
    pd.testing.assert_series_equal(y, expected)

    # process signal from index
    y = process.process_signal_from_index(
        signal,
        sampling_rate,
        expected.index,
    )
    pd.testing.assert_series_equal(y, expected)

    # add file to expected index
    expected.index = audformat.segmented_index(
        [file] * len(expected.index),
        expected.index.get_level_values('start'),
        expected.index.get_level_values('end'),
    )

    # process signal with file
    y = process.process_signal(
        signal,
        sampling_rate,
        file=file,
        start=start,
        end=end,
    )
    pd.testing.assert_series_equal(y, expected)

    # process file
    y = process.process_file(file, start=start, end=end, root=root)
    pd.testing.assert_series_equal(y, expected)

    # process index
    y = process.process_index(expected.index, root=root)
    pd.testing.assert_series_equal(y, expected)


def test_process_with_special_args(tmpdir):

    duration = 3
    sampling_rate = 1
    signal = np.zeros((2, duration), np.float32)
    num_files = 10
    win_dur = 1
    num_frames = duration // win_dur
    num_workers = 3

    # create files
    root = tmpdir
    files = [f'f{idx}.wav' for idx in range(num_files)]
    index = audformat.segmented_index(
        np.repeat(files, num_frames),
        np.tile(range(num_frames), num_files),
        np.tile(range(1, num_frames + 1), num_files),
    )
    for file in files:
        path = os.path.join(root, file)
        audiofile.write(path, signal, sampling_rate, bit_depth=32)

    # create interface
    def process_func(signal, sampling_rate, idx, file, root):
        return (idx, file, root)

    process = audinterface.Process(
        process_func=process_func,
        num_workers=num_workers,
    )

    # process signal
    y = process.process_signal(signal, sampling_rate)
    expected = pd.Series(
        [(0, None, None)],
        audinterface.utils.signal_index(0, duration),
    )
    pd.testing.assert_series_equal(y, expected)

    # process signal from index
    y = process.process_signal_from_index(
        signal,
        sampling_rate,
        expected.index,
    )
    pd.testing.assert_series_equal(y, expected)

    # process file
    y = process.process_file(files[0], root=root)
    expected = pd.Series(
        [(0, files[0], root)],
        audformat.segmented_index(files[0], 0, duration),
    )
    pd.testing.assert_series_equal(y, expected)

    # process files
    y = process.process_files(files, root=root)
    expected = pd.Series(
        [(idx, files[idx], root) for idx in range(num_files)],
        audformat.segmented_index(
            files,
            [0] * num_files,
            [duration] * num_files,
        ),
    )
    pd.testing.assert_series_equal(y, expected)

    # process index with a filewise index
    y = process.process_index(
        audformat.filewise_index(files),
        root=root,
    )
    pd.testing.assert_series_equal(y, expected)

    # process index with a segmented index
    y = process.process_index(index, root=root)
    expected = pd.Series(
        [(idx, file, root) for idx, (file, _, _) in enumerate(index)],
        index,
    )
    pd.testing.assert_series_equal(y, expected)

    # sliding window
    # frames belonging to the same files have same idx
    process = audinterface.Process(
        process_func=process_func,
        win_dur=win_dur,
        hop_dur=win_dur,
        num_workers=num_workers,
    )
    y = process.process_files(files, root=root)
    values = []
    for idx in range(num_files):
        file = files[idx]
        for _ in range(num_frames):
            values.append((idx, file, root))
    expected = pd.Series(values, index)
    pd.testing.assert_series_equal(y, expected)

    # mono processing function
    # returns
    # [((0, files[0], root), (0, files[0], root)),
    #  ((1, files[1], root), (1, files[1], root)),
    #  ... ]
    process = audinterface.Process(
        process_func=process_func,
        process_func_is_mono=True,
        num_workers=num_workers,
    )
    y = process.process_index(index, root=root)
    expected = pd.Series(
        [((idx, file, root), (idx, file, root))
         for idx, (file, _, _) in enumerate(index)],
        index,
    )
    pd.testing.assert_series_equal(y, expected)

    # explicitely pass special arguments

    process = audinterface.Process(
        process_func=process_func,
        process_func_args={'idx': 99, 'file': 'my/file', 'root': None},
        num_workers=num_workers,
    )
    y = process.process_index(index, root=root)
    expected = pd.Series([(99, 'my/file', None)] * len(index), index)
    pd.testing.assert_series_equal(y, expected)


@pytest.mark.parametrize(
    'segment',
    [
        audinterface.Segment(
            process_func=lambda x, sr: audinterface.utils.signal_index()
        ),
        audinterface.Segment(
            process_func=lambda x, sr:
                audinterface.utils.signal_index(
                    pd.to_timedelta(0),
                    pd.to_timedelta(x.shape[1] / sr, unit='s') / 2,
                )
        ),
        audinterface.Segment(
            process_func=lambda x, sr:
            audinterface.utils.signal_index(
                pd.to_timedelta(x.shape[1] / sr, unit='s') / 2,
                pd.to_timedelta(x.shape[1] / sr, unit='s'),
            )
        ),
        audinterface.Segment(
            process_func=lambda x, sr:
                audinterface.utils.signal_index(
                    [
                        pd.to_timedelta(0),
                        pd.to_timedelta(x.shape[1] / sr, unit='s') / 2,
                    ],
                    [
                        pd.to_timedelta(x.shape[1] / sr, unit='s') / 2,
                        pd.to_timedelta(x.shape[1] / sr),
                    ],
                )
        )
    ]
)
def test_process_with_segment(tmpdir, segment):

    process = audinterface.Process()
    process_with_segment = audinterface.Process(
        segment=segment,
    )

    # create signal and file
    sampling_rate = 8000
    signal = np.zeros((1, sampling_rate))
    root = tmpdir
    file = 'file.wav'
    path = os.path.join(root, file)
    audiofile.write(path, signal, sampling_rate)

    # process signal
    index = segment.process_signal(
        signal,
        sampling_rate,
        file=file,
    )
    pd.testing.assert_series_equal(
        process.process_index(index, root=root),
        process_with_segment.process_signal(
            signal,
            sampling_rate,
            file=file,
        )
    )
    index = segment.process_signal_from_index(
        signal,
        sampling_rate,
        audformat.filewise_index(file),
    )
    pd.testing.assert_series_equal(
        process.process_index(index, root=root),
        process_with_segment.process_signal_from_index(
            signal,
            sampling_rate,
            audformat.filewise_index(file),
        )
    )

    # process file
    index = segment.process_file(file, root=root)
    pd.testing.assert_series_equal(
        process.process_index(index, root=root),
        process_with_segment.process_file(file, root=root)
    )
    index = segment.process_index(
        audformat.filewise_index(file),
        root=root,
    )
    pd.testing.assert_series_equal(
        process.process_index(index, root=root),
        process_with_segment.process_index(
            audformat.filewise_index(file),
            root=root,
        )
    )


def test_read_audio(tmpdir):
    sampling_rate = 8000
    signal = np.ones((1, 8000))
    path = str(tmpdir.mkdir('wav'))
    file = os.path.join(path, 'file.wav')
    af.write(file, signal, sampling_rate)
    s, sr = audinterface.utils.read_audio(
        file,
        start=pd.Timedelta('00:00:00.1'),
        end=pd.Timedelta('00:00:00.2'),
    )
    assert sr == sampling_rate
    assert s.shape[1] == 0.1 * sr


@pytest.mark.parametrize(
    'signal_sampling_rate, model_sampling_rate, resample',
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
    ],
)
def test_sampling_rate_mismatch(
        signal_sampling_rate,
        model_sampling_rate,
        resample,
):
    process = audinterface.Process(
        process_func=None,
        sampling_rate=model_sampling_rate,
        resample=resample,
        verbose=False,
    )
    signal = np.array([1., 2., 3.]).astype('float32')
    process.process_signal(signal, signal_sampling_rate)
