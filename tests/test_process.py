import os

import audiofile as af
import numpy as np
import pandas as pd
import pytest

import audinterface


def signal_duration(signal, sampling_rate):
    return signal.shape[1] / sampling_rate


def signal_max(signal, sampling_rate):
    return np.max(signal)


SEGMENT = audinterface.Segment(
    process_func=lambda x, sr:
        pd.MultiIndex.from_arrays(
            [
                [
                    pd.to_timedelta(0),
                ],
                [
                    pd.to_timedelta(x.shape[1] / sr, unit='sec'),
                ],
            ],
            names=['start', 'end'],
        )
)


def signal_modification(signal, sampling_rate, subtract=False):
    if subtract:
        signal -= 0.1 * signal
    else:
        signal += 0.1 * signal
    return signal


@pytest.mark.parametrize(
    'process_func, segment, signal, selected_channel, expected_output',
    [
        (
            signal_max,
            None,
            np.ones((1, 3)),
            None,
            1,
        ),
        (
            signal_max,
            SEGMENT,
            np.ones((1, 8000)),
            None,
            1,
        ),
        (
            signal_max,
            None,
            np.ones(3),
            0,
            1,
        ),
        (
            signal_max,
            None,
            np.array([[0., 0., 0.], [1., 1., 1.]]),
            0,
            0,
        ),
        (
            signal_max,
            None,
            np.array([[0., 0., 0.], [1., 1., 1.]]),
            1,
            1,
        ),
        pytest.param(
            signal_max,
            None,
            np.ones((1, 3)),
            1,
            1,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_process_file(
    tmpdir,
    process_func,
    segment,
    signal,
    selected_channel,
    expected_output,
):
    sampling_rate = 8000
    model = audinterface.Process(
        process_func=process_func,
        sampling_rate=sampling_rate,
        resample=False,
        segment=segment,
        verbose=False,
    )
    path = str(tmpdir.mkdir('wav'))
    filename = f'{path}/channel.wav'
    af.write(filename, signal, sampling_rate)
    output = model.process_file(filename, channel=selected_channel)
    np.testing.assert_almost_equal(
        output.values, expected_output, decimal=4,
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
    model = audinterface.Process(
        process_func=None,
        sampling_rate=None,
        resample=False,
        segment=segment,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
        verbose=False,
    )
    sampling_rate = 8000
    path = str(tmpdir.mkdir('wav'))
    files = [f'{path}/file{n}.wav' for n in range(num_files)]
    rel_path = os.path.relpath(path)
    rel_files = [f'{rel_path}/file{n}.wav' for n in range(num_files)]
    for file in files:
        signal = np.random.uniform(-1.0, 1.0, (1, sampling_rate))
        af.write(file, signal, sampling_rate)
    result = model.process_folder(path)
    rel_result = model.process_folder(rel_path)
    for idx, (index, values) in enumerate(result.iteritems()):
        file = index[0] if isinstance(index, tuple) else index
        assert file == files[idx]
        signal, sampling_rate = model.read_audio(file)
        np.testing.assert_equal(values, signal)
    for idx, (index, values) in enumerate(rel_result.iteritems()):
        file = index[0] if isinstance(index, tuple) else index
        assert file == rel_files[idx]
        signal, sampling_rate = model.read_audio(file)
        np.testing.assert_equal(values, signal)


@pytest.mark.parametrize(
    'process_func, process_func_kwargs, segment, signal, sampling_rate,'
    'file, start, end, expected_signal',
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
            np.array([1., 2., 3.]),
        ),
        (
            None,
            {},
            SEGMENT,
            np.array([1., 2., 3.]),
            44100,
            None,
            None,
            None,
            np.array([1., 2., 3.]),
        ),
        (
            None,
            {},
            SEGMENT,
            np.array([1., 2., 3.]),
            44100,
            'file',
            None,
            None,
            np.array([1., 2., 3.]),
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
            1.0,
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
            np.array([[0.9, 0.9, 0.9]]),
        ),
    ],
)
def test_process_signal(
        process_func,
        process_func_kwargs,
        segment,
        signal,
        sampling_rate,
        file,
        start,
        end,
        expected_signal,
):
    model = audinterface.Process(
        process_func=process_func,
        sampling_rate=None,
        resample=False,
        segment=segment,
        verbose=False,
        **process_func_kwargs,
    )
    x = model.process_signal(
        signal, sampling_rate, file=file, start=start, end=end,
    )
    signal = np.atleast_2d(signal)
    if start is None or pd.isna(start):
        start = pd.to_timedelta(0)
    if end is None or pd.isna(end):
        end = pd.to_timedelta(
            signal.shape[1] / sampling_rate, unit='sec',
        )
    if file is None:
        y = pd.Series(
            [expected_signal],
            index=pd.MultiIndex.from_arrays(
                [[start], [end]], names=['start', 'end']
            ),
        )
    else:
        y = pd.Series(
            [expected_signal],
            index=pd.MultiIndex.from_arrays(
                [[file], [start], [end]], names=['file', 'start', 'end']
            ),
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
            pd.MultiIndex.from_arrays(
                [
                    pd.to_timedelta([]),
                    pd.to_timedelta([]),
                ],
                names=['start', 'end']
            ),
        ),
        (
            None,
            np.random.random(5 * 44100),
            44100,
            pd.MultiIndex.from_arrays(
                [
                    pd.timedelta_range('0s', '3s', 3),
                    pd.timedelta_range('1s', '4s', 3),
                ],
                names=['start', 'end']
            ),
        ),
        (
            signal_max,
            np.random.random(5 * 44100),
            44100,
            pd.MultiIndex.from_arrays(
                [
                    pd.timedelta_range('0s', '3s', 3),
                    pd.timedelta_range('1s', '4s', 3),
                ],
                names=['start', 'end']
            ),
        ),
        (
            signal_max,
            np.random.random(5 * 44100),
            44100,
            pd.MultiIndex.from_arrays(
                [
                    pd.to_timedelta([]),
                    pd.to_timedelta([]),
                ],
                names=['start', 'end']
            ),
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
    model = audinterface.Process(
        process_func=process_func,
        sampling_rate=None,
        resample=False,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
        verbose=False,
    )
    result = model.process_signal_from_index(signal, sampling_rate, index)
    expected = []
    for start, end in index:
        expected.append(
            model.process_signal(signal, sampling_rate, start=start, end=end)
        )
    if not expected:
        pd.testing.assert_series_equal(
            result,
            pd.Series([], index),
        )
    else:
        pd.testing.assert_series_equal(
            result,
            pd.concat(expected, names=['start', 'end']),
        )


def test_read_audio(tmpdir):
    sampling_rate = 8000
    signal = np.ones((1, 8000))
    path = str(tmpdir.mkdir('wav'))
    file = os.path.join(path, 'file.wav')
    af.write(file, signal, sampling_rate)
    s, sr = audinterface.Process().read_audio(
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
    model = audinterface.Process(
        process_func=None,
        sampling_rate=model_sampling_rate,
        resample=resample,
        verbose=False,
    )
    signal = np.array([1., 2., 3.])
    model.process_signal(signal, signal_sampling_rate)


@pytest.mark.parametrize(
    'num_workers, multiprocessing',
    [
        (1, False, ),
        (2, False, ),
        (None, False, ),
    ]
)
def test_unified_format_index(tmpdir, num_workers, multiprocessing):

    model = audinterface.Process(
        process_func=None,
        sampling_rate=None,
        resample=False,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
        verbose=False,
    )
    sampling_rate = 8000
    signal = np.random.uniform(-1.0, 1.0, (1, 3 * sampling_rate))
    path = str(tmpdir.mkdir('wav'))
    file = f'{path}/file.wav'
    af.write(file, signal, sampling_rate)

    # empty index
    index = pd.MultiIndex.from_arrays(
        [
            [],
            pd.to_timedelta([]),
            pd.to_timedelta([]),
        ],
        names=('file', 'start', 'end')
    )
    result = model.process_unified_format_index(index)
    assert result.empty

    # valid index
    index = pd.MultiIndex.from_arrays(
        [
            [file] * 3,
            pd.timedelta_range('0s', '2s', 3),
            pd.timedelta_range('1s', '3s', 3),
        ],
        names=('file', 'start', 'end')
    )
    result = model.process_unified_format_index(index)
    for (file, start, end), value in result.items():
        signal, sampling_rate = model.read_audio(
            file, start=start, end=end
        )
        np.testing.assert_equal(signal, value)

    # bad index
    index = pd.MultiIndex.from_arrays(
        [
            [file] * 3,
            pd.to_timedelta([1, 2, 3], unit='sec'),
            pd.to_timedelta([2, 3, 4], unit='sec'),
        ],
        names=['no', 'unified', 'format'],
    )
    with pytest.raises(ValueError):
        model.process_unified_format_index(index)
    index = pd.MultiIndex.from_arrays(
        [
            [file] * 3,
            [1, 2, 3],
            pd.to_timedelta([2, 3, 4], unit='sec'),
        ],
        names=['file', 'start', 'end'],
    )
    with pytest.raises(ValueError):
        model.process_unified_format_index(index)
    index = pd.MultiIndex.from_arrays(
        [
            [file] * 3,
            pd.to_timedelta([1, 2, 3], unit='sec'),
            [2, 3, 4],
        ],
        names=['file', 'start', 'end'],
    )
    with pytest.raises(ValueError):
        model.process_unified_format_index(index)
