import audiofile as af
import numpy as np
import pandas as pd
import pytest

import audinterface


SAMPLING_RATE = 8000
SIGNAL = np.ones((10, SAMPLING_RATE))
STARTS = pd.timedelta_range('0s', '10s', 3)
ENDS = STARTS + pd.to_timedelta('1s')
INDEX = pd.MultiIndex.from_arrays(
    [STARTS, ENDS],
    names=['start', 'end']
)


@pytest.mark.parametrize(
    'signal,sampling_rate,segment_func,result',
    [
        (
            SIGNAL,
            SAMPLING_RATE,
            None,
            pd.MultiIndex.from_arrays(
                [
                    pd.to_timedelta([]),
                    pd.to_timedelta([])
                ],
                names=['start', 'end']
            ),
        ),
        (
            SIGNAL,
            SAMPLING_RATE,
            lambda x, sr: INDEX,
            pd.MultiIndex.from_arrays(
                [
                    STARTS,
                    ENDS
                ],
                names=['start', 'end'],
            )
        ),
    ]
)
def test_call(signal, sampling_rate, segment_func, result):
    model = audinterface.Segment(
        process_func=segment_func,
    )
    index = model(signal, sampling_rate)
    pd.testing.assert_index_equal(index, result)


def test_file(tmpdir):
    model = audinterface.Segment(
        process_func=lambda s, sr: INDEX,
        sampling_rate=None,
        resample=False,
        verbose=False,
    )
    path = str(tmpdir.mkdir('wav'))
    file = f'{path}/file.wav'
    af.write(file, SIGNAL, SAMPLING_RATE)
    result = model.process_file(file)
    assert all(result.levels[0] == file)
    assert all(result.levels[1] == INDEX.levels[0])
    assert all(result.levels[2] == INDEX.levels[1])
    result = model.process_file(file, start=pd.to_timedelta('1s'))
    assert all(result.levels[0] == file)
    assert all(result.levels[1] == INDEX.levels[0] + pd.to_timedelta('1s'))
    assert all(result.levels[2] == INDEX.levels[1] + pd.to_timedelta('1s'))


@pytest.mark.parametrize(
    'num_workers, multiprocessing',
    [
        (1, False, ),
        (2, False, ),
        (None, False, ),
    ]
)
def test_folder(tmpdir, num_workers, multiprocessing):
    model = audinterface.Segment(
        process_func=lambda s, sr: INDEX,
        sampling_rate=None,
        resample=False,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
        verbose=False,
    )
    path = str(tmpdir.mkdir('wav'))
    files = [f'{path}/file{n}.wav' for n in range(3)]
    for file in files:
        af.write(file, SIGNAL, SAMPLING_RATE)
    result = model.process_folder(path)
    assert all(result.levels[0] == files)
    assert all(result.levels[1] == INDEX.levels[0])
    assert all(result.levels[2] == INDEX.levels[1])


@pytest.mark.parametrize(
    'num_workers, multiprocessing',
    [
        (1, False, ),
        (2, False, ),
        (None, False, ),
    ]
)
def test_index(tmpdir, num_workers, multiprocessing):

    def process_func(x, sr):
        dur = pd.to_timedelta(x.shape[-1] / sr, unit='s')
        return pd.MultiIndex.from_arrays(
            [
                [pd.to_timedelta('0.1s')],
                [dur - pd.to_timedelta('0.1s')],
            ],
            names=['start', 'end'],
        )

    segment = audinterface.Segment(
        process_func=process_func,
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
    result = segment.process_index(index)
    assert result.empty

    # segmented index
    index = pd.MultiIndex.from_arrays(
        [
            [file] * 3,
            pd.timedelta_range('0s', '2s', 3),
            pd.timedelta_range('1s', '3s', 3),
        ],
        names=('file', 'start', 'end')
    )
    result = segment.process_index(index)
    expected = pd.MultiIndex.from_arrays(
        [
            [file] * 3,
            index.get_level_values('start') + pd.to_timedelta('0.1s'),
            index.get_level_values('end') - pd.to_timedelta('0.1s'),
        ],
        names=('file', 'start', 'end')
    )
    pd.testing.assert_index_equal(result, expected)

    # filewise index
    index = pd.Index([file], name='file')
    result = segment.process_index(index)
    expected = pd.MultiIndex.from_arrays(
        [
            [file],
            [pd.to_timedelta('0.1s')],
            [pd.to_timedelta('2.9s')],
        ],
        names=('file', 'start', 'end')
    )
    pd.testing.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    'signal,sampling_rate,segment_func,start,end,result',
    [
        (
            SIGNAL,
            SAMPLING_RATE,
            None,
            None,
            None,
            pd.MultiIndex.from_arrays(
                [
                    pd.to_timedelta([]),
                    pd.to_timedelta([])
                ],
                names=['start', 'end']
            ),
        ),
        (
            SIGNAL,
            SAMPLING_RATE,
            lambda x, sr: INDEX,
            None,
            None,
            pd.MultiIndex.from_arrays(
                [
                    STARTS,
                    ENDS
                ],
                names=['start', 'end'],
            )
        ),
        (
            SIGNAL,
            SAMPLING_RATE,
            lambda x, sr: INDEX,
            pd.to_timedelta('1s'),
            pd.to_timedelta('10s'),
            pd.MultiIndex.from_arrays(
                [
                    STARTS + pd.to_timedelta('1s'),
                    ENDS + pd.to_timedelta('1s')
                ],
                names=['start', 'end'],
            )
        ),
        pytest.param(
            SIGNAL,
            SAMPLING_RATE,
            lambda x, sr: pd.MultiIndex.from_arrays(
                [STARTS],
            ),
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            SIGNAL,
            SAMPLING_RATE,
            lambda x, sr: pd.MultiIndex.from_arrays(
                [['wrong', 'data', 'type'], ENDS],
            ),
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            SIGNAL,
            SAMPLING_RATE,
            lambda x, sr: pd.MultiIndex.from_arrays(
                [STARTS, ['wrong', 'data', 'type']],
            ),
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_signal(signal, sampling_rate, segment_func, start, end, result):
    model = audinterface.Segment(
        process_func=segment_func,
    )
    index = model.process_signal(signal, sampling_rate, start=start, end=end)
    pd.testing.assert_index_equal(index, result)


def test_signal_kwargs():
    def segment_func(s, sr, arg1, arg2):
        assert arg1 == 'foo'
        assert arg2 == 'bar'
    audinterface.Segment(
        process_func=segment_func,
        arg1='foo',
        arg2='bar',
    )
