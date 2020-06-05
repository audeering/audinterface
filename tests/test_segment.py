import audiofile as af
import numpy as np
import pandas as pd
import pytest

import audinterface


sampling_rate = 8000
signal = np.ones((10, sampling_rate))
starts = pd.timedelta_range('0s', '10s', 3)
ends = starts + pd.to_timedelta('1s')
index = pd.MultiIndex.from_arrays(
    [starts, ends],
    names=['start', 'end']
)


def test_file(tmpdir):
    model = audinterface.Segment(
        segment_func=lambda s, sr: index,
        sampling_rate=None,
        resample=False,
        verbose=False,
    )
    path = str(tmpdir.mkdir('wav'))
    file = f'{path}/file.wav'
    af.write(file, signal, sampling_rate)
    result = model.segment_file(file)
    assert all(result.levels[0] == file)
    assert all(result.levels[1] == index.levels[0])
    assert all(result.levels[2] == index.levels[1])
    result = model.segment_file(file, start=pd.to_timedelta('1s'))
    assert all(result.levels[0] == file)
    assert all(result.levels[1] == index.levels[0] + pd.to_timedelta('1s'))
    assert all(result.levels[2] == index.levels[1] + pd.to_timedelta('1s'))


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
        segment_func=lambda s, sr: index,
        sampling_rate=None,
        resample=False,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
        verbose=False,
    )
    path = str(tmpdir.mkdir('wav'))
    files = [f'{path}/file{n}.wav' for n in range(3)]
    for file in files:
        af.write(file, signal, sampling_rate)
    result = model.segment_folder(path)
    assert all(result.levels[0] == files)
    assert all(result.levels[1] == index.levels[0])
    assert all(result.levels[2] == index.levels[1])


@pytest.mark.parametrize(
    'signal,segment_func,start,end,result',
    [
        (
            signal,
            None,
            None,
            None,
            pd.MultiIndex.from_arrays(
                [pd.to_timedelta([]), pd.to_timedelta([])],
                names=['start', 'end']
            ),
        ),
        (
            signal,
            lambda x, sr: index,
            None,
            None,
            pd.MultiIndex.from_arrays(
                [starts, ends],
                names=['start', 'end'],
            )
        ),
        (
            signal,
            lambda x, sr: index,
            pd.to_timedelta('1s'),
            pd.to_timedelta('10s'),
            pd.MultiIndex.from_arrays(
                [
                    starts + pd.to_timedelta('1s'),
                    ends + pd.to_timedelta('1s')
                ],
                names=['start', 'end'],
            )
        ),
        pytest.param(
            signal,
            lambda x, sr: pd.MultiIndex.from_arrays(
                [starts],
            ),
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            signal,
            lambda x, sr: pd.MultiIndex.from_arrays(
                [['wrong', 'data', 'type'], ends],
            ),
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            signal,
            lambda x, sr: pd.MultiIndex.from_arrays(
                [starts, ['wrong', 'data', 'type']],
            ),
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_signal(signal, segment_func, start, end, result):
    model = audinterface.Segment(
        segment_func=segment_func,
    )
    index = model.segment_signal(signal, sampling_rate, start=start, end=end)
    pd.testing.assert_index_equal(index, result)


def test_signal_kwargs():
    def segment_func(s, sr, arg1, arg2):
        assert arg1 == 'foo'
        assert arg2 == 'bar'
    audinterface.Segment(
        segment_func=segment_func,
        arg1='foo',
        arg2='bar',
    )
