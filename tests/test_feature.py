import os

import audiofile as af
import numpy as np
import pandas as pd
import pytest

import audinterface


SAMPLING_RATE = 8000
NUM_CHANNELS = 2
NUM_FEATURES = 3
SIGNAL = np.ones((NUM_CHANNELS, SAMPLING_RATE))
STARTS = [pd.to_timedelta('0s')] * 3
ENDS = [pd.to_timedelta('1s')] * 3
INDEX = pd.MultiIndex.from_arrays(
    [STARTS, ENDS],
    names=['start', 'end']
)
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


def feature_extrator(signal, _):
    return np.ones((NUM_CHANNELS, NUM_FEATURES))


def features_extractor_sliding_window(signal, _, hop_size):
    num_time_steps = int(np.ceil(signal.shape[1] / hop_size))
    return np.ones((NUM_CHANNELS, NUM_FEATURES, num_time_steps))


def test_feature():
    # You have to specify sampling rate with unit == 'samples'
    with pytest.raises(ValueError):
        audinterface.Feature(
            feature_names=('o1', 'o2', 'o3'),
            sampling_rate=None,
            unit='samples',
        )
    # Only hop_dur is given
    with pytest.raises(ValueError):
        audinterface.Feature(
            feature_names=('o1', 'o2', 'o3'),
            hop_dur=0.1,
        )
    audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        win_dur=2048,
        unit='samples',
        sampling_rate=8000,
    )


@pytest.mark.parametrize(
    'start, end, segment',
    [
        (None, None, None),
        (None, None, SEGMENT),
        (pd.NaT, pd.NaT, None),
        (pd.to_timedelta('0.25s'), None, None),
        (pd.to_timedelta('0.25s'), pd.NaT, None),
        (None, pd.to_timedelta('0.75s'), None),
        (pd.NaT, pd.to_timedelta('0.75s'), None),
        (pd.to_timedelta('0.25s'), pd.to_timedelta('0.75s'), None),
    ]
)
def test_process_file(tmpdir, start, end, segment):
    extractor = audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=feature_extrator,
        sampling_rate=None,
        num_channels=NUM_CHANNELS,
        resample=False,
        segment=segment,
        verbose=False,
    )
    expected_features = np.ones((1, NUM_CHANNELS * NUM_FEATURES))
    path = str(tmpdir.mkdir('wav'))
    file = os.path.join(path, 'file.wav')
    af.write(file, SIGNAL, SAMPLING_RATE)
    features = extractor.process_file(file, start=start, end=end)
    if start is None or pd.isna(start):
        start = pd.to_timedelta(0)
    if end is None or pd.isna(end):
        end = pd.to_timedelta(af.duration(file), unit='sec')
    assert features.index.levels[0][0] == file
    assert features.index.levels[1][0] == start
    assert features.index.levels[2][0] == end
    np.testing.assert_array_equal(features, expected_features)


def test_process_folder(tmpdir):
    extractor = audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=feature_extrator,
        sampling_rate=None,
        num_channels=NUM_CHANNELS,
        resample=False,
        verbose=False,
    )
    expected_features = np.ones((3, NUM_CHANNELS * NUM_FEATURES))
    path = str(tmpdir.mkdir('wav'))
    files = [f'{path}/file{n}.wav' for n in range(3)]
    for file in files:
        af.write(file, SIGNAL, SAMPLING_RATE)
    features = extractor.process_folder(path)
    assert all(features.index.levels[0] == files)
    assert all(features.index.levels[1] == INDEX.levels[0])
    assert all(features.index.levels[2] == INDEX.levels[1])
    np.testing.assert_array_equal(features.values, expected_features)


@pytest.mark.parametrize(
    'process_func,start,end,expected_features',
    [
        (
            None,
            None,
            None,
            np.zeros((1, NUM_CHANNELS * NUM_FEATURES), dtype=np.float),
        ),
        (
            feature_extrator,
            None,
            None,
            np.ones((1, NUM_CHANNELS * NUM_FEATURES), dtype=np.float),
        ),
        (
            feature_extrator,
            pd.to_timedelta('1s'),
            pd.to_timedelta('10s'),
            np.ones((1, NUM_CHANNELS * NUM_FEATURES), dtype=np.float),
        ),
        # Feature extractor function returns too many dimensions
        pytest.param(
            lambda s, sr: np.ones((1, 1, 1, 1)),
            None,
            None,
            np.ones((1, NUM_CHANNELS * NUM_FEATURES), dtype=np.float),
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # Feature extractor function returns wrong number of channels
        pytest.param(
            lambda s, sr: np.ones((NUM_CHANNELS + 1, NUM_FEATURES)),
            None,
            None,
            np.ones((1, NUM_CHANNELS * NUM_FEATURES), dtype=np.float),
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # Feature extractor function returns wrong number of features
        pytest.param(
            lambda s, sr: np.ones((NUM_CHANNELS, NUM_FEATURES + 1)),
            None,
            None,
            np.ones((1, NUM_CHANNELS * NUM_FEATURES), dtype=np.float),
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # Feature extrator function returns more than one time step
        pytest.param(
            lambda s, sr: np.ones((NUM_CHANNELS, NUM_FEATURES, 2)),
            None,
            None,
            np.ones((1, NUM_CHANNELS * NUM_FEATURES), dtype=np.float),
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
    ]
)
def test_process_signal(process_func, start, end, expected_features):
    extractor = audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=process_func,
        num_channels=NUM_CHANNELS,
    )
    features = extractor.process_signal(
        SIGNAL,
        SAMPLING_RATE,
        start=start,
        end=end,
    )
    np.testing.assert_array_equal(features.values, expected_features)


@pytest.mark.parametrize(
    'index,expected_features',
    [
        (
            pd.MultiIndex.from_arrays(
                [
                    (pd.to_timedelta('0s'), pd.to_timedelta('1s')),
                    (pd.to_timedelta('2s'), pd.to_timedelta('3s')),
                ],
                names=['start', 'end'],
            ),
            np.ones((2, NUM_CHANNELS * NUM_FEATURES)),
        ),
    ],
)
def test_process_signal_from_index(index, expected_features):
    extractor = audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=feature_extrator,
        num_channels=NUM_CHANNELS,
    )
    features = extractor.process_signal_from_index(
        SIGNAL,
        SAMPLING_RATE,
        index,
    )
    np.testing.assert_array_equal(features.values, expected_features)


def test_process_unified_format_index(tmpdir):
    path = str(tmpdir.mkdir('wav'))
    file = f'{path}/file.wav'
    af.write(file, SIGNAL, SAMPLING_RATE)
    index = pd.MultiIndex.from_arrays(
        [
            (file, ) * 2,
            (pd.to_timedelta('0s'), pd.to_timedelta('1s')),
            (pd.to_timedelta('2s'), pd.to_timedelta('3s')),
        ],
        names=['file', 'start', 'end'],
    )
    expected_features = np.ones((2, NUM_CHANNELS * NUM_FEATURES))
    extractor = audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=feature_extrator,
        num_channels=NUM_CHANNELS,
    )
    features = extractor.process_unified_format_index(
        index,
    )
    np.testing.assert_array_equal(features.values, expected_features)


def test_signal_sliding_window():
    # Test sliding window with two time steps
    expected_features = np.ones((NUM_CHANNELS, 2 * NUM_FEATURES))
    extractor = audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=features_extractor_sliding_window,
        num_channels=NUM_CHANNELS,
        win_dur=1,
        hop_dur=0.5,
        unit='seconds',
        hop_size=SAMPLING_RATE // 2,  # argument to process_func
    )
    features = extractor.process_signal(
        SIGNAL,
        SAMPLING_RATE,
    )
    np.testing.assert_array_equal(features.values, expected_features)


def test_signal_kwargs():
    def process_func(s, sr, arg1, arg2):
        assert arg1 == 'foo'
        assert arg2 == 'bar'
    audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=process_func,
        arg1='foo',
        arg2='bar',
    )


def test_to_numpy():
    expected_features = np.ones((NUM_CHANNELS, NUM_FEATURES, 1))
    extractor = audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=feature_extrator,
        num_channels=NUM_CHANNELS,
    )
    features = extractor.process_signal(
        SIGNAL,
        SAMPLING_RATE,
    )
    features = extractor.to_numpy(features)
    np.testing.assert_array_equal(features, expected_features)
