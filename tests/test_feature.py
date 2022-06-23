import os

import numpy as np
import pandas as pd
import pytest

import audeer
import audformat
import audinterface
import audiofile as af


SAMPLING_RATE = 8000
NUM_CHANNELS = 2
NUM_FEATURES = 3
NUM_FRAMES = 5
SIGNAL_1D = np.ones((1, SAMPLING_RATE))
SIGNAL_2D = np.ones((NUM_CHANNELS, SAMPLING_RATE))
SEGMENT = audinterface.Segment(
    process_func=lambda x, sr:
        audinterface.utils.signal_index(
            pd.to_timedelta(0),
            pd.to_timedelta(x.shape[1] / sr, unit='s') / 2,
        )
)


def feature_extractor(signal, _):
    return np.ones((NUM_CHANNELS, NUM_FEATURES))


def features_extractor_sliding_window(signal, _, hop_size):
    num_time_steps = int(np.ceil(signal.shape[1] / hop_size))
    return np.ones((NUM_CHANNELS, NUM_FEATURES, num_time_steps))


def test_deprecated_unit_argument():
    if (
            audeer.LooseVersion(audinterface.__version__)
            < audeer.LooseVersion('1.2.0')
    ):
        with pytest.warns(UserWarning, match='is deprecated'):
            interface = audinterface.Feature(
                ['a'],
                win_dur=1000,
                unit='samples',
                sampling_rate=16000,
            )
            assert interface.win_dur == '1000'
            interface = audinterface.Feature(
                ['a'],
                win_dur=1000,
                hop_dur=500,
                unit='milliseconds',
            )
            assert interface.win_dur == '1000milliseconds'
            assert interface.hop_dur == '500milliseconds'
    else:
        with pytest.raises(TypeError, match='unexpected keyword argument'):
            audinterface.Feature(['a'], unit='samples')


def test_feature():
    # You have to specify sampling rate when win_dur is in samples
    with pytest.raises(ValueError):
        audinterface.Feature(
            feature_names=('o1', 'o2', 'o3'),
            sampling_rate=None,
            win_dur='2048',
        )
    # If no win_dur is given, no error should occur
    audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        sampling_rate=None,
    )
    # Only hop_dur is given
    with pytest.raises(ValueError):
        audinterface.Feature(
            feature_names=('o1', 'o2', 'o3'),
            hop_dur=0.1,
        )
    audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        win_dur='2048',
        sampling_rate=8000,
    )


@pytest.mark.parametrize(
    'signal, feature, expected',
    [
        (
            SIGNAL_1D,
            audinterface.Feature(
                feature_names='feature',
                process_func=lambda x, sr: 1,
            ),
            np.ones((1, 1, 1)),
        ),
        (
            SIGNAL_1D,
            audinterface.Feature(
                feature_names=['feature'],
                process_func=lambda x, sr: 1,
            ),
            np.ones((1, 1, 1)),
        ),
        (
            SIGNAL_1D,
            audinterface.Feature(
                feature_names=['f1', 'f2', 'f3'],
                process_func=lambda x, sr: np.ones(3),
            ),
            np.ones((1, 3, 1)),
        ),
        (
            SIGNAL_1D,
            audinterface.Feature(
                feature_names=['f1', 'f2', 'f3'],
                process_func=lambda x, sr: np.ones((1, 3)),
            ),
            np.ones((1, 3, 1)),
        ),
        (
            SIGNAL_1D,
            audinterface.Feature(
                feature_names=['f1', 'f2', 'f3'],
                process_func=lambda x, sr: np.ones((3, 1)),
            ),
            np.ones((1, 3, 1)),
        ),
        (
            SIGNAL_1D,
            audinterface.Feature(
                feature_names=['f1', 'f2', 'f3'],
                process_func=lambda x, sr: np.ones((1, 3, 5)),
            ),
            np.ones((1, 3, 5)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=['f1', 'f2', 'f3'],
                process_func=lambda x, sr: np.ones((1, 3, 5)),
                channels=1,
            ),
            np.ones((1, 3, 5)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=['f1', 'f2', 'f3'],
                process_func=lambda x, sr: np.ones((2, 3)),
                channels=range(2),
            ),
            np.ones((2, 3, 1)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=['f1', 'f2', 'f3'],
                process_func=lambda x, sr: np.ones((2, 3, 5)),
                channels=range(2),
            ),
            np.ones((2, 3, 5)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=['f1', 'f2', 'f3'],
                process_func=lambda x, sr: np.ones(3),
                channels=range(2),
                process_func_is_mono=True,
            ),
            np.ones((2, 3, 1)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=['f1', 'f2', 'f3'],
                process_func=lambda x, sr: np.ones((1, 3)),
                channels=range(2),
                process_func_is_mono=True,
            ),
            np.ones((2, 3, 1)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=['f1', 'f2', 'f3'],
                process_func=lambda x, sr: np.ones((1, 3, 1)),
                channels=range(2),
                process_func_is_mono=True,
            ),
            np.ones((2, 3, 1)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=['f1', 'f2', 'f3'],
                process_func=lambda x, sr: np.ones((3, 5)),
                channels=range(2),
                process_func_is_mono=True,
            ),
            np.ones((2, 3, 5)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=['f1', 'f2', 'f3'],
                process_func=lambda x, sr: np.ones((1, 3, 5)),
                channels=range(2),
                process_func_is_mono=True,
            ),
            np.ones((2, 3, 5)),
        ),
    ]
)
def test_process_call(signal, feature, expected):
    np.testing.assert_array_equal(
        feature(signal, SAMPLING_RATE),
        expected,
    )


@pytest.mark.parametrize(
    'signal, extractor',
    [
        (
            np.random.randn(1, SAMPLING_RATE),
            audinterface.Feature(
                feature_names='mean',
                process_func=lambda x, sr: np.mean(x, axis=1),
            ),
        ),
        (
            np.random.randn(2, SAMPLING_RATE),
            audinterface.Feature(
                feature_names='mean',
                process_func=lambda x, sr: np.mean(x, axis=1),
            ),
        ),
        (
            np.random.randn(2, SAMPLING_RATE),
            audinterface.Feature(
                feature_names='mean',
                process_func=lambda x, sr: np.mean(x, axis=1),
                channels=0,
            ),
        ),
    ]
)
def test_process_call_datatype(signal, extractor):
    features = extractor(signal, SAMPLING_RATE)
    assert signal.dtype == features.dtype


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

    start_org = start
    end_org = end

    feature = audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=feature_extractor,
        sampling_rate=None,
        channels=range(NUM_CHANNELS),
        resample=False,
        segment=segment,
        verbose=False,
    )
    y_expected = np.ones((1, NUM_CHANNELS * NUM_FEATURES))

    # create test file
    root = str(tmpdir.mkdir('wav'))
    file = 'file.wav'
    path = os.path.join(root, file)
    af.write(path, SIGNAL_2D, SAMPLING_RATE)

    # test absolute path
    start = start_org
    end = end_org

    y = feature.process_file(path, start=start, end=end)
    if start is None or pd.isna(start):
        start = pd.to_timedelta(0)
    if end is None or pd.isna(end):
        end = pd.to_timedelta(af.duration(path), unit='s')

    if segment is not None:
        index = segment.process_file(path)
        start = index[0][1]
        end = index[0][2]

    assert y.index.levels[0][0] == path
    assert y.index.levels[1][0] == start
    assert y.index.levels[2][0] == end
    np.testing.assert_array_equal(y, y_expected)

    # test relative path
    start = start_org
    end = end_org

    y = feature.process_file(file, start=start, end=end, root=root)
    if start is None or pd.isna(start):
        start = pd.to_timedelta(0)
    if end is None or pd.isna(end):
        end = pd.to_timedelta(af.duration(path), unit='s')

    if segment is not None:
        index = segment.process_file(file, root=root)
        start = index[0][1]
        end = index[0][2]

    assert y.index.levels[0][0] == file
    assert y.index.levels[1][0] == start
    assert y.index.levels[2][0] == end
    np.testing.assert_array_equal(y, y_expected)


def test_process_folder(tmpdir):

    index = audinterface.utils.signal_index(0, 1)
    feature_names = ['o1', 'o2', 'o3']
    feature = audinterface.Feature(
        feature_names,
        process_func=feature_extractor,
        sampling_rate=None,
        channels=range(NUM_CHANNELS),
        resample=False,
        verbose=False,
    )

    path = str(tmpdir.mkdir('wav'))
    files = [
        os.path.join(path, f'file{n}.wav') for n in range(3)
    ]
    for file in files:
        af.write(file, SIGNAL_2D, SAMPLING_RATE)

    y = feature.process_folder(path)
    y_expected = np.ones((3, NUM_CHANNELS * NUM_FEATURES))

    assert all(y.index.levels[0] == files)
    assert all(y.index.levels[1] == index.levels[0])
    assert all(y.index.levels[2] == index.levels[1])
    np.testing.assert_array_equal(y.values, y_expected)

    # non-existing folder
    with pytest.raises(FileNotFoundError):
        feature.process_folder('bad-folder')

    # empty folder
    root = str(tmpdir.mkdir('empty'))
    df = feature.process_folder(root)
    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame(
            dtype=object,
            columns=feature.column_names,
        ),
    )


def test_process_func_args():
    def process_func(s, sr, arg1, arg2):
        assert arg1 == 'foo'
        assert arg2 == 'bar'
    audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=process_func,
        process_func_args={
            'arg1': 'foo',
            'arg2': 'bar',
        }
    )
    with pytest.warns(UserWarning):
        audinterface.Feature(
            feature_names=('o1', 'o2', 'o3'),
            process_func=process_func,
            arg1='foo',
            arg2='bar',
        )


@pytest.mark.parametrize(
    'process_func, num_feat, signal, start, end, expand, expected',
    [
        # no process function
        (
            None,
            3,
            SIGNAL_2D,
            None,
            None,
            False,
            np.zeros((1, 2 * 3)),
        ),
        # 1 channel, 1 feature
        (
            lambda s, sr: 1,
            1,
            SIGNAL_1D,
            None,
            None,
            False,
            np.ones((1, 1)),
        ),
        (
            lambda s, sr: np.ones(1),
            1,
            SIGNAL_1D,
            None,
            None,
            False,
            np.ones((1, 1)),
        ),
        # 1 channel, 1 feature
        (
            lambda s, sr: [1],
            1,
            SIGNAL_1D,
            None,
            None,
            False,
            np.ones((1, 1)),
        ),
        (
            lambda s, sr: np.ones((1, 1)),
            1,
            SIGNAL_1D,
            None,
            None,
            False,
            np.ones((1, 1)),
        ),
        # 1 channel, 3 features
        (
            lambda s, sr: [1, 1, 1],
            3,
            SIGNAL_1D,
            None,
            None,
            False,
            np.ones((1, 3)),
        ),
        (
            lambda s, sr: np.ones(3),
            3,
            SIGNAL_1D,
            None,
            None,
            False,
            np.ones((1, 3)),
        ),
        # 1 channel, 3 features
        (
            lambda s, sr: [[1, 1, 1]],
            3,
            SIGNAL_1D,
            None,
            None,
            False,
            np.ones((1, 3)),
        ),
        (
            lambda s, sr: np.ones((1, 3)),
            3,
            SIGNAL_1D,
            None,
            None,
            False,
            np.ones((1, 3)),
        ),
        # 2 channels, 1 feature
        (
            lambda s, sr: [[1], [1]],
            1,
            SIGNAL_2D,
            None,
            None,
            False,
            np.ones((1, 2)),
        ),
        (
            lambda s, sr: np.ones((2, 1)),
            1,
            SIGNAL_2D,
            None,
            None,
            False,
            np.ones((1, 2)),
        ),
        # 2 channels, 3 features
        (
            lambda s, sr: [[1, 1, 1], [1, 1, 1]],
            3,
            SIGNAL_2D,
            None,
            None,
            False,
            np.ones((1, 2 * 3)),
        ),
        (
            lambda s, sr: np.ones((2, 3)),
            3,
            SIGNAL_2D,
            None,
            None,
            False,
            np.ones((1, 2 * 3)),
        ),
        # 2 channels, 3 features + start, end
        (
            lambda s, sr: np.ones((2, 3)),
            3,
            SIGNAL_2D,
            pd.to_timedelta('1s'),
            pd.to_timedelta('10s'),
            False,
            np.ones((1, 2 * 3)),
        ),
        # 2 channels, 3 features, 5 frames
        (
            lambda s, sr: [[[1] * 5] * 3] * 2,
            3,
            SIGNAL_2D,
            None,
            None,
            False,
            np.ones((5, 2 * 3)),
        ),
        (
            lambda s, sr: np.ones((2, 3, 5)),
            3,
            SIGNAL_2D,
            None,
            None,
            False,
            np.ones((5, 2 * 3)),
        ),
        # 1 channel, 1 feature + mono processing
        (
            lambda s, sr: 1,
            1,
            SIGNAL_1D,
            None,
            None,
            True,
            np.ones((1, 1)),
        ),
        (
            lambda s, sr: np.ones(1),
            1,
            SIGNAL_1D,
            None,
            None,
            True,
            np.ones((1, 1)),
        ),
        (
            lambda s, sr: np.ones((1, 1)),
            1,
            SIGNAL_1D,
            None,
            None,
            True,
            np.ones((1, 1)),
        ),
        # 2 channels, 1 feature + mono processing
        (
            lambda s, sr: [1],
            1,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((1, 2)),
        ),
        (
            lambda s, sr: np.ones(1),
            1,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((1, 2)),
        ),
        (
            lambda s, sr: np.ones((1, 1)),
            1,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((1, 2)),
        ),
        # 2 channels, 3 features + mono processing
        (
            lambda s, sr: [1, 1, 1],
            3,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((1, 2 * 3)),
        ),
        (
            lambda s, sr: np.ones(3),
            3,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((1, 2 * 3)),
        ),
        (
            lambda s, sr: np.ones((1, 3, 1)),
            3,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((1, 2 * 3)),
        ),
        # 2 channels, 3 features, 5 frames + mono processing
        (
            lambda s, sr: [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            3,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((5, 2 * 3)),
        ),
        (
            lambda s, sr: np.ones((3, 5)),
            3,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((5, 2 * 3)),
        ),
        (
            lambda s, sr: np.ones((1, 3, 5)),
            3,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((5, 2 * 3)),
        ),
        # Feature extractor function returns too less dimensions
        pytest.param(
            lambda s, sr: np.ones(1),
            3,
            SIGNAL_2D,
            None,
            None,
            None,
            False,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # Feature extractor function returns too many dimensions
        pytest.param(
            lambda s, sr: np.ones((1, 1, 1, 1)),
            3,
            SIGNAL_2D,
            None,
            None,
            None,
            False,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # Feature extractor function returns wrong number of channels
        pytest.param(
            lambda s, sr: np.ones((1, 3)),
            3,
            SIGNAL_2D,
            None,
            None,
            None,
            False,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # Feature extractor function returns wrong number of channels
        pytest.param(
            lambda s, sr: np.ones((2 + 1, 3)),
            3,
            SIGNAL_2D,
            None,
            None,
            None,
            False,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # Feature extractor function returns wrong number of features
        pytest.param(
            lambda s, sr: np.ones((2, 3 + 1)),
            3,
            SIGNAL_2D,
            None,
            None,
            None,
            False,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # Feature extractor function returns wrong number of features
        pytest.param(
            lambda s, sr: np.ones((2, 3 + 1, 1)),
            3,
            SIGNAL_2D,
            None,
            None,
            None,
            False,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
    ]
)
def test_process_signal(
        process_func, num_feat, signal, start, end, expand, expected,
):
    feature = audinterface.Feature(
        feature_names=[f'f{i}' for i in range(num_feat)],
        process_func=process_func,
        channels=range(signal.shape[0]),
        process_func_is_mono=expand,
        win_dur=1,
    )
    y = feature.process_signal(
        signal,
        SAMPLING_RATE,
        start=start,
        end=end,
    )
    np.testing.assert_array_equal(y.values, expected)


@pytest.mark.parametrize(
    'index,expected_features',
    [
        (
            audinterface.utils.signal_index(
                [pd.to_timedelta('0s'), pd.to_timedelta('1s')],
                [pd.to_timedelta('2s'), pd.to_timedelta('3s')],
            ),
            np.ones((2, NUM_CHANNELS * NUM_FEATURES)),
        ),
    ],
)
def test_process_signal_from_index(index, expected_features):
    extractor = audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=feature_extractor,
        channels=range(NUM_CHANNELS),
    )
    features = extractor.process_signal_from_index(
        SIGNAL_2D,
        SAMPLING_RATE,
        index,
    )
    np.testing.assert_array_equal(features.values, expected_features)


def test_process_index(tmpdir):

    cache_root = os.path.join(tmpdir, 'cache')

    feature = audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=feature_extractor,
        channels=range(NUM_CHANNELS),
    )

    # empty

    index = audformat.segmented_index()
    df = feature.process_index(index)
    assert df.empty
    assert df.columns.tolist() == feature.column_names

    # non-empty

    # create file
    root = str(tmpdir.mkdir('wav'))
    files = ['file-1.wav', 'file-2.wav']
    paths = [os.path.join(root, file) for file in files]
    for path in paths:
        af.write(path, SIGNAL_2D, SAMPLING_RATE)
    df_expected = np.ones((2, NUM_CHANNELS * NUM_FEATURES))

    # absolute paths
    index = audformat.segmented_index(paths, [0, 1], [2, 3])
    df = feature.process_index(index)
    assert df.index.get_level_values('file')[0] == paths[0]
    np.testing.assert_array_equal(df.values, df_expected)
    assert df.columns.tolist() == feature.column_names
    df = feature.process_index(index)

    # relative paths
    index = audformat.segmented_index(files, [0, 1], [2, 3])
    df = feature.process_index(index, root=root)
    assert df.index.get_level_values('file')[0] == files[0]
    np.testing.assert_array_equal(df.values, df_expected)
    assert df.columns.tolist() == feature.column_names

    # cache result
    df = feature.process_index(
        index,
        root=root,
        cache_root=cache_root,
    )
    os.remove(paths[1])

    # fails because second file does not exist
    with pytest.raises(RuntimeError):
        feature.process_index(
            index,
            root=root,
        )

    # loading from cache still works
    df_cached = feature.process_index(
        index,
        root=root,
        cache_root=cache_root,
    )
    pd.testing.assert_frame_equal(df, df_cached)


@pytest.mark.parametrize(
    'win_dur, hop_dur',
    [
        (1, 0.5),
        (pd.to_timedelta(1, unit='s'), None),
        ('16000', None),
        ('1000ms', '500ms'),
        ('1000milliseconds', '500milliseconds'),
        (f'{SAMPLING_RATE}', f'{SAMPLING_RATE // 2}'),
        pytest.param(  # multiple frames, but win_dur is None
            None, None,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
    ],
)
def test_signal_sliding_window(win_dur, hop_dur):
    # Test sliding window with two time steps
    expected_features = np.ones((NUM_CHANNELS, 2 * NUM_FEATURES))
    extractor = audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=features_extractor_sliding_window,
        process_func_args={
            'hop_size': SAMPLING_RATE // 2,  # argument to process_func
        },
        channels=range(NUM_CHANNELS),
        win_dur=win_dur,
        hop_dur=hop_dur,
        sampling_rate=SAMPLING_RATE,
    )
    features = extractor.process_signal(
        SIGNAL_2D,
        SAMPLING_RATE,
    )
    n_time_steps = len(features)

    if isinstance(win_dur, str):
        if all(s.isdigit() for s in win_dur):
            # samples
            win_dur = pd.to_timedelta(int(win_dur) / SAMPLING_RATE, unit='s')
        else:
            win_dur = pd.to_timedelta(win_dur)
    else:
        win_dur = pd.to_timedelta(win_dur, unit='s')

    if hop_dur is None:
        hop_dur = win_dur / 2
    elif isinstance(hop_dur, str):
        if all(s.isdigit() for s in hop_dur):
            # samples
            hop_dur = pd.to_timedelta(int(hop_dur) / SAMPLING_RATE, unit='s')
        else:
            hop_dur = pd.to_timedelta(hop_dur)
    else:
        hop_dur = pd.to_timedelta(hop_dur, unit='s')

    starts = pd.timedelta_range(
        pd.to_timedelta(0),
        freq=hop_dur,
        periods=n_time_steps,
    )
    ends = starts + win_dur

    index = audinterface.utils.signal_index(starts, ends)
    pd.testing.assert_frame_equal(
        features,
        pd.DataFrame(
            expected_features,
            index=index,
            columns=extractor.column_names,
        ),
    )


def test_to_numpy():
    expected_features = np.ones((NUM_CHANNELS, NUM_FEATURES, 1))
    extractor = audinterface.Feature(
        feature_names=('o1', 'o2', 'o3'),
        process_func=feature_extractor,
        channels=range(NUM_CHANNELS),
    )
    features = extractor.process_signal(
        SIGNAL_2D,
        SAMPLING_RATE,
    )
    features = extractor.to_numpy(features)
    np.testing.assert_array_equal(features, expected_features)
