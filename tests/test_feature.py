import os

import numpy as np
import pandas as pd
import pytest

import audeer
import audformat
import audiofile
import audiofile as af

import audinterface


SAMPLING_RATE = 8000
NUM_CHANNELS = 2
NUM_FEATURES = 3
NUM_FRAMES = 5
SIGNAL_1D = np.ones((1, SAMPLING_RATE))
SIGNAL_2D = np.ones((NUM_CHANNELS, SAMPLING_RATE))


def segment(signal, sampling_rate):
    return audinterface.utils.signal_index(
        pd.to_timedelta(0),
        pd.to_timedelta(signal.shape[1] / sampling_rate, unit="s") / 2,
    )


SEGMENT = audinterface.Segment(process_func=segment)


def feature_extractor(signal, _):
    return np.ones((NUM_CHANNELS, NUM_FEATURES))


def mean(signal, sampling_rate):
    return signal.mean(axis=1, keepdims=True)


def mean_mono(signal, sampling_rate):
    return signal.mean()


def mean_sliding_window(signal, sampling_rate, win_dur, hop_dur):
    frames = audinterface.utils.sliding_window(
        signal,
        sampling_rate,
        win_dur,
        hop_dur,
    )
    return frames.mean(axis=1, keepdims=True)


def mean_sliding_window_mono(signal, sampling_rate, win_dur, hop_dur):
    frames = audinterface.utils.sliding_window(
        signal,
        sampling_rate,
        win_dur,
        hop_dur,
    )
    return frames.mean(axis=1, keepdims=False)


def test_feature():
    # You have to specify sampling rate when win_dur is in samples
    with pytest.raises(ValueError):
        audinterface.Feature(
            feature_names=("o1", "o2", "o3"),
            process_func_applies_sliding_window=False,
            sampling_rate=None,
            win_dur="2048",
        )
    # If no win_dur is given, no error should occur
    audinterface.Feature(
        feature_names=("o1", "o2", "o3"),
        sampling_rate=None,
    )
    # Only hop_dur is given
    with pytest.raises(ValueError):
        audinterface.Feature(
            feature_names=("o1", "o2", "o3"),
            hop_dur=0.1,
        )
    audinterface.Feature(
        feature_names=("o1", "o2", "o3"),
        process_func_applies_sliding_window=False,
        win_dur="2048",
        sampling_rate=8000,
    )


@pytest.mark.parametrize(
    "signal, feature, expected",
    [
        (
            SIGNAL_1D,
            audinterface.Feature(
                feature_names="feature",
                process_func=lambda x, sr: 1,
            ),
            np.ones((1, 1, 1)),
        ),
        (
            SIGNAL_1D,
            audinterface.Feature(
                feature_names=["feature"],
                process_func=lambda x, sr: 1,
            ),
            np.ones((1, 1, 1)),
        ),
        (
            SIGNAL_1D,
            audinterface.Feature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: np.ones(3),
            ),
            np.ones((1, 3, 1)),
        ),
        (
            SIGNAL_1D,
            audinterface.Feature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: np.ones((1, 3)),
            ),
            np.ones((1, 3, 1)),
        ),
        (
            SIGNAL_1D,
            audinterface.Feature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: np.ones((3, 1)),
            ),
            np.ones((1, 3, 1)),
        ),
        (
            SIGNAL_1D,
            audinterface.Feature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: np.ones((1, 3, 5)),
            ),
            np.ones((1, 3, 5)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: np.ones((1, 3, 5)),
                channels=1,
            ),
            np.ones((1, 3, 5)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: np.ones((2, 3)),
                channels=range(2),
            ),
            np.ones((2, 3, 1)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: np.ones((2, 3, 5)),
                channels=range(2),
            ),
            np.ones((2, 3, 5)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: np.ones(3),
                channels=range(2),
                process_func_is_mono=True,
            ),
            np.ones((2, 3, 1)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: np.ones((1, 3)),
                channels=range(2),
                process_func_is_mono=True,
            ),
            np.ones((2, 3, 1)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: np.ones((1, 3, 1)),
                channels=range(2),
                process_func_is_mono=True,
            ),
            np.ones((2, 3, 1)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: np.ones((3, 5)),
                channels=range(2),
                process_func_is_mono=True,
            ),
            np.ones((2, 3, 5)),
        ),
        (
            SIGNAL_2D,
            audinterface.Feature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: np.ones((1, 3, 5)),
                channels=range(2),
                process_func_is_mono=True,
            ),
            np.ones((2, 3, 5)),
        ),
    ],
)
def test_process_call(signal, feature, expected):
    np.testing.assert_array_equal(
        feature(signal, SAMPLING_RATE),
        expected,
    )


@pytest.mark.parametrize(
    "signal, extractor",
    [
        (
            np.random.randn(1, SAMPLING_RATE),
            audinterface.Feature(
                feature_names="mean",
                process_func=lambda x, sr: np.mean(x, axis=1),
            ),
        ),
        (
            np.random.randn(2, SAMPLING_RATE),
            audinterface.Feature(
                feature_names="mean",
                process_func=lambda x, sr: np.mean(x, axis=1),
            ),
        ),
        (
            np.random.randn(2, SAMPLING_RATE),
            audinterface.Feature(
                feature_names="mean",
                process_func=lambda x, sr: np.mean(x, axis=1),
                channels=0,
            ),
        ),
    ],
)
def test_process_call_datatype(signal, extractor):
    features = extractor(signal, SAMPLING_RATE)
    assert signal.dtype == features.dtype


@pytest.mark.parametrize(
    "start, end, segment",
    [
        (None, None, None),
        (None, None, SEGMENT),
        (pd.NaT, pd.NaT, None),
        (pd.to_timedelta("0.25s"), None, None),
        (pd.to_timedelta("0.25s"), pd.NaT, None),
        (None, pd.to_timedelta("0.75s"), None),
        (pd.NaT, pd.to_timedelta("0.75s"), None),
        (pd.to_timedelta("0.25s"), pd.to_timedelta("0.75s"), None),
    ],
)
def test_process_file(tmpdir, start, end, segment):
    start_org = start
    end_org = end

    feature = audinterface.Feature(
        feature_names=("o1", "o2", "o3"),
        process_func=feature_extractor,
        sampling_rate=None,
        channels=range(NUM_CHANNELS),
        resample=False,
        segment=segment,
        verbose=False,
    )
    y_expected = np.ones((1, NUM_CHANNELS * NUM_FEATURES))

    # create test file
    root = str(tmpdir.mkdir("wav"))
    file = "file.wav"
    path = os.path.join(root, file)
    af.write(path, SIGNAL_2D, SAMPLING_RATE)

    # test absolute path
    start = start_org
    end = end_org

    y = feature.process_file(path, start=start, end=end)
    if start is None or pd.isna(start):
        start = pd.to_timedelta(0)
    if end is None or pd.isna(end):
        end = pd.to_timedelta(af.duration(path), unit="s")

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
        end = pd.to_timedelta(af.duration(path), unit="s")

    if segment is not None:
        index = segment.process_file(file, root=root)
        start = index[0][1]
        end = index[0][2]

    assert y.index.levels[0][0] == file
    assert y.index.levels[1][0] == start
    assert y.index.levels[2][0] == end
    np.testing.assert_array_equal(y, y_expected)


@pytest.mark.parametrize("num_files", [3])
@pytest.mark.parametrize("num_workers", [1, 2, None])
@pytest.mark.parametrize("multiprocessing", [False, True])
def test_process_folder(
    tmpdir,
    num_files,
    num_workers,
    multiprocessing,
):
    index = audinterface.utils.signal_index(0, 1)
    feature_names = ["o1", "o2", "o3"]
    feature = audinterface.Feature(
        feature_names,
        process_func=feature_extractor,
        sampling_rate=None,
        channels=range(NUM_CHANNELS),
        resample=False,
        verbose=False,
    )

    path = str(tmpdir.mkdir("wav"))
    files = [f"file{n}.wav" for n in range(num_files)]
    files_abs = [os.path.join(path, file) for file in files]
    for file in files_abs:
        af.write(file, SIGNAL_2D, SAMPLING_RATE)

    # folder with include_root=True
    y = feature.process_folder(path)
    y_expected = np.ones((3, NUM_CHANNELS * NUM_FEATURES))
    assert all(y.index.levels[0] == files_abs)
    assert all(y.index.levels[1] == index.levels[0])
    assert all(y.index.levels[2] == index.levels[1])
    np.testing.assert_array_equal(y.values, y_expected)

    # folder with include_root=False
    y = feature.process_folder(path, include_root=False)
    y_expected = np.ones((3, NUM_CHANNELS * NUM_FEATURES))
    assert all(y.index.levels[0] == files)
    assert all(y.index.levels[1] == index.levels[0])
    assert all(y.index.levels[2] == index.levels[1])
    np.testing.assert_array_equal(y.values, y_expected)

    # non-existing folder
    with pytest.raises(FileNotFoundError):
        feature.process_folder("bad-folder")

    # empty folder
    root = str(tmpdir.mkdir("empty"))
    df = feature.process_folder(root)
    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame(
            index=pd.Index([], dtype="object"),
            dtype=object,
            columns=feature.column_names,
        ),
    )


@pytest.mark.parametrize("num_files", [3])
@pytest.mark.parametrize("num_workers", [1, 2, None])
@pytest.mark.parametrize("multiprocessing", [False, True])
def test_process_folder_default_process_func(
    tmpdir,
    num_files,
    num_workers,
    multiprocessing,
):
    feature_names = ["o1", "o2", "o3"]
    feature = audinterface.Feature(
        feature_names,
        process_func=None,
        sampling_rate=None,
        channels=range(NUM_CHANNELS),
        resample=False,
        verbose=False,
    )

    path = str(tmpdir.mkdir("wav"))
    files = [f"file{n}.wav" for n in range(num_files)]
    files_abs = [os.path.join(path, file) for file in files]
    for file in files_abs:
        af.write(file, SIGNAL_2D, SAMPLING_RATE)

    y = feature.process_folder(path)
    assert all(y.index.levels[0] == files_abs)


def test_process_func_args():
    def process_func(s, sr, arg1, arg2):
        assert arg1 == "foo"
        assert arg2 == "bar"

    audinterface.Feature(
        feature_names=("o1", "o2", "o3"),
        process_func=process_func,
        process_func_args={
            "arg1": "foo",
            "arg2": "bar",
        },
    )


@pytest.mark.parametrize("preserve_index", [False, True])
def test_process_index(tmpdir, preserve_index):
    cache_root = os.path.join(tmpdir, "cache")

    feature = audinterface.Feature(
        feature_names=("o1", "o2", "o3"),
        process_func=feature_extractor,
        channels=range(NUM_CHANNELS),
    )

    # empty

    index = audformat.segmented_index()
    df = feature.process_index(index, preserve_index=preserve_index)
    assert df.empty
    pd.testing.assert_index_equal(df.columns, feature.column_names)

    # non-empty

    # create file
    root = str(tmpdir.mkdir("wav"))
    files = ["file-1.wav", "file-2.wav"]
    paths = [os.path.join(root, file) for file in files]
    for path in paths:
        af.write(path, SIGNAL_2D, SAMPLING_RATE)
    df_expected = np.ones((2, NUM_CHANNELS * NUM_FEATURES))

    # absolute paths segmented index
    index = audformat.segmented_index(paths, [0, 1], [None, 3])
    df = feature.process_index(index, preserve_index=preserve_index)
    assert df.index.get_level_values("file")[0] == paths[0]
    if preserve_index:
        pd.testing.assert_index_equal(df.index, index)
    np.testing.assert_array_equal(df.values, df_expected)
    pd.testing.assert_index_equal(df.columns, feature.column_names)

    # relative paths segmented index
    index = audformat.segmented_index(files, [0, 1], [None, 3])
    df = feature.process_index(index, preserve_index=preserve_index, root=root)
    assert df.index.get_level_values("file")[0] == files[0]
    if preserve_index:
        pd.testing.assert_index_equal(df.index, index)
    np.testing.assert_array_equal(df.values, df_expected)
    pd.testing.assert_index_equal(df.columns, feature.column_names)

    # absolute paths filewise index
    index = audformat.filewise_index(paths)
    df = feature.process_index(index, preserve_index=preserve_index)
    if preserve_index:
        assert df.index[0] == paths[0]
        pd.testing.assert_index_equal(df.index, index)
    else:
        assert df.index.get_level_values("file")[0] == paths[0]
    np.testing.assert_array_equal(df.values, df_expected)
    pd.testing.assert_index_equal(df.columns, feature.column_names)

    # relative paths filewise index
    index = audformat.filewise_index(files)
    df = feature.process_index(index, preserve_index=preserve_index, root=root)
    if preserve_index:
        assert df.index[0] == files[0]
        pd.testing.assert_index_equal(df.index, index)
    else:
        assert df.index.get_level_values("file")[0] == files[0]
    np.testing.assert_array_equal(df.values, df_expected)
    pd.testing.assert_index_equal(df.columns, feature.column_names)

    # cache result
    df = feature.process_index(
        index,
        preserve_index=preserve_index,
        root=root,
        cache_root=cache_root,
    )
    os.remove(paths[1])

    # fails because second file does not exist
    with pytest.raises(RuntimeError):
        feature.process_index(
            index,
            preserve_index=preserve_index,
            root=root,
        )

    # loading from cache still works
    df_cached = feature.process_index(
        index,
        preserve_index=preserve_index,
        root=root,
        cache_root=cache_root,
    )
    pd.testing.assert_frame_equal(df, df_cached)


@pytest.mark.parametrize(
    "process_func, applies_sliding_window, num_feat, signal, start, end, "
    "is_mono, expected",
    [
        # no process function
        (
            None,
            False,
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
            False,
            1,
            SIGNAL_1D,
            None,
            None,
            False,
            np.ones((1, 1)),
        ),
        (
            lambda s, sr: np.ones(1),
            False,
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
            False,
            1,
            SIGNAL_1D,
            None,
            None,
            False,
            np.ones((1, 1)),
        ),
        (
            lambda s, sr: np.ones((1, 1)),
            False,
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
            False,
            3,
            SIGNAL_1D,
            None,
            None,
            False,
            np.ones((1, 3)),
        ),
        (
            lambda s, sr: np.ones(3),
            False,
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
            False,
            3,
            SIGNAL_1D,
            None,
            None,
            False,
            np.ones((1, 3)),
        ),
        (
            lambda s, sr: np.ones((1, 3)),
            False,
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
            False,
            1,
            SIGNAL_2D,
            None,
            None,
            False,
            np.ones((1, 2)),
        ),
        (
            lambda s, sr: np.ones((2, 1)),
            False,
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
            False,
            3,
            SIGNAL_2D,
            None,
            None,
            False,
            np.ones((1, 2 * 3)),
        ),
        (
            lambda s, sr: np.ones((2, 3)),
            False,
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
            True,
            3,
            SIGNAL_2D,
            pd.to_timedelta("1s"),
            pd.to_timedelta("10s"),
            False,
            np.ones((1, 2 * 3)),
        ),
        # 2 channels, 3 features, 5 frames
        (
            lambda s, sr: [[[1] * 5] * 3] * 2,
            True,
            3,
            SIGNAL_2D,
            None,
            None,
            False,
            np.ones((5, 2 * 3)),
        ),
        (
            lambda s, sr: np.ones((2, 3, 5)),
            True,
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
            False,
            1,
            SIGNAL_1D,
            None,
            None,
            True,
            np.ones((1, 1)),
        ),
        (
            lambda s, sr: np.ones(1),
            False,
            1,
            SIGNAL_1D,
            None,
            None,
            True,
            np.ones((1, 1)),
        ),
        (
            lambda s, sr: np.ones((1, 1)),
            False,
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
            False,
            1,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((1, 2)),
        ),
        (
            lambda s, sr: np.ones(1),
            False,
            1,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((1, 2)),
        ),
        (
            lambda s, sr: np.ones((1, 1)),
            False,
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
            False,
            3,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((1, 2 * 3)),
        ),
        (
            lambda s, sr: np.ones(3),
            False,
            3,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((1, 2 * 3)),
        ),
        (
            lambda s, sr: np.ones((1, 3, 1)),
            False,
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
            True,
            3,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((5, 2 * 3)),
        ),
        (
            lambda s, sr: np.ones((3, 5)),
            True,
            3,
            SIGNAL_2D,
            None,
            None,
            True,
            np.ones((5, 2 * 3)),
        ),
        (
            lambda s, sr: np.ones((1, 3, 5)),
            True,
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
            False,
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
            False,
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
            False,
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
            False,
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
            False,
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
            False,
            3,
            SIGNAL_2D,
            None,
            None,
            None,
            False,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
    ],
)
def test_process_signal(
    process_func,
    applies_sliding_window,
    num_feat,
    signal,
    start,
    end,
    is_mono,
    expected,
):
    channels = range(signal.shape[0])
    feature_names = [f"f{i}" for i in range(num_feat)]

    feature = audinterface.Feature(
        feature_names=feature_names,
        process_func=process_func,
        process_func_applies_sliding_window=applies_sliding_window,
        channels=channels,
        process_func_is_mono=is_mono,
        win_dur=1,
    )

    df = feature.process_signal(
        signal,
        SAMPLING_RATE,
        start=start,
        end=end,
    )
    np.testing.assert_array_equal(df.values, expected)

    # test individual channels
    if len(channels) > 1:
        for channel in channels:
            np.testing.assert_array_equal(
                df[channel].values,
                expected[:, channel * num_feat : (channel + 1) * num_feat],
            )
            assert df[channel].columns.to_list() == feature_names


@pytest.mark.parametrize(
    "feature, signal, sampling_rate, index, expected",
    [
        (
            audinterface.Feature(
                feature_names=("o1", "o2", "o3"),
                process_func=feature_extractor,
                channels=range(NUM_CHANNELS),
            ),
            SIGNAL_2D,
            SAMPLING_RATE,
            audinterface.utils.signal_index(
                [pd.to_timedelta("0s"), pd.to_timedelta("1s")],
                [pd.to_timedelta("2s"), pd.to_timedelta("3s")],
            ),
            np.ones((2, NUM_CHANNELS * NUM_FEATURES)),
        ),
        (
            audinterface.Feature(
                feature_names=("string"),
                process_func=lambda x, sr, idx: ["a", "abc"][idx],
            ),
            SIGNAL_1D,
            SAMPLING_RATE,
            audinterface.utils.signal_index(
                [pd.to_timedelta("0s"), pd.to_timedelta("1s")],
                [pd.to_timedelta("2s"), pd.to_timedelta("3s")],
            ),
            np.array([["a"], ["abc"]]),
        ),
    ],
)
def test_process_signal_from_index(feature, signal, sampling_rate, index, expected):
    df = feature.process_signal_from_index(
        signal,
        sampling_rate,
        index,
    )
    np.testing.assert_array_equal(df.values, expected)


@pytest.mark.parametrize("audio", [(3, 8000)], indirect=True)  # s, Hz
@pytest.mark.parametrize(
    # `starts` and `ends`
    # are used to create a segment object
    # using audinterface.utils.signal_index()
    "starts, ends",
    [
        (None, None),
        (0, 1.5),
        (1.5, 3),
        ([0, 1.5], [1.5, 3]),
        ([0, 2], [1, 3]),
        ([0, 1], [2, 2]),
        # https://github.com/audeering/audinterface/pull/145
        ([0, 1.5], [1, 2.000000003]),
        ([0.000000003, 1.5], [1, 2]),
        ([1.000000003, 1.5], [1.1, 2]),
        ([1.000000003, 2.1], [2.000000003, 2.5]),
        # https://github.com/audeering/audinterface/issues/135
        ([0, 1], [3, 2]),
    ],
)
def test_feature_with_segment(audio, starts, ends):
    path, signal, sampling_rate = audio
    root, file = os.path.split(path)
    duration = signal.shape[1] / sampling_rate

    # Segment and process objects
    segment = audinterface.Segment(
        process_func=lambda x, sr: audinterface.utils.signal_index(starts, ends)
    )
    feature = audinterface.Feature("f")
    feature_with_segment = audinterface.Feature("f", segment=segment)

    # Expected index
    if starts is None:
        files = None
        files_abs = None
    else:
        files = [file] * len(audeer.to_list(starts))
        files_abs = [os.path.join(root, file) for file in files]
    expected = audformat.segmented_index(files, starts, ends)
    expected_folder_index = audformat.segmented_index(files_abs, starts, ends)
    expected_signal_index = audinterface.utils.signal_index(starts, ends)

    # process signal
    index = segment.process_signal(signal, sampling_rate)
    pd.testing.assert_index_equal(index, expected_signal_index)

    # process signal with start argument
    index = segment.process_signal(signal, sampling_rate, start=0)
    pd.testing.assert_index_equal(index, expected_signal_index)

    # process signal with file argument
    index = segment.process_signal(signal, sampling_rate, file=file)
    pd.testing.assert_index_equal(index, expected)

    pd.testing.assert_frame_equal(
        feature.process_index(index, root=root, preserve_index=True),
        feature_with_segment.process_signal(signal, sampling_rate, file=file),
    )

    # process signal from index
    index = segment.process_signal_from_index(
        signal,
        sampling_rate,
        audinterface.utils.signal_index(0, duration),
    )
    pd.testing.assert_index_equal(index, expected_signal_index)
    index = segment.process_signal_from_index(
        signal,
        sampling_rate,
        audformat.segmented_index(file, 0, duration),
    )
    pd.testing.assert_index_equal(index, expected)
    index = segment.process_signal_from_index(
        signal,
        sampling_rate,
        audformat.filewise_index(file),
    )
    pd.testing.assert_index_equal(index, expected)

    pd.testing.assert_frame_equal(
        feature.process_index(index, root=root, preserve_index=True),
        feature_with_segment.process_signal_from_index(
            signal,
            sampling_rate,
            audformat.filewise_index(file),
        ),
    )

    # process file
    index = segment.process_file(file, root=root)
    pd.testing.assert_index_equal(index, expected)

    pd.testing.assert_frame_equal(
        feature.process_index(index, root=root, preserve_index=True),
        feature_with_segment.process_file(file, root=root),
    )

    # process files
    index = segment.process_files([file], root=root)
    pd.testing.assert_index_equal(index, expected)

    # https://github.com/audeering/audinterface/issues/138
    pd.testing.assert_frame_equal(
        feature.process_index(index, root=root, preserve_index=True),
        feature_with_segment.process_files([file], root=root),
    )

    # process folder
    index = segment.process_folder(root)
    pd.testing.assert_index_equal(index, expected_folder_index)

    pd.testing.assert_frame_equal(
        feature.process_index(index, root=root, preserve_index=True),
        feature_with_segment.process_folder(root),
    )

    # process folder without root
    # https://github.com/audeering/audinterface/issues/139
    index = segment.process_folder(root, include_root=False)
    pd.testing.assert_index_equal(index, expected)

    pd.testing.assert_frame_equal(
        feature.process_index(index, root=root, preserve_index=True),
        feature_with_segment.process_folder(root, include_root=False),
    )

    # process index
    index = segment.process_index(audformat.filewise_index(file), root=root)
    pd.testing.assert_index_equal(index, expected)

    pd.testing.assert_frame_equal(
        feature.process_index(index, root=root, preserve_index=True),
        feature_with_segment.process_index(
            audformat.filewise_index(file),
            root=root,
        ),
    )


@pytest.mark.parametrize(
    "signal, num_channels, sampling_rate",
    [
        (SIGNAL_1D, 1, SAMPLING_RATE),
        (SIGNAL_2D, 2, SAMPLING_RATE),
    ],
)
@pytest.mark.parametrize(
    "process_func, is_mono, applies_sliding_window, feature_names",
    [
        (mean, False, False, "mean"),
        (mean_mono, True, False, "mean"),
        (mean_sliding_window, False, True, "mean"),
        (mean_sliding_window_mono, True, True, "mean"),
    ],
)
@pytest.mark.parametrize(
    "win_dur, hop_dur",
    [
        (0.5, None),
        (0.5, 0.25),
        (0.5, 0.5),
        (0.25, 0.5),
        (pd.to_timedelta(1, unit="s"), pd.to_timedelta(0.5, unit="s")),
        ("4000", "2000"),
        ("500ms", "250ms"),
        ("500milliseconds", "250milliseconds"),
        (f"{SAMPLING_RATE // 2}", f"{SAMPLING_RATE // 4}"),
    ],
)
def test_signal_sliding_window(
    tmpdir,
    signal,
    num_channels,
    sampling_rate,
    process_func,
    is_mono,
    applies_sliding_window,
    feature_names,
    win_dur,
    hop_dur,
):
    interface = audinterface.Feature(
        feature_names=feature_names,
        process_func=process_func,
        process_func_is_mono=is_mono,
        process_func_applies_sliding_window=applies_sliding_window,
        channels=range(num_channels),
        win_dur=win_dur,
        hop_dur=hop_dur,
        sampling_rate=sampling_rate,
    )

    # signal

    df = interface.process_signal(
        signal,
        sampling_rate,
    )
    n_time_steps = len(df)

    win_dur = audinterface.utils.to_timedelta(win_dur, sampling_rate)
    if hop_dur is None:
        hop_dur = win_dur / 2
    hop_dur = audinterface.utils.to_timedelta(hop_dur, sampling_rate)

    starts = pd.timedelta_range(
        pd.to_timedelta(0),
        freq=hop_dur,
        periods=n_time_steps,
    )
    ends = starts + win_dur

    index = audinterface.utils.signal_index(starts, ends)
    expected = pd.DataFrame(
        np.ones((n_time_steps, len(interface.column_names))),
        index=index,
        columns=interface.column_names,
    )
    pd.testing.assert_frame_equal(df, expected)

    # file

    file = audeer.path(tmpdir, "tmp.wav")
    audiofile.write(file, signal, sampling_rate, bit_depth=32)

    df = interface.process_file(file)

    index = audformat.segmented_index([file] * len(starts), starts, ends)
    expected = pd.DataFrame(
        np.ones((n_time_steps, len(interface.column_names))),
        index=index,
        columns=interface.column_names,
        dtype=np.float32,
    )
    pd.testing.assert_frame_equal(df, expected)


def test_signal_sliding_window_error():
    interface = audinterface.Feature(
        feature_names="mean",
        process_func=mean_sliding_window,
        process_func_args={
            "win_dur": 0.5,
            "hop_dur": 0.25,
        },
        process_func_is_mono=False,
        process_func_applies_sliding_window=True,
        channels=range(NUM_CHANNELS),
        win_dur=None,
        hop_dur=None,
        sampling_rate=SAMPLING_RATE,
    )

    # returns multiple frames but win_dur is None
    with pytest.raises(RuntimeError):
        interface.process_signal(
            SIGNAL_2D,
            SAMPLING_RATE,
        )


def test_to_numpy():
    expected_features = np.ones((NUM_CHANNELS, NUM_FEATURES, 1))
    extractor = audinterface.Feature(
        feature_names=("o1", "o2", "o3"),
        process_func=feature_extractor,
        channels=range(NUM_CHANNELS),
    )
    features = extractor.process_signal(
        SIGNAL_2D,
        SAMPLING_RATE,
    )
    features = extractor.to_numpy(features)
    np.testing.assert_array_equal(features, expected_features)
