import os

import numpy as np
import pandas as pd
import pytest

import audformat
import audiofile as af

import audinterface


SAMPLING_RATE = 8000
N_SECONDS = 10
ONES_1D = np.ones((1, SAMPLING_RATE * N_SECONDS))
ZEROS_1D = np.zeros((1, SAMPLING_RATE * N_SECONDS))
ONES_2D = np.ones((2, SAMPLING_RATE * N_SECONDS))
ONES_1S_1D = np.ones((1, SAMPLING_RATE))
ZEROS_1S_1D = np.zeros((1, SAMPLING_RATE))
STARTS = [pd.to_timedelta(i, "s") for i in range(N_SECONDS)]
ENDS = [start + pd.to_timedelta("1s") for start in STARTS]
INDEX = audinterface.utils.signal_index(STARTS, ENDS)
REVERSE_INDEX = audinterface.utils.signal_index(
    list(reversed(STARTS)), list(reversed(ENDS))
)


def predefined_process_func(signal, sr, *, num_features=2, dtype=None):
    if dtype is not None:
        data = [np.ones(num_features, dtype=dtype)] * len(INDEX)
    else:
        data = [np.ones(num_features)] * len(INDEX)
    return pd.Series(data=data, index=INDEX)


def segment_non_zeros_with_mean_mono(signal, sampling_rate):
    r"""Segment a mono signal into its nonzero parts and compute the segment means."""
    signal_1d = signal[0]
    non_zero_index = np.where(signal_1d != 0)[0]
    if len(non_zero_index) == 0:
        return pd.Series(index=audinterface.utils.signal_index([], []))
    # Find points where non-zero segments stop
    # by selecting the points where there is a gap (np.diff != 1)
    # in the non-zero indices
    split_indices = np.where(np.diff(non_zero_index) != 1)[0]
    frames = np.split(signal_1d[non_zero_index], split_indices + 1)
    frame_lengths = [len(frame) for frame in frames]
    end_indices = non_zero_index[split_indices] + 1
    # We need to include the last value of the non-zero index
    # to the end indices, since no split was done there
    end_indices = [int(end) for end in end_indices]
    max_non_zero = int(max(non_zero_index) + 1)
    end_indices.append(max_non_zero)
    start_indices = [end - length for end, length in zip(end_indices, frame_lengths)]
    ends = [pd.to_timedelta(end / sampling_rate, unit="s") for end in end_indices]
    starts = [pd.to_timedelta(start / sampling_rate, "s") for start in start_indices]
    means = [frame.mean() for frame in frames]
    index = pd.MultiIndex.from_tuples(zip(starts, ends), names=["start", "end"])
    return pd.Series(data=means, dtype=signal.dtype, index=index)


def segment_with_mean(signal, sampling_rate, *, win_size=1.0, hop_size=1.0):
    size = signal.shape[1] / sampling_rate
    starts = pd.to_timedelta(
        np.arange(0, size - win_size + (1 / sampling_rate), hop_size), unit="s"
    )
    ends = pd.to_timedelta(
        np.arange(win_size, size + (1 / sampling_rate), hop_size), unit="s"
    )
    frames = audinterface.utils.sliding_window(
        signal, sampling_rate, win_size, hop_size
    )
    means = frames.mean(axis=(0, 1))
    index = pd.MultiIndex.from_tuples(zip(starts, ends), names=["start", "end"])
    return pd.Series(data=means, index=index)


def segment_with_mean_std(signal, sampling_rate, *, win_size=1.0, hop_size=1.0):
    size = signal.shape[1] / sampling_rate
    starts = pd.to_timedelta(
        np.arange(0, size - win_size + (1 / sampling_rate), hop_size), unit="s"
    )
    ends = pd.to_timedelta(
        np.arange(win_size, size + (1 / sampling_rate), hop_size), unit="s"
    )
    frames = audinterface.utils.sliding_window(
        signal, sampling_rate, win_size, hop_size
    )
    means = frames.mean(axis=(0, 1))
    stds = frames.std(axis=(0, 1))
    index = pd.MultiIndex.from_tuples(zip(starts, ends), names=["start", "end"])
    features = list(np.stack((means, stds), axis=-1))
    return pd.Series(data=features, index=index)


def write_files(paths, signals, sampling_rate):
    for path, signal in zip(paths, signals):
        af.write(path, signal, sampling_rate)


@pytest.mark.parametrize(
    "signal, sampling_rate, segment_with_feature, expected",
    [
        # invalid multiindex return type
        pytest.param(
            ONES_1D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=lambda x, sr: pd.Series(
                    data=[1.0],
                    index=pd.MultiIndex.from_tuples(
                        [("a", "b")], names=["start", "end"]
                    ),
                ),
            ),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # incorrectly returning dataframe instead of series
        pytest.param(
            ONES_1D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=lambda x, sr: pd.DataFrame(
                    data={"feature": []},
                    index=audinterface.utils.signal_index(),
                ),
            ),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # incorrectly returning index instead of series
        pytest.param(
            ONES_1D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=lambda x, sr: audinterface.utils.signal_index(),
            ),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # incorrectly returning series with filewise index
        pytest.param(
            ONES_1D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=lambda x, sr: pd.Series(
                    index=audformat.filewise_index(),
                ),
            ),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # incorrectly returning series with wrong number of features
        pytest.param(
            ONES_1D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=lambda x, sr: pd.Series(
                    data=[np.ones(2)] * len(INDEX), index=INDEX
                ),
            ),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            ONES_1D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            pd.Series(index=audinterface.utils.signal_index([], [])),
        ),
        (
            ONES_1D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=lambda x, sr: pd.Series(
                    data=[np.ones(1)] * len(INDEX), index=INDEX
                ),
            ),
            pd.Series(data=[np.ones(1)] * len(INDEX), index=INDEX),
        ),
        (
            ONES_1D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=lambda x, sr: pd.Series(
                    data=[1] * len(INDEX), index=INDEX
                ),
            ),
            pd.Series(data=[np.ones(1)] * len(INDEX), index=INDEX),
        ),
        (
            ONES_1D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: pd.Series(
                    data=[np.ones(3)] * len(INDEX), index=INDEX
                ),
            ),
            pd.Series(data=[np.ones(3)] * len(INDEX), index=INDEX),
        ),
        (
            ONES_1D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: pd.Series(
                    data=[[1, 1, 1]] * len(INDEX), index=INDEX
                ),
            ),
            pd.Series(data=[np.ones(3)] * len(INDEX), index=INDEX),
        ),
        (
            ONES_2D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=lambda x, sr: pd.Series(
                    data=[np.ones(1)] * len(INDEX), index=INDEX
                ),
                channels=1,
            ),
            pd.Series(data=[np.ones(1)] * len(INDEX), index=INDEX),
        ),
        (
            ONES_2D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: pd.Series(
                    data=[np.ones((3))] * len(INDEX), index=INDEX
                ),
                channels=range(2),
            ),
            pd.Series(data=[np.ones(3)] * len(INDEX), index=INDEX),
        ),
        (
            ONES_2D,
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: pd.Series(
                    data=[np.ones(3)] * len(REVERSE_INDEX), index=REVERSE_INDEX
                ),
            ),
            pd.Series(data=[np.ones(3)] * len(REVERSE_INDEX), index=REVERSE_INDEX),
        ),
        (
            np.concat((ONES_1D, ZEROS_1D)),
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names=["mean"],
                process_func=segment_non_zeros_with_mean_mono,
                channels=range(2),
            ),
            pd.Series(
                data=[np.array([1])],
                index=audinterface.utils.signal_index([0], [10]),
            ),
        ),
        # test with float64
        (
            np.concat([ONES_1S_1D, ZEROS_1S_1D] * 5, axis=1).astype(np.float64),
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names=["mean"],
                process_func=segment_non_zeros_with_mean_mono,
                channels=0,
            ),
            pd.Series(
                data=[np.array([1], dtype=np.float64)] * 5,
                index=audinterface.utils.signal_index(
                    [i * 2 for i in range(5)], [(i * 2) + 1 for i in range(5)]
                ),
            ),
        ),
        # test with float32
        (
            np.concat([ONES_1S_1D, ZEROS_1S_1D] * 5, axis=1).astype(np.float32),
            SAMPLING_RATE,
            audinterface.SegmentWithFeature(
                feature_names=["mean"],
                process_func=segment_non_zeros_with_mean_mono,
                channels=0,
            ),
            pd.Series(
                data=[np.array([1], dtype=np.float32)] * 5,
                index=audinterface.utils.signal_index(
                    [i * 2 for i in range(5)], [(i * 2) + 1 for i in range(5)]
                ),
            ),
        ),
    ],
)
def test_call(signal, sampling_rate, segment_with_feature, expected):
    pd.testing.assert_series_equal(
        segment_with_feature(signal, sampling_rate),
        expected,
    )


@pytest.mark.parametrize(
    "num_channels",
    [1, 2],
)
@pytest.mark.parametrize(
    "num_features",
    [1, 2, 3],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_file(tmpdir, num_channels, num_features, dtype):
    feature_names = [f"f{i}" for i in range(num_features)]
    segmentwithfeature = audinterface.SegmentWithFeature(
        feature_names=feature_names,
        process_func=lambda x, sr: pd.Series(
            data=[np.ones(num_features, dtype=dtype)] * len(INDEX), index=INDEX
        ),
        sampling_rate=None,
        channels=range(num_channels),
        resample=False,
        verbose=False,
    )
    # create test file
    root = str(tmpdir.mkdir("wav"))
    file = "file.wav"
    path = os.path.join(root, file)
    af.write(path, ONES_2D, SAMPLING_RATE)

    expected_feats = {
        feature: np.ones(len(INDEX), dtype=dtype) for feature in feature_names
    }
    # test absolute path
    result = segmentwithfeature.process_file(path)
    expected_index = audformat.segmented_index(
        files=[path] * len(INDEX),
        starts=INDEX.levels[0],
        ends=INDEX.levels[1],
    )
    expected_frame = pd.DataFrame(index=expected_index, data=expected_feats)
    pd.testing.assert_frame_equal(result, expected_frame)
    result = segmentwithfeature.process_file(path, start=pd.to_timedelta("1s"))
    expected_index = audformat.segmented_index(
        files=[path] * len(INDEX),
        starts=INDEX.levels[0] + pd.to_timedelta("1s"),
        ends=INDEX.levels[1] + pd.to_timedelta("1s"),
    )
    expected_frame = pd.DataFrame(index=expected_index, data=expected_feats)
    pd.testing.assert_frame_equal(result, expected_frame)

    # test relative path
    result = segmentwithfeature.process_file(file, root=root)
    expected_index = audformat.segmented_index(
        files=[file] * len(INDEX),
        starts=INDEX.levels[0],
        ends=INDEX.levels[1],
    )
    expected_frame = pd.DataFrame(index=expected_index, data=expected_feats)
    pd.testing.assert_frame_equal(result, expected_frame)

    result = segmentwithfeature.process_file(
        file, root=root, start=pd.to_timedelta("1s")
    )
    expected_index = audformat.segmented_index(
        files=[file] * len(INDEX),
        starts=INDEX.levels[0] + pd.to_timedelta("1s"),
        ends=INDEX.levels[1] + pd.to_timedelta("1s"),
    )
    expected_frame = pd.DataFrame(index=expected_index, data=expected_feats)
    pd.testing.assert_frame_equal(result, expected_frame)


@pytest.mark.parametrize("num_features", [1, 3])
@pytest.mark.parametrize("num_channels", [1, 2])
@pytest.mark.parametrize("index", [INDEX, REVERSE_INDEX])
@pytest.mark.parametrize("num_files", [3])
@pytest.mark.parametrize("num_workers", [1, 2, None])
@pytest.mark.parametrize("multiprocessing", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_folder(
    tmpdir,
    num_features,
    num_channels,
    index,
    num_files,
    num_workers,
    multiprocessing,
    dtype,
):
    feature_names = [f"o{i}" for i in range(num_features)]

    segmentwithfeature = audinterface.SegmentWithFeature(
        feature_names=feature_names,
        process_func=predefined_process_func,
        sampling_rate=None,
        channels=range(num_channels),
        resample=False,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
        verbose=False,
        process_func_args={"num_features": num_features, "dtype": dtype},
    )
    path = str(tmpdir.mkdir("wav"))
    files = [f"file{n}.wav" for n in range(num_files)]
    files_abs = [os.path.join(path, file) for file in files]
    signals = [ONES_2D] * num_files
    write_files(files_abs, signals, sampling_rate=SAMPLING_RATE)

    expected_feats = {
        feature: np.ones(len(index) * num_files, dtype=dtype)
        for feature in feature_names
    }

    # folder with include_root=True
    result = segmentwithfeature.process_folder(path)
    expected_index = audformat.segmented_index(
        files=[f for f in files_abs for _ in range(len(index))],
        starts=np.tile(index.levels[0], num_files),
        ends=np.tile(index.levels[1], num_files),
    )
    expected_frame = pd.DataFrame(index=expected_index, data=expected_feats)
    pd.testing.assert_frame_equal(result, expected_frame)

    # folder with include_root=False
    result = segmentwithfeature.process_folder(path, include_root=False)
    expected_index = audformat.segmented_index(
        files=[f for f in files for _ in range(len(index))],
        starts=np.tile(index.levels[0], num_files),
        ends=np.tile(index.levels[1], num_files),
    )
    expected_frame = pd.DataFrame(index=expected_index, data=expected_feats)
    pd.testing.assert_frame_equal(result, expected_frame)

    # non-existing folder
    with pytest.raises(FileNotFoundError):
        segmentwithfeature.process_folder("bad-folder")

    # empty folder
    root = str(tmpdir.mkdir("empty"))
    result = segmentwithfeature.process_folder(root)
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            index=audformat.segmented_index(),
            dtype=object,
            columns=segmentwithfeature.feature_names,
        ),
    )


@pytest.mark.parametrize("num_workers", [1, 2, None])
@pytest.mark.parametrize("multiprocessing", [False, True])
def test_folder_default_process_func(
    tmpdir,
    num_workers,
    multiprocessing,
):
    feature_names = ["o1", "o2", "o3"]
    segmentwithfeature = audinterface.SegmentWithFeature(
        feature_names,
        process_func=None,
        sampling_rate=None,
        channels=range(2),
        resample=False,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
        verbose=False,
    )
    path = str(tmpdir.mkdir("wav"))
    files = [os.path.join(path, f"file{n}.wav") for n in range(3)]
    signals = [ONES_2D] * len(files)
    write_files(files, signals, sampling_rate=SAMPLING_RATE)
    result = segmentwithfeature.process_folder(path)
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            index=audformat.segmented_index(),
            columns=segmentwithfeature.feature_names,
        ),
    )


@pytest.mark.parametrize(
    "signals, sampling_rate, files, index, segment_with_feature, df_expected",
    [
        (
            [ONES_1D, ONES_1D],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            audformat.segmented_index(
                files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[pd.NaT, pd.NaT]
            ),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            pd.DataFrame(index=audformat.segmented_index(), columns=["feature"]),
        ),
        (
            [ONES_1S_1D, ONES_1S_1D],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            audformat.segmented_index(
                files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[pd.NaT, pd.NaT]
            ),
            audinterface.SegmentWithFeature(
                feature_names=["mean", "std"],
                process_func=segment_with_mean_std,
                channels=0,
            ),
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[1, 1]
                ),
                # Expect float32 output since signals are read as float32
                # by audiofile by default
                data={
                    "mean": np.ones(2, dtype=np.float32),
                    "std": np.zeros(2, dtype=np.float32),
                },
            ),
        ),
        (
            [ONES_1S_1D, ONES_1S_1D],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            audformat.segmented_index(
                files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[pd.NaT, pd.NaT]
            ),
            audinterface.SegmentWithFeature(
                feature_names="mean",
                process_func=segment_with_mean,
                channels=0,
            ),
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[1, 1]
                ),
                data={
                    "mean": np.ones(2, dtype=np.float32),
                },
            ),
        ),
        (
            [
                np.concat(
                    (
                        np.concat((ONES_1S_1D, ZEROS_1S_1D, ONES_1S_1D), axis=1),
                        np.concat((ZEROS_1S_1D, ONES_1S_1D, ZEROS_1S_1D), axis=1),
                    )
                ),
                np.concat(
                    (
                        np.concat((ZEROS_1S_1D, ONES_1S_1D, ZEROS_1S_1D), axis=1),
                        np.concat((ONES_1S_1D, ZEROS_1S_1D, ONES_1S_1D), axis=1),
                    )
                ),
            ],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            audformat.segmented_index(
                files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[pd.NaT, pd.NaT]
            ),
            audinterface.SegmentWithFeature(
                feature_names="mean",
                process_func=segment_non_zeros_with_mean_mono,
                channels=range(2),
            ),
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav", "f1.wav", "f2.wav"],
                    starts=[0, 2, 1],
                    ends=[1, 3, 2],
                ),
                data={
                    "mean": np.ones(3, dtype=np.float32),
                },
            ),
        ),
    ],
)
def test_index(
    tmpdir, signals, sampling_rate, files, index, segment_with_feature, df_expected
):
    root = str(tmpdir.mkdir("wav"))
    cache_root = os.path.join(tmpdir, "cache")
    paths = [os.path.join(root, file) for file in files]
    write_files(paths, signals, sampling_rate)

    # relative paths
    result = segment_with_feature.process_index(index, root=root)
    pd.testing.assert_frame_equal(result, df_expected, atol=1e-4)

    # absolute paths
    abs_index = audformat.utils.expand_file_path(index, root)
    result = segment_with_feature.process_index(abs_index)
    df_expected_abs = df_expected.copy()
    df_expected_abs.index = audformat.utils.expand_file_path(df_expected.index, root)
    pd.testing.assert_frame_equal(result, df_expected_abs, check_exact=False, atol=1e-4)

    # cache result
    df = segment_with_feature.process_index(
        index,
        root=root,
        cache_root=cache_root,
    )

    os.remove(paths[0])
    # fails because file does not exist
    with pytest.raises(RuntimeError):
        segment_with_feature.process_index(
            index,
            root=root,
        )

    # loading from cache still works
    df_cached = segment_with_feature.process_index(
        index,
        root=root,
        cache_root=cache_root,
    )
    pd.testing.assert_frame_equal(df, df_cached, atol=1e-4)


@pytest.mark.parametrize(
    "index, segment_with_feature, df_expected",
    [
        (
            audformat.segmented_index(),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            pd.DataFrame(index=audformat.segmented_index(), columns=["feature"]),
        ),
    ],
)
def test_index_empty(tmpdir, index, segment_with_feature, df_expected):
    cache_root = os.path.join(tmpdir, "cache")

    result = segment_with_feature.process_index(index)
    pd.testing.assert_frame_equal(result, df_expected, atol=1e-4)

    # cache result
    df = segment_with_feature.process_index(
        index,
    )
    df_cached = segment_with_feature.process_index(
        index,
        cache_root=cache_root,
    )
    pd.testing.assert_frame_equal(df, df_cached, atol=1e-4)


def test_process_func_args():
    def segment_func(s, sr, arg1, arg2):
        assert arg1 == "foo"
        assert arg2 == "bar"

    audinterface.Segment(
        process_func=segment_func,
        process_func_args={
            "arg1": "foo",
            "arg2": "bar",
        },
    )


@pytest.mark.parametrize(
    "signal, sampling_rate, file, segment_with_feature, expected",
    [
        (
            ONES_1D,
            SAMPLING_RATE,
            None,
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            pd.DataFrame(
                index=audinterface.utils.signal_index([], []), columns=["feature"]
            ),
        ),
        (
            ONES_1D,
            SAMPLING_RATE,
            None,
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=lambda x, sr: predefined_process_func(
                    x, sr, num_features=1, dtype=np.float32
                ),
            ),
            pd.DataFrame(
                data={"feature": np.ones(len(INDEX), dtype=np.float32)}, index=INDEX
            ),
        ),
        (
            ONES_1D,
            SAMPLING_RATE,
            "f1.wav",
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=lambda x, sr: predefined_process_func(
                    x, sr, num_features=1
                ),
            ),
            pd.DataFrame(
                data={"feature": np.ones(len(INDEX))},
                index=audformat.segmented_index(
                    files=["f1.wav"] * len(INDEX),
                    starts=INDEX.get_level_values(0),
                    ends=INDEX.get_level_values(1),
                ),
            ),
        ),
        (
            ONES_2D,
            SAMPLING_RATE,
            None,
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=lambda x, sr: predefined_process_func(
                    x, sr, num_features=1
                ),
                channels=range(2),
            ),
            pd.DataFrame(
                data={
                    "feature": np.ones(len(INDEX)),
                },
                index=INDEX,
            ),
        ),
        (
            ONES_2D.astype(np.float32),
            SAMPLING_RATE,
            None,
            audinterface.SegmentWithFeature(
                feature_names="mean",
                process_func=segment_with_mean,
                channels=range(2),
            ),
            pd.DataFrame(
                data={"mean": np.ones(10, dtype=np.float32)},
                index=audinterface.utils.signal_index(
                    starts=range(10), ends=range(1, 11)
                ),
            ),
        ),
        (
            ONES_2D,
            SAMPLING_RATE,
            None,
            audinterface.SegmentWithFeature(
                feature_names=["mean", "std"],
                process_func=segment_with_mean_std,
                channels=range(2),
            ),
            pd.DataFrame(
                data={
                    "mean": np.ones(10),
                    "std": np.zeros(10),
                },
                index=audinterface.utils.signal_index(
                    starts=range(10), ends=range(1, 11)
                ),
            ),
        ),
        (
            np.concat((np.concat([ONES_1S_1D, ZEROS_1S_1D] * 5, axis=1), ONES_1D)),
            SAMPLING_RATE,
            None,
            audinterface.SegmentWithFeature(
                feature_names=["mean"],
                process_func=segment_non_zeros_with_mean_mono,
                channels=range(2),
            ),
            pd.DataFrame(
                data={
                    "mean": np.ones(5),
                },
                index=audinterface.utils.signal_index(
                    [i * 2 for i in range(5)], [(i * 2) + 1 for i in range(5)]
                ),
            ),
        ),
        # min_signal_dur=2.0
        (
            ONES_1S_1D,
            SAMPLING_RATE,
            None,
            audinterface.SegmentWithFeature(
                feature_names=["n_samples"],
                process_func=lambda x, sr: pd.Series(
                    [x.shape[1]], index=audinterface.utils.signal_index([0], [2])
                ),
                min_signal_dur=2.0,
            ),
            pd.DataFrame(
                data={
                    "n_samples": SAMPLING_RATE * 2,
                },
                index=audinterface.utils.signal_index([0], [2]),
            ),
        ),
        # max_signal_dur=1.0
        (
            ONES_1D,
            SAMPLING_RATE,
            None,
            audinterface.SegmentWithFeature(
                feature_names=["n_samples"],
                process_func=lambda x, sr: pd.Series(
                    [x.shape[1]], index=audinterface.utils.signal_index([0], [1])
                ),
                max_signal_dur=1.0,
            ),
            pd.DataFrame(
                data={
                    "n_samples": SAMPLING_RATE,
                },
                index=audinterface.utils.signal_index([0], [1]),
            ),
        ),
    ],
)
def test_signal(signal, sampling_rate, file, segment_with_feature, expected):
    result = segment_with_feature.process_signal(signal, sampling_rate, file=file)
    pd.testing.assert_frame_equal(
        result,
        expected,
    )


@pytest.mark.parametrize(
    "signal, sampling_rate, index, segment_with_feature, df_expected",
    [
        (
            ONES_1D,
            SAMPLING_RATE,
            audformat.segmented_index(),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            pd.DataFrame(index=audformat.segmented_index(), columns=["feature"]),
        ),
        (
            ONES_1D,
            SAMPLING_RATE,
            audformat.segmented_index(files=["f1.wav"], starts=[0], ends=[pd.NaT]),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            pd.DataFrame(index=audformat.segmented_index(), columns=["feature"]),
        ),
        (
            ONES_1D,
            SAMPLING_RATE,
            audinterface.utils.signal_index(starts=[0], ends=[pd.NaT]),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            pd.DataFrame(index=audinterface.utils.signal_index(), columns=["feature"]),
        ),
        (
            ONES_1S_1D,
            SAMPLING_RATE,
            audformat.segmented_index(
                files=[
                    "f1.wav",
                ],
                starts=[0],
                ends=[pd.NaT],
            ),
            audinterface.SegmentWithFeature(
                feature_names=["mean", "std"],
                process_func=segment_with_mean_std,
                channels=0,
            ),
            pd.DataFrame(
                index=audformat.segmented_index(files=["f1.wav"], starts=[0], ends=[1]),
                data={"mean": np.ones(1), "std": np.zeros(1)},
            ),
        ),
        (
            ONES_1S_1D,
            SAMPLING_RATE,
            audinterface.utils.signal_index(
                starts=[0],
                ends=[pd.NaT],
            ),
            audinterface.SegmentWithFeature(
                feature_names=["mean", "std"],
                process_func=segment_with_mean_std,
                channels=0,
            ),
            pd.DataFrame(
                index=audinterface.utils.signal_index(starts=[0], ends=[1]),
                data={"mean": np.ones(1), "std": np.zeros(1)},
            ),
        ),
    ],
)
def test_signal_from_index(
    signal, sampling_rate, index, segment_with_feature, df_expected
):
    result = segment_with_feature.process_signal_from_index(
        signal, sampling_rate, index
    )
    pd.testing.assert_frame_equal(result, df_expected, atol=1e-4)


@pytest.mark.parametrize(
    "signals, sampling_rate, files, table, segment_with_feature, df_expected",
    [
        pytest.param(
            [],
            SAMPLING_RATE,
            [],
            audformat.segmented_index(),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            [ONES_1D, ONES_1D],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[pd.NaT, pd.NaT]
                ),
                data=np.ones(2),
                name="feature",
            ),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            [ONES_1D, ONES_1D],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[pd.NaT, pd.NaT]
                ),
                data={"feature": np.ones(2)},
            ),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            [ONES_1D, ONES_1D],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[pd.NaT, pd.NaT]
                ),
                data=np.ones(2),
                name="label",
            ),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            pd.DataFrame(
                index=audformat.segmented_index(),
                data={"label": np.ones(0)},  # add empty label data for correct dtype
                columns=["feature", "label"],
            ),
        ),
        (
            [ONES_1D, ONES_1D],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[pd.NaT, pd.NaT]
                )
            ),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            pd.DataFrame(index=audformat.segmented_index(), columns=["feature"]),
        ),
        (
            [ONES_2D, ONES_2D],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[pd.NaT, pd.NaT]
                ),
                data=range(2),
                name="label",
            ),
            audinterface.SegmentWithFeature(
                feature_names=["f1", "f2", "f3"],
                process_func=lambda x, sr: pd.Series(
                    data=[np.ones(3)] * len(REVERSE_INDEX), index=REVERSE_INDEX
                ),
            ),
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav"] * len(REVERSE_INDEX)
                    + ["f2.wav"] * len(REVERSE_INDEX),
                    starts=REVERSE_INDEX.get_level_values("start").to_list() * 2,
                    ends=REVERSE_INDEX.get_level_values("end").to_list() * 2,
                ),
                data={
                    "f1": np.ones(2 * len(REVERSE_INDEX)),
                    "f2": np.ones(2 * len(REVERSE_INDEX)),
                    "f3": np.ones(2 * len(REVERSE_INDEX)),
                    "label": [0] * len(REVERSE_INDEX) + [1] * len(REVERSE_INDEX),
                },
            ),
        ),
        (
            [ONES_2D, ONES_2D],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[1, 1]
                ),
                data=range(2),
                name="label",
            ),
            audinterface.SegmentWithFeature(
                feature_names="mean",
                process_func=segment_with_mean,
                channels=0,
            ),
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[1, 1]
                ),
                # Expect float32 output since signals are read as float32
                # by audiofile by default
                data={
                    "mean": np.ones(2, dtype=np.float32),
                    "label": range(2),
                },
            ),
        ),
        (
            [ONES_2D, ONES_2D],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[1, 1]
                ),
                data=range(2),
                name="label",
            ),
            audinterface.SegmentWithFeature(
                feature_names=["mean", "std"],
                process_func=segment_with_mean_std,
                channels=0,
            ),
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[1, 1]
                ),
                data={
                    "mean": np.ones(2, dtype=np.float32),
                    "std": np.zeros(2, dtype=np.float32),
                    "label": range(2),
                },
            ),
        ),
        (
            [ONES_1S_1D, ONES_1S_1D],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[pd.NaT, pd.NaT]
                ),
                data={"label": range(2)},
            ),
            audinterface.SegmentWithFeature(
                feature_names=["mean", "std"],
                process_func=segment_with_mean_std,
                channels=0,
            ),
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[1, 1]
                ),
                data={
                    "mean": np.ones(2, dtype=np.float32),
                    "std": np.zeros(2, dtype=np.float32),
                    "label": range(2),
                },
            ),
        ),
        (
            [ONES_2D, ONES_2D],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[1, 2]
                ),
                data=range(2),
                name="label",
            ),
            audinterface.SegmentWithFeature(
                feature_names=["mean", "std"],
                process_func=segment_with_mean_std,
                channels=range(2),
            ),
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav", "f2.wav"],
                    starts=[0, 0, 1],
                    ends=[1, 1, 2],
                ),
                data={
                    "mean": np.ones(3, dtype=np.float32),
                    "std": np.zeros(3, dtype=np.float32),
                    "label": [0, 1, 1],
                },
            ),
        ),
        (
            [
                np.concat((ONES_1S_1D, ZEROS_1S_1D), axis=1),
                ONES_1S_1D,
            ],
            SAMPLING_RATE,
            ["f1.wav", "f2.wav"],
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav", "f2.wav"], starts=[0, 0], ends=[pd.NaT, pd.NaT]
                ),
                data={"label": [0, 1]},
            ),
            audinterface.SegmentWithFeature(
                feature_names=["mean", "std"],
                process_func=lambda x, sr: segment_with_mean_std(
                    x, sr, win_size=1, hop_size=0.5
                ),
                channels=0,
            ),
            pd.DataFrame(
                index=audformat.segmented_index(
                    files=["f1.wav", "f1.wav", "f1.wav", "f2.wav"],
                    starts=[0.0, 0.5, 1.0, 0.0],
                    ends=[1.0, 1.5, 2, 1.0],
                ),
                data={
                    "mean": np.array([1, 0.5, 0, 1], dtype=np.float32),
                    "std": np.array([0, 0.5, 0, 0], dtype=np.float32),
                    "label": [0, 0, 0, 1],
                },
            ),
        ),
    ],
)
def test_table(
    tmpdir, signals, sampling_rate, files, table, segment_with_feature, df_expected
):
    root = str(tmpdir.mkdir("wav"))
    cache_root = os.path.join(tmpdir, "cache")
    paths = [os.path.join(root, file) for file in files]
    for path, signal in zip(paths, signals):
        af.write(path, signal, sampling_rate)

    # relative paths
    result = segment_with_feature.process_table(table, root=root)
    pd.testing.assert_frame_equal(result, df_expected, atol=1e-4)

    # absolute paths
    abs_index = audformat.utils.expand_file_path(table.index, root)
    abs_table = table.copy()
    abs_table.index = abs_index
    result = segment_with_feature.process_table(abs_table)
    df_expected_abs = df_expected.copy()
    df_expected_abs.index = audformat.utils.expand_file_path(df_expected.index, root)
    pd.testing.assert_frame_equal(result, df_expected_abs, check_exact=False, atol=1e-4)

    # cache result
    df = segment_with_feature.process_table(
        table,
        root=root,
        cache_root=cache_root,
    )

    os.remove(paths[0])
    # fails because second file does not exist
    with pytest.raises(RuntimeError):
        segment_with_feature.process_table(
            table,
            root=root,
        )

    # loading from cache still works
    df_cached = segment_with_feature.process_table(
        table,
        root=root,
        cache_root=cache_root,
    )
    pd.testing.assert_frame_equal(df, df_cached, atol=1e-4)


@pytest.mark.parametrize(
    "table, segment_with_feature, df_expected",
    [
        (
            pd.Series(index=audformat.segmented_index(), name="label"),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            pd.DataFrame(
                index=audformat.segmented_index(), columns=["feature", "label"]
            ),
        ),
        (
            pd.DataFrame(index=audformat.segmented_index()),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            pd.DataFrame(index=audformat.segmented_index(), columns=["feature"]),
        ),
        (
            pd.DataFrame(index=audformat.segmented_index(), columns=["label"]),
            audinterface.SegmentWithFeature(
                feature_names="feature",
                process_func=None,
            ),
            pd.DataFrame(
                index=audformat.segmented_index(), columns=["feature", "label"]
            ),
        ),
    ],
)
def test_table_empty(tmpdir, table, segment_with_feature, df_expected):
    cache_root = os.path.join(tmpdir, "cache")

    # relative paths
    result = segment_with_feature.process_table(table)
    pd.testing.assert_frame_equal(result, df_expected, atol=1e-4)

    # cache result
    df = segment_with_feature.process_table(
        table,
        cache_root=cache_root,
    )
    df_cached = segment_with_feature.process_table(
        table,
        cache_root=cache_root,
    )
    pd.testing.assert_frame_equal(df, df_cached, atol=1e-4)
