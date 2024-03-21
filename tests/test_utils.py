import re

import numpy as np
import pandas as pd
import pytest

import audeer
import audformat
import audiofile

import audinterface


def to_array(value):
    if value is not None:
        if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
            value = value.tolist()
        elif not isinstance(value, list):
            value = [value]
    return value


@pytest.mark.parametrize(
    "obj",
    [
        audinterface.utils.signal_index(),
        pytest.param(  # invalid start type
            pd.MultiIndex.from_arrays(
                [
                    [0.0, 1.0],
                    pd.to_timedelta([1.0, 2.0], unit="s"),
                ],
                names=["start", "end"],
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # invalid end type
            pd.MultiIndex.from_arrays(
                [
                    pd.to_timedelta([0.0, 1.0], unit="s"),
                    [1.0, 2.0],
                ],
                names=["start", "end"],
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_assert_index(obj):
    audinterface.core.utils.assert_index(obj)


@pytest.mark.parametrize(
    "starts,ends",
    [
        (
            None,
            None,
        ),
        (
            [],
            [],
        ),
        (
            pd.Timedelta("0s"),
            None,
        ),
        (
            pd.Timedelta("0s"),
            pd.Timedelta("1s"),
        ),
        (
            [pd.Timedelta("0s"), pd.Timedelta("1s")],
            None,
        ),
        (
            None,
            [pd.Timedelta("1s"), pd.Timedelta("2s")],
        ),
        (
            [pd.Timedelta("0s"), pd.Timedelta("1s")],
            [pd.Timedelta("1s"), pd.Timedelta("2s")],
        ),
        (
            pd.timedelta_range("0s", freq="1s", periods=2),
            pd.timedelta_range("1s", freq="1s", periods=2),
        ),
        pytest.param(  # len starts != len ends
            [pd.Timedelta("0s"), pd.Timedelta("1s")],
            [pd.Timedelta("1s")],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # duplicates
            [pd.Timedelta("0s"), pd.Timedelta("0s")],
            [pd.Timedelta("1s"), pd.Timedelta("1s")],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_create_segmented_index(starts, ends):
    index = audinterface.utils.signal_index(starts=starts, ends=ends)

    starts = to_array(starts)
    ends = to_array(ends)

    if starts is None and ends is None:
        assert index.get_level_values(audformat.define.IndexField.START).tolist() == []
        assert index.get_level_values(audformat.define.IndexField.END).tolist() == []

    else:
        if starts is not None:
            assert (
                index.get_level_values(audformat.define.IndexField.START).tolist()
                == starts
            )
        else:
            assert index.get_level_values(
                audformat.define.IndexField.START
            ).tolist() == [pd.Timedelta(0)] * len(ends)

        if ends is not None:
            assert (
                index.get_level_values(audformat.define.IndexField.END).tolist() == ends
            )
        else:
            assert index.get_level_values(audformat.define.IndexField.END).tolist() == [
                pd.NaT
            ] * len(starts)


def test_read_audio(tmpdir):
    # Ensures that we apply the same rounding
    # when reading with `audinterface.utils.read_audio()`
    # with `start` and `end`
    # or when using `start` and `end` with `process_signal()`.

    # Use critical `start` and `end` values
    # as reported in
    # https://github.com/audeering/audinterface/issues/123
    start = pd.Timedelta("0 days 00:00:01.140000")
    end = pd.Timedelta("0 days 00:00:01.560000")

    sampling_rate = 16000
    signal = np.zeros((1, 3 * sampling_rate))

    # Use `audinterface.core.utils.segment_to_indices()` as ground truth
    start_i, end_i = audinterface.core.utils.segment_to_indices(
        signal,
        sampling_rate,
        start,
        end,
    )
    signal[:, start_i:end_i] = 0.9

    audio_file = audeer.path(tmpdir, "signal.wav")
    audiofile.write(audio_file, signal, sampling_rate)

    signal, _ = audinterface.utils.read_audio(
        audio_file,
        start=start,
        end=end,
    )
    expected_signal, _ = audiofile.read(audio_file, always_2d=True)
    expected_signal = expected_signal[:, start_i:end_i]

    np.testing.assert_array_equal(signal, expected_signal)


@pytest.mark.parametrize(
    "starts, ends, expected",
    [
        (
            1,
            2,
            pd.MultiIndex.from_arrays(
                [
                    pd.TimedeltaIndex([pd.Timedelta("0 days 00:00:01")]),
                    pd.TimedeltaIndex([pd.Timedelta("0 days 00:00:02")]),
                ],
                names=["start", "end"],
            ),
        ),
        (
            [1, 2],
            [3, 4],
            pd.MultiIndex.from_arrays(
                [
                    pd.TimedeltaIndex(
                        [
                            pd.Timedelta("0 days 00:00:01"),
                            pd.Timedelta("0 days 00:00:02"),
                        ]
                    ),
                    pd.TimedeltaIndex(
                        [
                            pd.Timedelta("0 days 00:00:03"),
                            pd.Timedelta("0 days 00:00:04"),
                        ]
                    ),
                ],
                names=["start", "end"],
            ),
        ),
        (
            [pd.Timedelta("0 days 00:00:35.511437999")],
            [36],
            pd.MultiIndex.from_arrays(
                [
                    pd.TimedeltaIndex([pd.Timedelta("0 days 00:00:35.511437999")]),
                    pd.TimedeltaIndex([pd.Timedelta("0 days 00:00:36")]),
                ],
                names=["start", "end"],
            ),
        ),
    ],
)
def test_signal_index(starts, ends, expected):
    index = audinterface.utils.signal_index(starts, ends)
    pd.testing.assert_index_equal(index, expected)


@pytest.mark.parametrize(
    "signal, sampling_rate, win_dur, hop_dur, expected",
    [
        # empty
        (
            np.array([]),
            1,
            1,
            1,
            np.array([]),
        ),
        # single dimension
        (
            np.array([0, 1, 2, 3]),
            1,
            1,
            1,
            np.array([[[0, 1, 2, 3]]]),
        ),
        (
            np.array([[0, 1, 2, 3]]),
            1,
            1,
            1,
            np.array([[[0, 1, 2, 3]]]),
        ),
        (
            np.array([[0, 1, 2, 3]]),
            1,
            2,
            1,
            np.array([[[0, 1, 2], [1, 2, 3]]]),
        ),
        (
            np.array([[0, 1, 2, 3]]),
            1,
            2,
            2,
            np.array([[[0, 2], [1, 3]]]),
        ),
        (
            np.array([[0, 1, 2, 3]]),
            1,
            2,
            3,
            np.array([[[0], [1]]]),
        ),
        (
            np.array([[0, 1, 2, 3]]),
            1,
            1,
            2,
            np.array([[[0, 2]]]),
        ),
        (
            np.array([[0, 1, 2, 3]]),
            1,
            2,
            10,
            np.array([[[0], [1]]]),
        ),
        (
            np.array([[0, 1, 2, 3]]),
            1,
            10,
            2,
            np.array([]),
        ),
        # multiple dimensions
        (
            np.array(
                [
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                    [30, 31, 32, 33],
                ],
            ),
            1,
            1,
            1,
            np.array(
                [
                    [
                        [10, 11, 12, 13],
                    ],
                    [
                        [20, 21, 22, 23],
                    ],
                    [
                        [30, 31, 32, 33],
                    ],
                ]
            ),
        ),
        (
            np.array(
                [
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                    [30, 31, 32, 33],
                ],
            ),
            1,
            2,
            1,
            np.array(
                [
                    [
                        [10, 11, 12],
                        [11, 12, 13],
                    ],
                    [
                        [20, 21, 22],
                        [21, 22, 23],
                    ],
                    [
                        [30, 31, 32],
                        [31, 32, 33],
                    ],
                ],
            ),
        ),
        (
            np.array(
                [
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                    [30, 31, 32, 33],
                ],
            ),
            1,
            2,
            2,
            np.array(
                [
                    [
                        [10, 12],
                        [11, 13],
                    ],
                    [
                        [20, 22],
                        [21, 23],
                    ],
                    [
                        [30, 32],
                        [31, 33],
                    ],
                ],
            ),
        ),
        (
            np.array(
                [
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                    [30, 31, 32, 33],
                ],
            ),
            1,
            2,
            3,
            np.array(
                [
                    [[10], [11]],
                    [
                        [20],
                        [21],
                    ],
                    [
                        [30],
                        [31],
                    ],
                ],
            ),
        ),
        (
            np.array(
                [
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                    [30, 31, 32, 33],
                ],
            ),
            1,
            1,
            2,
            np.array(
                [
                    [
                        [10, 12],
                    ],
                    [
                        [20, 22],
                    ],
                    [
                        [30, 32],
                    ],
                ],
            ),
        ),
        (
            np.array(
                [
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                    [30, 31, 32, 33],
                ],
            ),
            1,
            2,
            10,
            np.array(
                [
                    [
                        [10],
                        [11],
                    ],
                    [
                        [20],
                        [21],
                    ],
                    [
                        [30],
                        [31],
                    ],
                ],
            ),
        ),
        (
            np.array(
                [
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                    [30, 31, 32, 33],
                ],
            ),
            1,
            10,
            2,
            np.array([]),
        ),
        # invalid win duration
        pytest.param(
            np.array([]),
            1,
            0.999,
            1,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # invalid hop duration
        pytest.param(
            np.array([]),
            1,
            1,
            0.999,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_sliding_window(signal, sampling_rate, win_dur, hop_dur, expected):
    frames = audinterface.utils.sliding_window(
        signal,
        sampling_rate,
        win_dur,
        hop_dur,
    )
    np.testing.assert_equal(frames, expected)


@pytest.mark.parametrize(
    "duration, expected",
    [
        (
            10,
            pd.to_timedelta(10, unit="s"),
        ),
        (
            [1],
            [pd.Timedelta("0 days 00:00:01")],
        ),
        (
            [1, 2],
            [pd.Timedelta("0 days 00:00:01"), pd.Timedelta("0 days 00:00:02")],
        ),
        (
            [1, pd.Timedelta("0 days 00:00:02")],
            [pd.Timedelta("0 days 00:00:01"), pd.Timedelta("0 days 00:00:02")],
        ),
        # Example with high precision
        # https://github.com/audeering/audinterface/issues/134
        (
            pd.Timedelta("0 days 00:00:35.511437999"),
            pd.Timedelta("0 days 00:00:35.511437999"),
        ),
        (
            [1e-09, 1],
            [
                pd.Timedelta("0 days 00:00:00.000000001"),
                pd.Timedelta("0 days 00:00:01"),
            ],
        ),
    ],
)
def test_to_timedelta(duration, expected):
    assert audinterface.utils.to_timedelta(duration) == expected


@pytest.mark.parametrize(
    "durations, sampling_rate, error_msg, error",
    [
        (
            "200",
            None,
            (
                "You have to provide 'sampling_rate' "
                "when specifying the duration in samples "
                "as you did with '200'. "
            ),
            ValueError,
        ),
        (
            [200, "200"],
            None,
            (
                "You have to provide 'sampling_rate' "
                "when specifying the duration in samples "
                "as you did with '200'. "
            ),
            ValueError,
        ),
        (
            "200 a b",
            None,
            (
                "Your given duration '200 a b' "
                "is not conform to the <value><unit> pattern."
            ),
            ValueError,
        ),
        (
            [200, "200 a b"],
            None,
            (
                "Your given duration '200 a b' "
                "is not conform to the <value><unit> pattern."
            ),
            ValueError,
        ),
    ],
)
def test_to_timedelta_errors(durations, sampling_rate, error_msg, error):
    with pytest.raises(error, match=re.escape(error_msg)):
        audinterface.utils.to_timedelta(durations, sampling_rate)
