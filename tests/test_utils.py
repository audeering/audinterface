import pandas as pd
import pytest

import audformat
import audinterface


def to_array(value):
    if value is not None:
        if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
            value = value.tolist()
        elif not isinstance(value, list):
            value = [value]
    return value


@pytest.mark.parametrize(
    'obj',
    [
        audinterface.utils.signal_index(),
        pytest.param(  # invalid start type
            pd.MultiIndex.from_arrays(
                [
                    [0.0, 1.0],
                    pd.to_timedelta([1.0, 2.0], unit='s'),
                ],
                names=['start', 'end'],
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # invalid end type
            pd.MultiIndex.from_arrays(
                [
                    pd.to_timedelta([0.0, 1.0], unit='s'),
                    [1.0, 2.0],
                ],
                names=['start', 'end'],
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_assert_index(obj):
    audinterface.core.utils.assert_index(obj)


@pytest.mark.parametrize(
    'starts,ends',
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
            pd.Timedelta('0s'),
            None,
        ),
        (
            pd.Timedelta('0s'),
            pd.Timedelta('1s'),
        ),
        (
            [pd.Timedelta('0s'), pd.Timedelta('1s')],
            None,
        ),
        (
            None,
            [pd.Timedelta('1s'), pd.Timedelta('2s')],
        ),
        (
            [pd.Timedelta('0s'), pd.Timedelta('1s')],
            [pd.Timedelta('1s'), pd.Timedelta('2s')],
        ),
        (
            pd.timedelta_range('0s', freq='1s', periods=2),
            pd.timedelta_range('1s', freq='1s', periods=2),
        ),
        pytest.param(  # len starts != len ends
            [pd.Timedelta('0s'), pd.Timedelta('1s')],
            [pd.Timedelta('1s')],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # duplicates
            [pd.Timedelta('0s'), pd.Timedelta('0s')],
            [pd.Timedelta('1s'), pd.Timedelta('1s')],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_create_segmented_index(starts, ends):

    index = audinterface.utils.signal_index(starts=starts, ends=ends)

    starts = to_array(starts)
    ends = to_array(ends)

    if starts is None and ends is None:

        assert index.get_level_values(
            audformat.define.IndexField.START
        ).tolist() == []
        assert index.get_level_values(
            audformat.define.IndexField.END
        ).tolist() == []

    else:

        if starts is not None:
            assert index.get_level_values(
                audformat.define.IndexField.START
            ).tolist() == starts
        else:
            assert index.get_level_values(
                audformat.define.IndexField.START
            ).tolist() == [pd.Timedelta(0)] * len(ends)

        if ends is not None:
            assert index.get_level_values(
                audformat.define.IndexField.END
            ).tolist() == ends
        else:
            assert index.get_level_values(
                audformat.define.IndexField.END
            ).tolist() == [pd.NaT] * len(starts)
