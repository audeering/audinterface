import os

import numpy as np
import pandas as pd
import pytest

import audeer
import audformat
import audiofile

import audinterface


def addition(signal, sampling_rate, value=1):
    return signal + value * signal


def identity(signal, sampling_rate):
    return signal


def mean(signal, sampling_rate, offset=0):
    return np.mean(signal + offset)


def segment(signal, sampling_rate, offset=0):
    return audinterface.utils.signal_index(
        starts=0 + offset,
        ends=1 + offset,
    )


def parse_output(output):
    r"""Return output as array or index.

    This removes the ``file`` level,
    if the output is an index
    as we cannot provide the file name
    as expected output.

    Args:
        output: output of process method

    Returns:
        output as array or index with start and end levels

    """
    if isinstance(output, pd.MultiIndex):
        output = audinterface.utils.signal_index(
            output.get_level_values('start'),
            output.get_level_values('end'),
        )
    else:
        output = output.values[0]
    return output


@pytest.mark.parametrize('signal', [np.ones((1, 3))])
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'interface_object, interface_args, process_func, process_func_args, '
    'process_func_args_during_call, expected_output',
    [
        (
            audinterface.Process,
            [],
            addition,
            None,
            None,
            2 * np.ones((1, 3)),
        ),
        (
            audinterface.Process,
            [],
            addition,
            {'value': 0},
            None,
            np.ones((1, 3)),
        ),
        (
            audinterface.Process,
            [],
            addition,
            None,
            {'value': 0},
            np.ones((1, 3)),
        ),
        (
            audinterface.Process,
            [],
            addition,
            {'value': 0},
            {'value': 2},
            3 * np.ones((1, 3)),
        ),
        (
            audinterface.Process,
            [],
            addition,
            {'value': 2},
            {'value': 0},
            np.ones((1, 3)),
        ),
        (
            audinterface.Feature,
            ['mean'],
            mean,
            None,
            None,
            1,
        ),
        (
            audinterface.Feature,
            ['mean'],
            mean,
            {'offset': 1},
            None,
            2,
        ),
        (
            audinterface.Feature,
            ['mean'],
            mean,
            None,
            {'offset': 1},
            2,
        ),
        (
            audinterface.Feature,
            ['mean'],
            mean,
            {'offset': 0},
            {'offset': 2},
            3,
        ),
        (
            audinterface.Feature,
            ['mean'],
            mean,
            {'offset': 2},
            {'offset': 0},
            1,
        ),
        (
            audinterface.Segment,
            [],
            segment,
            None,
            None,
            audinterface.utils.signal_index(0, 1),
        ),
        (
            audinterface.Segment,
            [],
            segment,
            {'offset': 1},
            None,
            audinterface.utils.signal_index(1, 2),
        ),
        (
            audinterface.Segment,
            [],
            segment,
            None,
            {'offset': 1},
            audinterface.utils.signal_index(1, 2),
        ),
        (
            audinterface.Segment,
            [],
            segment,
            {'offset': 0},
            {'offset': 2},
            audinterface.utils.signal_index(2, 3),
        ),
        (
            audinterface.Segment,
            [],
            segment,
            {'offset': 2},
            {'offset': 0},
            audinterface.utils.signal_index(0, 1),
        ),
    ],
)
def test_process(
    tmpdir,
    signal,
    sampling_rate,
    interface_object,
    interface_args,
    process_func,
    process_func_args,
    process_func_args_during_call,
    expected_output,
):

    # create test file
    folder = audeer.mkdir(tmpdir, 'wav')
    file = os.path.join(folder, 'file.wav')
    audiofile.write(file, signal, sampling_rate, bit_depth=32)

    interface = interface_object(
        *interface_args,
        process_func=process_func,
        process_func_args=process_func_args,
        verbose=False,
    )

    # signal
    y = interface.process_signal(
        signal,
        sampling_rate,
        process_func_args=process_func_args_during_call,
    )
    print(y)
    output = parse_output(y)
    np.testing.assert_equal(output, expected_output)

    # file
    y = interface.process_file(
        file,
        process_func_args=process_func_args_during_call,
    )
    output = parse_output(y)
    np.testing.assert_equal(output, expected_output)

    # files
    y = interface.process_files(
        [file],
        process_func_args=process_func_args_during_call,
    )
    output = parse_output(y)
    np.testing.assert_equal(output, expected_output)

    # folder
    y = interface.process_folder(
        folder,
        process_func_args=process_func_args_during_call,
    )
    output = parse_output(y)
    np.testing.assert_equal(output, expected_output)

    # index
    y = interface.process_index(
        audformat.filewise_index([file]),
        process_func_args=process_func_args_during_call,
    )
    output = parse_output(y)
    np.testing.assert_equal(output, expected_output)
