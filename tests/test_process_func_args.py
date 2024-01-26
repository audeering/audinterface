import os

import numpy as np
import pandas as pd
import pytest

import audeer
import audformat
import audiofile
import audobject

import audinterface


def identity(signal, sampling_rate):
    return signal


def addition(signal, sampling_rate, value=1):
    return signal + value * signal


@pytest.mark.parametrize('signal', [np.ones((1, 3))])
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize(
    'process_func, process_func_args, process_func_args_during_call, '
    'expected_output',
    [
        (
            addition,
            None,
            None,
            2 * np.ones((1, 3)),
        ),
        (
            addition,
            {'value': 0},
            None,
            np.ones((1, 3)),
        ),
        (
            addition,
            None,
            {'value': 0},
            np.ones((1, 3)),
        ),
        (
            addition,
            {'value': 0},
            {'value': 2},
            3 * np.ones((1, 3)),
        ),
        (
            addition,
            {'value': 2},
            {'value': 0},
            np.ones((1, 3)),
        ),
    ],
)
def test_process(
    tmpdir,
    signal,
    sampling_rate,
    process_func,
    process_func_args,
    process_func_args_during_call,
    expected_output,
):

    # create test file
    folder = audeer.mkdir(tmpdir, 'wav')
    file = os.path.join(folder, 'file.wav')
    audiofile.write(file, signal, sampling_rate, bit_depth=32)

    process = audinterface.Process(
        process_func=process_func,
        process_func_args=process_func_args,
        verbose=False,
    )

    # signal
    y = process.process_signal(
        signal,
        sampling_rate,
        process_func_args=process_func_args_during_call,
    )
    np.testing.assert_equal(y.values[0], expected_output)

    # file
    y = process.process_file(
        file,
        process_func_args=process_func_args_during_call,
    )
    np.testing.assert_equal(y.values[0], expected_output)

    # files
    y = process.process_files(
        [file],
        process_func_args=process_func_args_during_call,
    )
    np.testing.assert_equal(y.values[0], expected_output)

    # folder
    y = process.process_folder(
        folder,
        process_func_args=process_func_args_during_call,
    )
    np.testing.assert_equal(y.values[0], expected_output)

    # index
    y = process.process_index(
        audformat.filewise_index([file]),
        process_func_args=process_func_args_during_call,
    )
    np.testing.assert_equal(y.values[0], expected_output)
