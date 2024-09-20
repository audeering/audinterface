import json
import os
import typing

import pandas as pd
import pytest

import audeer
import audformat
import audobject

import audinterface


def identity(data):
    return data

def data_identity(data):
    return data


def length(data):
    return len(data)


class DataObject(audobject.Object):
    def __call__(self, data):
        return data[0]


def repeat(data, number=2):
    return "".join([data for _ in range(number)])


def write_text_file(file: str, data: typing.Union[dict, str]):
    r"""Store data in text file.

    Depending on the file extension,
    the data is stored in a json file
    or a txt file.

    Args:
        file: file path
        data: data to be written to ``file``

    """
    ext = audeer.file_extension(file).lower()
    with open(file, "w") as fp:
        if ext == "json":
            json.dump(data, fp)
        else:
            fp.write(data)


@pytest.mark.parametrize(
    "process_func, data, file_format, expected_data",
    [
        (identity, "abc", "txt", "abc"),
        (identity, {"a": 0}, "json", {"a": 0}),
    ],
)
def test_process_file(
    tmpdir,
    process_func,
    data,
    file_format,
    expected_data,
):
    process = audinterface.Process(process_func=process_func, verbose=False)

    # create test file
    root = audeer.mkdir(tmpdir, "test")
    file = f"file.{file_format}"
    path = os.path.join(root, file)
    write_text_file(path, data)

    # test absolute path
    y = process.process_file(path)

    expected_series = pd.Series(
        [expected_data],
        index=audformat.filewise_index(path),
    )
    print(f"{y=}")
    print(f"{expected_series=}")
    pd.testing.assert_series_equal(y, expected_series)

    # test relative path
    y = process.process_file(file, root=root)
    expected_series = pd.Series(
        [expected_data],
        index=audformat.filewise_index(file),
    )
    pd.testing.assert_series_equal(y, expected_series)


@pytest.mark.parametrize(
    "process_func, num_files, data, file_format, expected_output",
    [
        (identity, 0, "abc", "txt", []),
        (identity, 1, "abc", "txt", ["abc"]),
    ],
)
def test_process_files(
    tmpdir,
    process_func,
    num_files,
    data,
    file_format,
    expected_output,
):
    r"""Test processing of multiple text files.

    Args:
        tmpdir: tmpdir fixture
        process_func: processing function
        num_files: number of files to create from ``data``
        data: data to write into text files
        file_format: file format of text files,
            ``"json"`` or ``"txt"``
        expected_output: expected result of processing function

    """
    process = audinterface.Process(process_func=process_func, verbose=False)

    # create files
    files = []
    paths = []
    root = tmpdir
    for idx in range(num_files):
        file = f"file{idx}.{file_format}"
        path = os.path.join(root, file)
        write_text_file(path, data)
        files.append(file)
        paths.append(path)

    # test absolute paths
    y = process.process_files(paths)
    expected_y = pd.Series(
        expected_output,
        index=audformat.filewise_index(paths),
    )
    pd.testing.assert_series_equal(y, expected_y)

    # test relative paths
    y = process.process_files(files, root=root)
    expected_y = pd.Series(
        expected_output,
        index=audformat.filewise_index(files),
    )
    pd.testing.assert_series_equal(y, expected_y)


@pytest.mark.parametrize("num_files", [3])
@pytest.mark.parametrize("file_format", ["json", "txt"])
@pytest.mark.parametrize("num_workers", [1, 2, None])
@pytest.mark.parametrize("multiprocessing", [False, True])
def test_process_folder(
    tmpdir,
    num_files,
    file_format,
    num_workers,
    multiprocessing,
):
    process = audinterface.Process(
        process_func=None,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
        verbose=False,
    )

    if file_format == "json":
        data = {"a": 0}
    else:
        data = "abc"

    # Create test files
    root = audeer.mkdir(tmpdir, "text")
    files = [os.path.join(root, f"file{n}.{file_format}") for n in range(num_files)]
    for file in files:
        write_text_file(file, data)
    y = process.process_folder(root)
    pd.testing.assert_series_equal(
        y,
        process.process_files(files),
    )

    # non-existing folder
    with pytest.raises(FileNotFoundError):
        process.process_folder("bad-folder")

    # empty folder
    root = str(tmpdir.mkdir("empty"))
    y = process.process_folder(root)
    pd.testing.assert_series_equal(y, pd.Series(dtype=object))


def _get_idx_type(preserve_index, segment_is_None, idx):
    """Get expected index type.

    preserve_index: if ``True``
    and :attr:`audinterface.Process.segment` is ``None``
    the returned index
    will be of same type
    as the original one.
    Otherwise it will be a segmented index
    if any audio/video files are processed,
    or a filewise index otherwise
    """

    if preserve_index and segment_is_None:
        idx_type = "segmented" if audformat.is_segmented_index(idx) else "filewise"
        return idx_type

    extensions = [os.path.splitext(x)[-1] for x in idx.get_level_values(0).tolist()]
    # we only use wav in fixtures so this is ok
    any_media = any(["wav" in x for x in extensions])

    if any_media:
        idx_type = "segmented"
    else:
        idx_type = "filewise"

    return idx_type


def _series_generator(y, index_type: str):
    for idx, value in y.items():
        if index_type == "filewise":
            file = idx
            yield file, value
        elif index_type == "segmented":
           (file, _, _) = idx
           yield file, value
        else:
            raise ValueError("index type invalid")

@pytest.mark.parametrize("num_workers", [1, 2, None])
@pytest.mark.parametrize("file_format", ["json", "txt"]) # "json","txt"
@pytest.mark.parametrize("multiprocessing", [False, True])
@pytest.mark.parametrize("preserve_index", [False, True])
@pytest.mark.parametrize("process_func", [data_identity, None, identity])
def test_process_index(
    tmpdir,
    num_workers,
    file_format,
    multiprocessing,
    preserve_index,
    process_func,
):
    cache_root = os.path.join(tmpdir, "cache")

    process = audinterface.Process(
        process_func=process_func,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
        verbose=False,
    )

    if file_format == "json":
        data = {"a": 0}
    else:
        data = "abc"

    # Create file
    root = audeer.mkdir(tmpdir, "text")
    file = f"file.{file_format}"
    path = os.path.join(root, file)
    write_text_file(path, data)

    # Empty index
    index = audformat.filewise_index()
    y = process.process_index(index, preserve_index=preserve_index)
    assert y.empty

    # Segmented index with absolute paths
    index = audformat.segmented_index(
        [path] * 4,
        starts=[0, 0, 1, 2],
        ends=[None, 1, 2, 3],
    )

    y = process.process_index(
        index,
        preserve_index=preserve_index,
    )

    if preserve_index:
        pd.testing.assert_index_equal(y.index, index)

    expected_idx_type = _get_idx_type(preserve_index, process.segment is None, index)

    for path, value in _series_generator(y, expected_idx_type):
        assert audinterface.utils.read_text(path) == data
        assert value == data

    # for (path, _, _), value in y.items():
    #     assert audinterface.utils.read_text(path) == data
    #     assert value == data


    # # Segmented index with relative paths
    index = audformat.segmented_index(
        [file] * 4,
        starts=[0, 0, 1, 2],
        ends=[None, 1, 2, 3],
    )
    y = process.process_index(
        index,
        preserve_index=preserve_index,
        root=root,
    )
    if preserve_index:
        pd.testing.assert_index_equal(y.index, index)

    for file, value in _series_generator(y, expected_idx_type):
        assert audinterface.utils.read_text(file, root=root) == data
        assert value == data

    # for (file, _, _), value in y.items():
    #     assert audinterface.utils.read_text(file, root=root) == data
    #     assert value == data

    # Filewise index with absolute paths
    index = audformat.filewise_index(path)
    y = process.process_index(
        index,
        preserve_index=preserve_index,
    )

    if preserve_index:
        pd.testing.assert_index_equal(y.index, index)
        # for path, value in y.items():
        #     assert audinterface.utils.read_text(path) == data
        #     assert value == data
        expected_idx_type = _get_idx_type(preserve_index, process.segment is None, index)
        for path, value in _series_generator(y, expected_idx_type):
            assert audinterface.utils.read_text(path) == data
            assert value == data
    else:
        expected_idx_type = _get_idx_type(preserve_index, process.segment is None, index)
        expected_index = audformat.filewise_index(files=list(index))
        pd.testing.assert_index_equal(y.index, expected_index)
        # for (path, _, _), value in y.items():
        #     assert audinterface.utils.read_text(path) == data
        #     assert value == data
        # expected_idx_type = _get_idx_type(preserve_index, process.segment is None, index)
        for path, value in _series_generator(y, "filewise"):
            assert audinterface.utils.read_text(path) == data
            assert value == data

    # Filewise index with relative paths
    index = audformat.filewise_index(file)
    y = process.process_index(
        index,
        preserve_index=preserve_index,
        root=root,
    )
    if preserve_index:
        pd.testing.assert_index_equal(y.index, index)
        for file, value in _series_generator(y, "filewise"):
            assert audinterface.utils.read_text(file, root=root) == data
            assert value == data
        # for file, value in y.items():
        #     assert audinterface.utils.read_text(file, root=root) == data
        #     assert value == data
    else:
        for file, value in _series_generator(y, "filewise"):
            assert audinterface.utils.read_text(file, root=root) == data
            assert value == data
        # for (file, _, _), value in y.items():
        #     assert audinterface.utils.read_text(file, root=root) == data
        #     assert value == data

    # Cache result
    y = process.process_index(
        index,
        preserve_index=preserve_index,
        root=root,
        cache_root=cache_root,
    )

    # breakpoint()

    os.remove(path)
    # Fails because second file does not exist
    with pytest.raises(FileNotFoundError):
        process.process_index(
            index,
            preserve_index=preserve_index,
            root=root,
        )

    # Loading from cache still works
    y_cached = process.process_index(
        index,
        preserve_index=preserve_index,
        root=root,
        cache_root=cache_root,
    )
    pd.testing.assert_series_equal(y, y_cached)


@pytest.mark.parametrize(
    "process_func, process_func_args, data, file, expected_signal",
    [
        (
            identity,
            None,
            "abc",
            None,
            "abc",
        )
    ],
)
def test_process_data(
    process_func,
    process_func_args,
    data,
    file,
    expected_signal,
):
    process = audinterface.Process(
        process_func=process_func,
        process_func_args=process_func_args,
        verbose=False,
    )
    x = process.process_signal(data, file=file)

    if file is None:
        y = pd.Series([expected_signal])
    else:
        y = pd.Series(
            [expected_signal],
            index=audformat.filewise_index(file),
        )
    pd.testing.assert_series_equal(x, y)


# def test_process_with_special_args(tmpdir):
#     duration = 3
#     sampling_rate = 1
#     signal = np.zeros((2, duration), np.float32)
#     num_files = 10
#     win_dur = 1
#     num_frames = duration // win_dur
#     num_workers = 3
#
#     # create files
#     root = tmpdir
#     files = [f"f{idx}.wav" for idx in range(num_files)]
#     index = audformat.segmented_index(
#         np.repeat(files, num_frames),
#         np.tile(range(num_frames), num_files),
#         np.tile(range(1, num_frames + 1), num_files),
#     )
#     for file in files:
#         path = os.path.join(root, file)
#         audiofile.write(path, signal, sampling_rate, bit_depth=32)
#
#     # create interface
#     def process_func(signal, sampling_rate, idx, file, root):
#         return (idx, file, root)
#
#     process = audinterface.Process(
#         process_func=process_func,
#         num_workers=num_workers,
#     )
#
#     # process signal
#     y = process.process_signal(signal, sampling_rate)
#     expected = pd.Series(
#         [(0, None, None)],
#         audinterface.utils.signal_index(0, duration),
#     )
#     pd.testing.assert_series_equal(y, expected)
#
#     # process signal from index
#     y = process.process_signal_from_index(
#         signal,
#         sampling_rate,
#         expected.index,
#     )
#     pd.testing.assert_series_equal(y, expected)
#
#     # process file
#     y = process.process_file(files[0], root=root)
#     expected = pd.Series(
#         [(0, files[0], root)],
#         audformat.segmented_index(files[0], 0, duration),
#     )
#     pd.testing.assert_series_equal(y, expected)
#
#     # process files
#     y = process.process_files(files, root=root)
#     expected = pd.Series(
#         [(idx, files[idx], root) for idx in range(num_files)],
#         audformat.segmented_index(
#             files,
#             [0] * num_files,
#             [duration] * num_files,
#         ),
#     )
#     pd.testing.assert_series_equal(y, expected)
#
#     # process index with a filewise index
#     y = process.process_index(
#         audformat.filewise_index(files),
#         root=root,
#     )
#     pd.testing.assert_series_equal(y, expected)
#
#     # process index with a segmented index
#     y = process.process_index(index, root=root)
#     expected = pd.Series(
#         [(idx, file, root) for idx, (file, _, _) in enumerate(index)],
#         index,
#     )
#     pd.testing.assert_series_equal(y, expected)
#
#     # sliding window
#     # frames belonging to the same files have same idx
#     process = audinterface.Process(
#         process_func=process_func,
#         win_dur=win_dur,
#         hop_dur=win_dur,
#         num_workers=num_workers,
#     )
#     y = process.process_files(files, root=root)
#     values = []
#     for idx in range(num_files):
#         file = files[idx]
#         for _ in range(num_frames):
#             values.append((idx, file, root))
#     expected = pd.Series(values, index)
#     pd.testing.assert_series_equal(y, expected)
#
#     # mono processing function
#     # returns
#     # [((0, files[0], root), (0, files[0], root)),
#     #  ((1, files[1], root), (1, files[1], root)),
#     #  ... ]
#     process = audinterface.Process(
#         process_func=process_func,
#         process_func_is_mono=True,
#         num_workers=num_workers,
#     )
#     y = process.process_index(index, root=root)
#     expected = pd.Series(
#         [
#             ((idx, file, root), (idx, file, root))
#             for idx, (file, _, _) in enumerate(index)
#         ],
#         index,
#     )
#     pd.testing.assert_series_equal(y, expected)
#
#     # explicitly pass special arguments
#
#     process = audinterface.Process(
#         process_func=process_func,
#         process_func_args={"idx": 99, "file": "my/file", "root": None},
#         num_workers=num_workers,
#     )
#     y = process.process_index(index, root=root)
#     expected = pd.Series([(99, "my/file", None)] * len(index), index)
#     pd.testing.assert_series_equal(y, expected)


@pytest.mark.parametrize("data", ["abc"])
def test_read_data(tmpdir, data):
    file = audeer.path(tmpdir, "media.txt")
    with open(file, "w") as fp:
        fp.write(data)
    assert audinterface.utils.read_text(file) == data
