import pytest

import audinterface


def power(a: int = 0, *, b: int = 1):
    return a ** b


@pytest.mark.parametrize(
    'multiprocessing',
    [
        True, False,
    ]
)
@pytest.mark.parametrize(
    'num_workers',
    [
        1, 3, None,
    ]
)
@pytest.mark.parametrize(
    'task_fun, params',
    [
        (power, [([], {})]),
        (power, [([1], {})]),
        (power, [([1], {'b': 2})]),
        (power, [([], {'a': 1, 'b': 2})]),
        (power, [([x], {'b': x}) for x in range(5)]),
    ]
)
def test(multiprocessing, num_workers, task_fun, params):
    expected = [
        task_fun(*param[0], **param[1]) for param in params
    ]
    print(expected)
    results = audinterface.utils.run_tasks(
        task_fun,
        params,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
    )
    assert expected == results
