import typing

import pandas as pd


Timestamp = typing.Union[
    float,
    int,
    str,
    pd.Timedelta,
]

Timestamps = typing.Union[
    Timestamp,
    typing.Sequence[Timestamp],
]
