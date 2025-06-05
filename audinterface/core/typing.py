from collections.abc import Sequence
from typing import Union

import pandas as pd


Timestamp = Union[float, int, str, pd.Timedelta]

Timestamps = Union[Timestamp, Sequence[Timestamp]]
