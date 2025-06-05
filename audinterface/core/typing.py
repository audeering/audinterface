from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


Timestamp = float | int | str | pd.Timedelta

Timestamps = Timestamp | Sequence[Timestamp]
