from doctest import NORMALIZE_WHITESPACE

from sybil import Sybil
from sybil.parsers.rest import DocTestParser


pytest_collect_file = Sybil(
    parsers=[DocTestParser(optionflags=NORMALIZE_WHITESPACE)],
    patterns=["*.py"],
).pytest()
