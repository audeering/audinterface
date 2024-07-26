from sybil import Sybil
from sybil.parsers.rest import DocTestParser
from sybil.parsers.rest import PythonCodeBlockParser


pytest_collect_file = Sybil(
    parsers=[DocTestParser(), PythonCodeBlockParser()],
    patterns=["*.rst"],
).pytest()
