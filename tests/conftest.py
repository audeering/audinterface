import numpy as np
import pytest

import audiofile
import audmath


@pytest.fixture(scope="module")
def audio(tmpdir_factory, request):
    """Fixture to generate audio file.

    Provide ``(duration, sampling_rate)``
    as parameter to this fixture.

    """
    file = str(tmpdir_factory.mktemp("audio").join("file.wav"))
    duration, sampling_rate = request.param
    signal = np.zeros((1, audmath.samples(duration, sampling_rate)))
    audiofile.write(file, signal, sampling_rate)

    yield file, signal, sampling_rate
