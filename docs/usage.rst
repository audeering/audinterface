.. Limit precision of pandas output for doctests
.. invisible-code-block: python

    import pandas as pd
    pd.set_option("display.precision", 4)


Usage
=====

The basic idea of :mod:`audinterface` is
to provide easy and standardized interfaces
to apply a machine learning model,
or other digital signal processing algorithms
to audio files.
The only prerequisite is
the algorithm provides a callable
that takes at least the signal
as a :class:`numpy.ndarray`
and the sampling rate as input.

The interface can then apply the algorithm
on a list of files,
a folder,
or an index conform to the audformat_ database specification.
Results are always returned containing a `segmented index`_.
In the following we load three files from the emodb_ database
and define a list of files,
a folder,
and an index.

.. code-block:: python

    import audb
    import os

    media = [
        "wav/03a01Fa.wav",
        "wav/03a01Nc.wav",
        "wav/16b10Wb.wav",
    ]
    db = audb.load(
        "emodb",
        version="1.3.0",
        media=media,
        full_path=False,
        verbose=False,
    )
    files = list(db.files)
    folder = os.path.join(db.root, os.path.dirname(files[0]))
    index = db["emotion"].index


Processing interface
--------------------

Let's assume we want to calculate the root mean square (RMS)
value in dB.
We first define the function
and create an interface for it using :class:`audinterface.Process`.

.. code-block:: python

    import audinterface
    import numpy as np

    def rms(signal, sampling_rate):
        return 20 * np.log10(np.sqrt(np.mean(signal ** 2)))

    interface = audinterface.Process(process_func=rms)

You can then use one of the
``process_files()``,
``process_folder()``,
or ``process_index()`` methods
to apply the algorithm
and return the result
as a :class:`pandas.Series`.

>>> interface.process_index(index, root=db.root)
file             start   end
wav/03a01Fa.wav  0 days  0 days 00:00:01.898250      -21.6901
wav/03a01Nc.wav  0 days  0 days 00:00:01.611250      -18.0407
wav/16b10Wb.wav  0 days  0 days 00:00:02.522499999   -20.3945
dtype: float32

To calculate RMS with a sliding window,
we create a new interface
and set a window and hop duration.

>>> interface = audinterface.Process(process_func=rms, win_dur=1.0, hop_dur=0.5)
>>> interface.process_files(files, root=db.root)
file             start                   end
wav/03a01Fa.wav  0 days 00:00:00         0 days 00:00:01          -20.1652
                 0 days 00:00:00.500000  0 days 00:00:01.500000   -23.4730
wav/03a01Nc.wav  0 days 00:00:00         0 days 00:00:01          -16.3866
                 0 days 00:00:00.500000  0 days 00:00:01.500000   -19.5026
wav/16b10Wb.wav  0 days 00:00:00         0 days 00:00:01          -21.7340
                 0 days 00:00:00.500000  0 days 00:00:01.500000   -20.2331
                 0 days 00:00:01         0 days 00:00:02          -18.8565
                 0 days 00:00:01.500000  0 days 00:00:02.500000   -20.4036
dtype: float32


Feature interface
-----------------

When the result of the processing function has multiple dimensions
it is recommended to use :class:`audinterface.Feature`,
which returns a :class:`pandas.DataFrame`
and assigns names to the dimensions/features.

.. code-block:: python

    def features(signal, sampling_rate):
        return [signal.mean(), signal.std()]

    interface = audinterface.Feature(
        ["mean", "std"],
        process_func=features,
    )

>>> interface.process_index(index, root=db.root)
                                                    mean     std
file            start  end
wav/03a01Fa.wav 0 days 0 days 00:00:01.898250    -0.0003  0.0823
wav/03a01Nc.wav 0 days 0 days 00:00:01.611250    -0.0003  0.1253
wav/16b10Wb.wav 0 days 0 days 00:00:02.522499999 -0.0005  0.0956

To calculate features with a sliding window,
we create a new interface
and set a window and hop duration.
By setting
``process_func_applies_sliding_window=False``
the windowing is automatically handled
and single frames are passed on to the processing function.

.. code-block:: python

    interface = audinterface.Feature(
        ["mean", "std"],
        process_func=features,
        process_func_applies_sliding_window=False,
        win_dur=1.0,
        hop_dur=0.5,
    )

>>> interface.process_files(files, root=db.root)
                                                                     mean     std
file            start                  end                                       
wav/03a01Fa.wav 0 days 00:00:00        0 days 00:00:01        -3.2866e-04  0.0981
                0 days 00:00:00.500000 0 days 00:00:01.500000 -2.8513e-04  0.0670
wav/03a01Nc.wav 0 days 00:00:00        0 days 00:00:01         3.8935e-05  0.1516
                0 days 00:00:00.500000 0 days 00:00:01.500000 -4.1219e-04  0.1059
wav/16b10Wb.wav 0 days 00:00:00        0 days 00:00:01        -4.5467e-04  0.0819
                0 days 00:00:00.500000 0 days 00:00:01.500000 -4.6149e-04  0.0974
                0 days 00:00:01        0 days 00:00:02        -4.6923e-04  0.1141
                0 days 00:00:01.500000 0 days 00:00:02.500000 -4.4670e-04  0.0955


Feature interface for multi-channel input
-----------------------------------------

By default,
an interface will process
the first channel of an audio signal.
We can prove this
by running the previous interface
on the following multi-channel signal.

.. code-block:: python

    import audiofile

    signal, sampling_rate = audiofile.read(
        os.path.join(db.root, files[0]),
        always_2d=True,
    )
    signal_multi_channel = np.concatenate(
        [
            signal,
            signal * 0,
            signal - 0.5,
            signal + 0.5,
        ],
    )

>>> signal_multi_channel.shape
(4, 30372)
>>> interface.process_signal(signal_multi_channel, sampling_rate)
                                                 mean     std
start                  end                                       
0 days 00:00:00        0 days 00:00:01        -0.0003  0.0981
0 days 00:00:00.500000 0 days 00:00:01.500000 -0.0003  0.0670

To process the second and fourth channel,
we create a new interface
and set
``channels=[1, 3]``.
To reuse our processing function,
we additionally set
``process_func_is_mono=True``.
This will apply the function
on each channel and combine the results.
Otherwise,
the processing function must
return an array with the correct
number of channels (here 2).

.. code-block:: python

    interface_multi_channel = audinterface.Feature(
        ["mean", "std"],
        process_func=features,
        process_func_is_mono=True,
        process_func_applies_sliding_window=False,
        win_dur=1.0,
        hop_dur=0.5,
        channels=[1, 3],
    )

    df = interface_multi_channel.process_signal(signal_multi_channel, sampling_rate)

>>> df
                                                 1            3          
                                              mean  std    mean     std
start                  end                                                 
0 days 00:00:00        0 days 00:00:01         0.0  0.0  0.4997  0.0981
0 days 00:00:00.500000 0 days 00:00:01.500000  0.0  0.0  0.4997  0.0670

We can access the features of a specific
channel by its index.

>>> df[3]
                                                 mean     std
start                  end                                       
0 days 00:00:00        0 days 00:00:01         0.4997  0.0981
0 days 00:00:00.500000 0 days 00:00:01.500000  0.4997  0.0670


Feature interface for external function
---------------------------------------

If we interface a function from an external library
that already applies a sliding window,
we again specify the
``win_dur`` and ``hop_dur``
arguments.
However,
by setting
``process_func_applies_sliding_window=True``
we still request that the whole signal is passed on.
Now,
the processing function is responsible
for extracting the features in a framewise manner
and returning the values in the correct shape,
namely ``(num_channels, num_features, num_frames)``,
whereas the first dimension is optionally.

.. code-block:: python

    import librosa

    def features(signal, sampling_rate, win_dur, hop_dur, n_mfcc):
        hop_length = int(hop_dur * sampling_rate)
        win_length = int(win_dur * sampling_rate)
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sampling_rate,
            n_mfcc=13,
            hop_length=hop_length,
            win_length=win_length,
        )
        return mfcc

    n_mfcc = 13
    interface = audinterface.Feature(
        [f"mfcc-{idx}" for idx in range(n_mfcc)],
        process_func=features,
        process_func_args={"n_mfcc": n_mfcc},  # "win_dur" and "hop_dur" can be omitted
        process_func_applies_sliding_window=True,
        win_dur=0.02,
        hop_dur=0.01,
    )

>>> interface.process_index(index, root=db.root)
                                                                 mfcc-0  ...  mfcc-12
file            start                  end                               ...         
wav/03a01Fa.wav 0 days 00:00:00        0 days 00:00:00.020000 -611.9933  ...   1.1514
                0 days 00:00:00.010000 0 days 00:00:00.030000 -668.1758  ...  14.0685
                0 days 00:00:00.020000 0 days 00:00:00.040000 -664.6128  ...   7.9498
                0 days 00:00:00.030000 0 days 00:00:00.050000 -667.7147  ...  12.9575
                0 days 00:00:00.040000 0 days 00:00:00.060000 -669.3674  ...   4.3968
...                                                                 ...  ...      ...
wav/16b10Wb.wav 0 days 00:00:02.480000 0 days 00:00:02.500000 -664.6736  ...   1.8637
                0 days 00:00:02.490000 0 days 00:00:02.510000 -658.9581  ...   9.3450
                0 days 00:00:02.500000 0 days 00:00:02.520000 -644.1565  ...   7.4110
                0 days 00:00:02.510000 0 days 00:00:02.530000 -618.5459  ...  17.6454
                0 days 00:00:02.520000 0 days 00:00:02.540000 -666.8052  ...   3.7111
<BLANKLINE>
[605 rows x 13 columns]


Serializable feature interface
------------------------------

To use a feature extractor as an input transform
of a machine learning model
it is recommend to provide it in a serializable way
so it can be `stored as part of the model`_.
One example of such a feature extractor is :class:`opensmile.Smile`.

To create such a feature extractor,
we create a class that inherits
from :class:`audinterface.Feature`
and :class:`audobject.Object`.

.. code-block:: python

    import audobject

    class MeanStd(audinterface.Feature, audobject.Object):

        def __init__(self):
            super().__init__(
                ["mean", "std"],
                process_func=self.features,
            )

        def features(self, signal, sampling_rate):
            return [signal.mean(), signal.std()]

    fex = MeanStd()

>>> fex.process_index(index, root=db.root)
                                                    mean     std
file            start  end                                          
wav/03a01Fa.wav 0 days 0 days 00:00:01.898250    -0.0003  0.0823
wav/03a01Nc.wav 0 days 0 days 00:00:01.611250    -0.0003  0.1253
wav/16b10Wb.wav 0 days 0 days 00:00:02.522499999 -0.0005  0.0956

The advantage of the feature extraction object is
that we can save it to a YAML file
and re-instantiate it from there.

.. Specify version for storing and loading objects to YAML
.. invisible-code-block: python

    __builtins__["__version__"] = "1.0.0"
    __builtins__["MeanStd"] = MeanStd


>>> fex.to_yaml("mean-std.yaml")
>>> fex2 = audobject.from_yaml("mean-std.yaml")
>>> fex2.process_index(index, root=db.root)
                                                    mean     std
file            start  end                                          
wav/03a01Fa.wav 0 days 0 days 00:00:01.898250    -0.0003  0.0823
wav/03a01Nc.wav 0 days 0 days 00:00:01.611250    -0.0003  0.1253
wav/16b10Wb.wav 0 days 0 days 00:00:02.522499999 -0.0005  0.0956


Segmentation interface
----------------------

When the result of the processing function is an index
it is recommended to use :class:`audinterface.Segment`,
which returns a segmented index conform to audformat_.
An example for such a processing function
would be a voice activity detection algorithm.

.. code-block:: python

    import auditok
    import pandas as pd

    def segments(signal, sampling_rate):

        # Convert floating point array to 16bit PCM little-endian
        ints = (signal[0, :] * 32767).astype(np.int16)
        little_endian = ints.astype("<u2")
        signal = little_endian.tobytes()

        regions = auditok.split(
            signal,
            sampling_rate=sampling_rate,
            sample_width=2,
            channels=1,
            min_dur=0.2,
            energy_threshold=70,
        )
        index = pd.MultiIndex.from_tuples(
            [
                (
                    pd.Timedelta(region.start, unit="s"),
                    pd.Timedelta(region.end, unit="s"),
                )
                for region in regions
            ],
            names=["start", "end"],
        )
        return index

    interface = audinterface.Segment(process_func=segments)

>>> interface.process_file(files[0], root=db.root)
MultiIndex([('wav/03a01Fa.wav', '0 days 00:00:00.150000', ...),
            ('wav/03a01Fa.wav', '0 days 00:00:00.900000', ...)],
           names=['file', 'start', 'end'])

Sometimes,
it is required that a table
(i.e., :class:`pandas.Series` or :class`pandas.DataFrame`)
is segmented
and the ``labels`` from the original segments
should be kept.
For this,
:class:`audinterface.Segment` has a dedicated method
:meth:`process_table() <audinterface.Segment.process_table>`.
This method is useful,
if a segmentation
(e.g., voice activity detection)
is performed on an already labelled dataset
in order to do data augmentation
or teacher-student training.

>>> table = pd.DataFrame({"label": [n * 2 for n in range(len(index))]}, index=index)
>>> table
                 label
file                  
wav/03a01Fa.wav      0
wav/03a01Nc.wav      2
wav/16b10Wb.wav      4
>>> interface.process_table(table, root=db.root)
                                                               label
file            start                  end                          
wav/03a01Fa.wav 0 days 00:00:00.150000 0 days 00:00:00.700000      0
                0 days 00:00:00.900000 0 days 00:00:01.600000      0
wav/03a01Nc.wav 0 days 00:00:00.100000 0 days 00:00:01.350000      2
wav/16b10Wb.wav 0 days 00:00:00.300000 0 days 00:00:01             4
                0 days 00:00:01.050000 0 days 00:00:02.500000      4


Segmentation with feature interface
-----------------------------------

In some cases,
a processing function performs
both segmentation and feature extraction.
For this,
:class:`audinterface.SegmentWithFeature` can be used.
This interface returns a :class:`pd.DataFrame`
with a segmented index conform to audformat_.
An example of such a processing function
would be a speech recognition model
that also generates timestamps for its results.

.. code-block:: python

    from faster_whisper import WhisperModel
    import pandas as pd

    model_size = "tiny"
    model = WhisperModel(model_size, device="cpu")

    def word_transcripts(signal, sampling_rate):
        segments, _ = model.transcribe(
            signal[0], task="transcribe", word_timestamps=True
        )
        index = []
        words = []
        for segment in segments:
            for word in segment.words:
                index.append(
                    (
                        pd.to_timedelta(word.start, unit="s"),
                        pd.to_timedelta(word.end, unit="s")
                    )
                )
                words.append(word.word.strip())
        index = pd.MultiIndex.from_tuples(index, names=["start", "end"])
        return pd.Series(data=words, index=index)

    interface = audinterface.SegmentWithFeature(
        feature_names="word", process_func=word_transcripts
    )

>>> interface.process_file(files[0], root=db.root)
                                                                      word
file            start                  end
wav/03a01Fa.wav 0 days 00:00:00        0 days 00:00:00.360000          Der
                0 days 00:00:00.360000 0 days 00:00:00.720000       Lappen
                0 days 00:00:00.720000 0 days 00:00:00.880000        liegt
                0 days 00:00:00.880000 0 days 00:00:01.080000          auf
                0 days 00:00:01.080000 0 days 00:00:01.220000          dem
                0 days 00:00:01.220000 0 days 00:00:01.820000  Eisschrank.

Similarly to :class:`audinterface.Segment`,
:class:`audinterface.SegmentWithFeature`
also has a method
:meth:`process_table() <audinterface.SegmentWithFeature.process_table>`,
which can be applied to an already labelled dataset.

>>> interface.process_table(table.head(2), root=db.root)
                                                                      word  label
file            start                  end
wav/03a01Fa.wav 0 days 00:00:00        0 days 00:00:00.360000          Der      0
                0 days 00:00:00.360000 0 days 00:00:00.720000       Lappen      0
                0 days 00:00:00.720000 0 days 00:00:00.880000        liegt      0
                0 days 00:00:00.880000 0 days 00:00:01.080000          auf      0
                0 days 00:00:01.080000 0 days 00:00:01.220000          dem      0
                0 days 00:00:01.220000 0 days 00:00:01.820000  Eisschrank.      0
wav/03a01Nc.wav 0 days 00:00:00        0 days 00:00:00.240000          Der      2
                0 days 00:00:00.240000 0 days 00:00:00.520000       Lappen      2
                0 days 00:00:00.520000 0 days 00:00:00.660000        liegt      2
                0 days 00:00:00.660000 0 days 00:00:00.820000          auf      2
                0 days 00:00:00.820000 0 days 00:00:00.960000          dem      2
                0 days 00:00:00.960000 0 days 00:00:01.480000    Eiscrank.      2


Special processing function arguments
-------------------------------------

There are some special arguments
to the processing function,
which will be automatically set
if they are not specified in
``process_func_args``:

========  =============
argument  value
========  =============
idx       running index
file      file path
root      root folder
========  =============

The following processing function
returns the values of
``"idx"`` and ``"file"``.

.. code-block:: python

    def special_args(signal, sampling_rate, idx, file):
        return idx, os.path.basename(file)

    interface = audinterface.Process(process_func=special_args)

>>> interface.process_files(files, root=db.root)
file             start   end                      
wav/03a01Fa.wav  0 days  0 days 00:00:01.898250       (0, 03a01Fa.wav)
wav/03a01Nc.wav  0 days  0 days 00:00:01.611250       (1, 03a01Nc.wav)
wav/16b10Wb.wav  0 days  0 days 00:00:02.522499999    (2, 16b10Wb.wav)
dtype: object

For instance,
we can pass a list with gender labels
to the processing function
and use the running index
to select the appropriate f0 range.

.. code-block:: python

    gender = db["files"]["speaker"].get(map="gender")  # gender per file
    f0_range = {
        "female": [160, 300],  # [fmin, fmax]
        "male": [60, 180],
    }

    def f0(signal, sampling_rate, idx, gender, f0_range):
        # extract mean f0 using a gender adapted range
        y = librosa.yin(
            signal,
            fmin=f0_range[gender.iloc[idx]][0],
            fmax=f0_range[gender.iloc[idx]][1],
            sr=sampling_rate,
        ).mean().round(2)
        return y, gender.iloc[idx]

    interface = audinterface.Feature(
        ["f0", "gender"],
        process_func=f0,
        process_func_args={
            "gender": gender,
            "f0_range": f0_range,
        },
    )

>>> interface.process_index(gender.index, root=db.root)
                                                       f0  gender
file            start  end                                                  
wav/03a01Fa.wav 0 days 0 days 00:00:01.898250       134.0    male
wav/03a01Nc.wav 0 days 0 days 00:00:01.611250      113.16    male
wav/16b10Wb.wav 0 days 0 days 00:00:02.522499999   234.86  female


.. _audformat: https://audeering.github.io/audformat/
.. _emodb: http://emodb.bilderbar.info
.. _segmented index: https://audeering.github.io/audformat/data-tables.html#segmented
.. _stored as part of the model: https://audeering.github.io/audonnx/usage.html#export-model
