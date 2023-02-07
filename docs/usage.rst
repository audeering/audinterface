.. Specify pandas format output in cells
.. jupyter-execute::
    :hide-code:
    :hide-output:

    import pandas as pd


    def dataframe_to_html(df):
        # Replace beginning of data path with ...
        if len(df.index) > 0 and df.index.names[0] == 'file':
            old_path = r'.+/audb/emodb/1.3.0/d3b62a9b/wav/'
            new_path = r'.../'
            # Assuming segmented index
            df.index = df.index.set_levels(
                df.index.levels[0].str.replace(
                    old_path,
                    new_path,
                    regex=True,
                ),
                level=0,
            )

        return df.to_html(max_rows=6, max_cols=4)


    def series_to_html(y):
        df = y.to_frame()
        df.columns = ['']
        return dataframe_to_html(df)


    def index_to_html(index):
        df = pd.DataFrame(index=index)
        return dataframe_to_html(df)


    setattr(pd.Series, '_repr_html_', series_to_html)
    setattr(pd.Index, '_repr_html_', index_to_html)
    setattr(pd.DataFrame, '_repr_html_', dataframe_to_html)

.. Specify version for storing and loading objects to YAML
.. jupyter-execute::
    :hide-code:

    __version__ = '1.0.0'


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

.. jupyter-execute::

    import audb
    import os

    media = [
        'wav/03a01Fa.wav',
        'wav/03a01Nc.wav',
        'wav/16b10Wb.wav',
    ]
    db = audb.load(
        'emodb',
        version='1.3.0',
        media=media,
        verbose=False,
    )

    files = list(db.files)
    folder = os.path.dirname(files[0])
    index = db['emotion'].index


Processing interface
--------------------

Let's assume we want to calculate the root mean square (RMS)
value in dB.
We first define the function
and create an interface for it using :class:`audinterface.Process`.

.. jupyter-execute::

    import audinterface
    import numpy as np

    def rms(signal, sampling_rate):
        return 20 * np.log10(np.sqrt(np.mean(signal ** 2)))

    interface = audinterface.Process(process_func=rms)

The following three commands
apply the algorithm
and all return the same result
as a :class:`pandas.Series`.

.. jupyter-execute::

    y = interface.process_files(files)
    y = interface.process_folder(folder)
    y = interface.process_index(index)
    y

To calculate RMS with a sliding window,
we create a new interface
and set a window and hop duration.

.. jupyter-execute::

    interface = audinterface.Process(
        process_func=rms,
        win_dur=1.0,
        hop_dur=0.5,
    )
    y = interface.process_files(files)
    y

Feature interface
-----------------

When the result of the processing function has multiple dimensions
it is recommended to use :class:`audinterface.Feature`,
which returns a :class:`pandas.DataFrame`
and assigns names to the dimensions/features.

.. jupyter-execute::

    def features(signal, sampling_rate):
        return [signal.mean(), signal.std()]

    interface = audinterface.Feature(
        ['mean', 'std'],
        process_func=features,
    )

    df = interface.process_index(index)
    df

To calculate features with a sliding window,
we create a new interface
and set a window and hop duration.
By setting
``process_func_applies_sliding_window=False``
the windowing is automatically handeled
and single frames are passed on to the processing function.

.. jupyter-execute::

    interface = audinterface.Feature(
        ['mean', 'std'],
        process_func=features,
        process_func_applies_sliding_window=False,
        win_dur=1.0,
        hop_dur=0.5,
    )
    df = interface.process_files(files)
    df


Feature interface for multi-channel input
-----------------------------------------

By default,
an interface will process
the first channel of an audio signal.
We can prove this
by running the previous interface
on the following multi-channel signal.

.. jupyter-execute::

    import audiofile

    signal, sampling_rate = audiofile.read(
        files[0],
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
    signal_multi_channel.shape

.. jupyter-execute::

    df = interface.process_signal(
        signal_multi_channel,
        sampling_rate,
    )
    df

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

.. jupyter-execute::

    interface_multi_channel = audinterface.Feature(
        ['mean', 'std'],
        process_func=features,
        process_func_is_mono=True,
        process_func_applies_sliding_window=False,
        win_dur=1.0,
        hop_dur=0.5,
        channels=[1, 3],
    )

    df = interface_multi_channel.process_signal(
        signal_multi_channel,
        sampling_rate,
    )
    df

We can access the features of a specific
channel by its index.

.. jupyter-execute::

    df[3]


Feature interface for external function
---------------------------------------

If we interface a function from an external library
that already applies a sliding window,
we again specfiy the
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

.. jupyter-execute::

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
        [f'mfcc-{idx}' for idx in range(n_mfcc)],
        process_func=features,
        process_func_args={'n_mfcc': n_mfcc},  # 'win_dur' and 'hop_dur' can be omitted
        process_func_applies_sliding_window=True,
        win_dur=0.02,
        hop_dur=0.01,
    )
    df = interface.process_index(index)
    df


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

.. jupyter-execute::

    import audobject

    class MeanStd(audinterface.Feature, audobject.Object):

        def __init__(self):
            super().__init__(
                ['mean', 'std'],
                process_func=self.features,
            )

        def features(self, signal, sampling_rate):
            return [signal.mean(), signal.std()]

    fex = MeanStd()
    df = fex.process_index(index)
    df

The advantage of the feature extraction object is
that we can save it to a YAML file
and re-instantiate it from there.

.. jupyter-execute::

    fex.to_yaml('mean-std.yaml')
    fex2 = audobject.from_yaml('mean-std.yaml')
    df = fex2.process_index(index)
    df


Segmentation interface
----------------------

When the result of the processing function is an index
it is recommended to use :class:`audinterface.Segment`,
which returns a segmented index conform to audformat_.
An example for such a processing function
would be a voice activity detection algorithm.

.. jupyter-execute::

    import auditok

    def segments(signal, sampling_rate):

        # Convert floating point array to 16bit PCM little-endian
        ints = (signal[0, :] * 32767).astype(np.int16)
        little_endian = ints.astype('<u2')
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
                    pd.Timedelta(region.meta.start, unit='s'),
                    pd.Timedelta(region.meta.end, unit='s'),
                )
                for region in regions
            ],
            names=['start', 'end'],
        )
        return index

    interface = audinterface.Segment(process_func=segments)
    idx = interface.process_file(files[0])
    idx


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
``'idx'`` and ``'file'``.

.. jupyter-execute::

    def special_args(signal, sampling_rate, idx, file):
        return idx, os.path.basename(file)

    interface = audinterface.Process(process_func=special_args)
    y = interface.process_files(files)
    y

For instance,
we can pass a list with gender labels
to the processing function
and use the running index
to select the appropriate f0 range.

.. jupyter-execute::

    gender = db['files']['speaker'].get(map='gender')  # gender per file
    f0_range = {
        'female': [160, 300],  # [fmin, fmax]
        'male': [60, 180],
    }

    def f0(signal, sampling_rate, idx, gender, f0_range):
        # extract mean f0 using a gender adapted range
        y = librosa.yin(
            signal,
            fmin=f0_range[gender[idx]][0],
            fmax=f0_range[gender[idx]][1],
            sr=sampling_rate,
        ).mean()
        return y, gender[idx]

    interface = audinterface.Feature(
        ['f0', 'gender'],
        process_func=f0,
        process_func_args={
            'gender': gender,
            'f0_range': f0_range,
        },
    )
    df = interface.process_index(gender.index)
    df


.. _audformat: https://audeering.github.io/audformat/
.. _emodb: http://emodb.bilderbar.info
.. _segmented index: https://audeering.github.io/audformat/data-tables.html#segmented
.. _stored as part of the model: https://audeering.github.io/audonnx/usage.html#export-model
