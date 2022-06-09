.. Specify pandas format output in cells
.. jupyter-execute::
    :hide-code:
    :hide-output:

    import pandas as pd


    def series_to_html(self):
        df = self.to_frame()
        df.columns = ['']
        return df._repr_html_()


    setattr(pd.Series, '_repr_html_', series_to_html)
    pd.set_option('display.max_rows', 6)


Usage
=====

The basic idea of :mod:`audinterface` is
to provide easy and standardized interfaces
to apply a machine learning model,
or other digital signal processing algorithms (DSP)
to audio files.
The only prerequisite is
the algorithm provides a callable
that takes at least the signal
as a :class:`np.ndarray`
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

    media = ['wav/03a01Fa.wav', 'wav/03a01Nc.wav', 'wav/03a01Wa.wav']
    db = audb.load('emodb', version='1.2.0', media=media, verbose=False)

    files = list(db.files)
    folder = os.path.dirname(files[0])
    index = db['emotion'].index


DSP algorithm returning a series
--------------------------------

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
as a :class:`pd.Series`.

.. jupyter-execute::

    y = interface.process_files(files)
    y = interface.process_folder(folder)
    y = interface.process_index(index)
    y


DSP algorithm returning a dataframe
-----------------------------------

When using a DSP algorithm as feature extractor,
it is recommended to use :class:`audinterface.Feature`,
which returns results as a :class:`pd.DataFrame`
and assigns names to the single features.

.. jupyter-execute::

    def features(signal, sampling_rate):
        return [signal.mean(), signal.std()]

    interface = audinterface.Feature(
        ['mean', 'std'],
        process_func=features,
    )

    df = interface.process_index(index)
    df


Creating a feature extraction object
------------------------------------

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

    fex.to_yaml('mean-std.yml')
    fex2 = audobject.from_yaml('mean-std.yml')
    df = fex2.process_index(index)
    df

.. _audformat: https://audeering.github.io/audformat/
.. _emodb: http://emodb.bilderbar.info
.. _segmented index: https://audeering.github.io/audformat/data-tables.html#segmented
.. _stored as part of the model: https://audeering.github.io/audonnx/usage.html#export-model
