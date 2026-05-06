"""Python classes for loading and processing Blackchirp data.

This module is provides an easy means of loading Blackchirp data
from disk and performing most of the same data processing tasks that are available
in the Blackchirp program itself. Its goal is to rely on only a minimal set
of external Python dependencies (numpy, scipy, and pandas).

The Blackchirp python module is available from the `Python Package Index <https://pypi.org/project/blackchirp/>`_ and can be installed with::

    pip install blackchirp

The ``blackchirp`` module makes 5 classes available:

* ``BCExperiment``: Loads the contents of an experiment folder.
* ``BCFTMW``: Contains CP-FTMW data and processing functions that work with multiple FIDs.
* ``BCFid``: Contains data for a (possibly multi-frame) FID and its Fourier transform.
* ``BCLIF``: Contains LIF scan data and processing functions.
* ``BCLifTrace``: Contains a single LIF scan-point trace and its smoothing / integration helpers.

It is recommended to import these classes using::

    from blackchirp import *

This import statement will bring all 5 classes into the main namespace. Alternatively,
the statement ``import blackchirp as bc`` may be used, which requires prefixing all class names with ``bc.``.

Example:
    To quickly load an experiment and compute the Fourier transform for a single FID::

        from blackchirp import *
        from matplotlib import pyplot as plt

        exp = BCExperiment('path/to/experiment')
        x,y = exp.ftmw.get_fid().ft()

        fig,ax = plt.subplots()
        ax.plot(x,y.flatten())

    To load a LIF scan, fetch a single ``(laser, delay)`` trace, and plot
    its smoothed waveform::

        from blackchirp import *
        from matplotlib import pyplot as plt

        exp = BCExperiment('path/to/lif/experiment')
        trace = exp.lif.get_trace(l_index=0, d_index=0)
        x = trace.x(units='ns')

        fig,ax = plt.subplots()
        ax.plot(x, trace.smooth())

    Aggregating helpers on ``BCLIF`` (``delay_slice``, ``laser_slice``,
    ``image``) integrate every present scan point with a single
    processing-override surface; missing scan points are reported as
    ``np.nan`` (or any value passed via ``fill=``).

More detailed examples can be found on the individual class pages.
"""

from .bcfid import BCFid
from .bclif import BCLIF, BCLifTrace
from .bcftmw import BCFTMW
from .blackchirpexperiment import BCExperiment
