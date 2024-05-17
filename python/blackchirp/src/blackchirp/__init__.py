"""Python classes for loading and processing Blackchirp data.

This module is provides an easy means of loading Blackchirp data
from disk and performing most of the same data processing tasks that are available
in the Blackchirp program itself. Its goal is to rely on only a minimal set
of external Python dependencies (numpy, scipy, and pandas).

The Blackchirp python module is available from the `Python Package Index <https://pypi.org/project/blackchirp/>`_ and can be installed with::

    pip install blackchirp

The ``blackchirp`` module makes 4 classes available:

* ``BCExperiment``: Loads the contents of an experiment folder.
* ``BCFTMW``: Contains CP-FTMW data and processing functions that work with multiple FIDs.
* ``BCFid``: Contains data for a (possibly multi-frame) FID and its Fourier transform.
* ``BCLif``: Contains LIF data and processing functions.

It is recommended to import these classes using::

    from blackchirp import *
    
This import statement will bring all 4 classes into the main namespace. Alternatively,
the statement ``import blackchirp as bc`` may be used, which requires prefixing all class names with ``bc.``.

Example:
    To quickly load an experiment and compute the Fourier transform for a single FID::
    
        from blackchirp import *
        from matplotlib import pyplot as plt
        
        exp = BCExperiment('path/to/experiment')
        x,y = exp.get_fid().ft()
        
        fig,ax = plt.subplots()
        ax.plot(x,y.flatten())
        
More detailed examples can be found on the individual class pages.
"""

from .bcfid import BCFid
from .bclif import BCLIF
from .bcftmw import BCFTMW
from .blackchirpexperiment import BCExperiment
