Python Module
=============

The companion ``blackchirp`` Python package loads experiment folders
from disk and reproduces the bulk of Blackchirp's data-processing
pipeline (FID Fourier transforms, sideband deconvolution, LIF gate
integration) without depending on any of Blackchirp's C++ runtime. It
is published on the `Python Package Index
<https://pypi.org/project/blackchirp/>`_ and installs with::

    pip install blackchirp

The package depends only on numpy, scipy, and pandas; matplotlib and
other plotting libraries are intentionally excluded so that downstream
analysis pipelines can pick their own visualization stack.

Five top-level classes are exported from the package root —
:class:`~blackchirp.BCExperiment`, :class:`~blackchirp.BCFTMW`,
:class:`~blackchirp.BCFid`, :class:`~blackchirp.BCLIF`, and
:class:`~blackchirp.BCLifTrace` — together with two module-level
helpers, :func:`~blackchirp.coaverage_fids` and
:func:`~blackchirp.coaverage_spectra`, for combining FIDs across
separate Blackchirp experiments. The recommended import style is::

    from blackchirp import *

which brings all of them into the current namespace. The example
notebooks under :doc:`python/example` walk through end-to-end
CP-FTMW and LIF analysis sessions.

.. toctree::
   :caption: Examples

   python/example

.. toctree::
   :caption: Class Documentation
   :titlesonly:

   python/bcexperiment
   python/bcftmw
   python/bcfid
   python/bclif
   python/bcliftrace

.. toctree::
   :caption: Module-level Helpers
   :titlesonly:

   python/coaverage
