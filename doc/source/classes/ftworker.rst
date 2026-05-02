.. index::
   single: FtWorker
   single: FFT; FtWorker
   single: FidProcessingSettings
   single: FtWindowFunction
   single: FtUnits
   single: DeconvolutionMethod
   single: SidebandProcessingData
   single: FilterResult
   single: LO scan; sideband co-averaging

FtWorker
========

``FtWorker`` is the analysis-pipeline class that converts :cpp:class:`Fid`
objects into :cpp:class:`Ft` magnitude spectra. Three slot methods cover
the three usage patterns: ``doFT()`` for a single FT, ``doFtDiff()`` for
the difference between two spectra, and ``processSideband()`` for the
incremental LO-scan stitching that builds a broadband spectrum from a
series of per-LO-step FTs. Each slot is invoked through
``QtConcurrent::run`` rather than via a dedicated ``QThread``; results
are returned both by signal (``ftDone``, ``ftDiffDone``,
``sidebandDone``) and as the slot's return value, so the same instance
serves both the asynchronous UI path (``FtmwViewWidget``) and the
synchronous batch path (``BatchManager``).

The transform itself uses the GSL mixed-radix real FFT, which is most
efficient when the FID length factors into powers of 2, 3, and 5.
Preprocessing settings, window functions, output magnitude units, and
the LO-scan deconvolution method are documented at the member level
below. The frequency-domain plot, peak finder, and overlay system in
both blackchirp and the blackchirp-viewer consume the resulting
:cpp:class:`Ft` via ``ExperimentViewWidget``; the user-facing controls
that drive ``FidProcessingSettings`` are described in
:doc:`/user_guide/data_storage` and the FT-domain UI documentation.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: FtWorker
   :members:
   :protected-members:
   :undoc-members:
