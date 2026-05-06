.. index::
   single: BCFid
   single: Python module; BCFid
   single: FID; Python container
   single: Fourier transform; Python module

BCFid
=====

``BCFid`` is the single-FID container served by
:meth:`BCFTMW.get_fid <blackchirp.BCFTMW.get_fid>` and
:meth:`BCFTMW.get_differential_fid <blackchirp.BCFTMW.get_differential_fid>`.
On construction it reads one ``fid/N.csv`` file, decodes the
base-36-packed accumulated samples into per-shot voltages using the
matching ``fidparams.csv`` row (``vmult / shots``), and stores the
result as a 2-D numpy array. The first axis is time; the second axis is
the frame index. A single-frame acquisition still has shape
``(size, 1)`` so downstream code can index uniformly.

The ``ft`` method computes the Fourier transform of every frame using
the default settings drawn from ``fid/processing.csv`` (start / end
window in microseconds, window function, exponential filter,
zero-padding factor, FT units). Any of those settings may be overridden
per call via keyword argument; ``frame=N`` restricts the FT to a single
frame, and ``freq_units=`` rescales the returned frequency axis to Hz,
kHz, MHz (default), GHz, or THz. ``apply_lo`` and ``is_lower_sideband``
expose the sideband-aware mapping from scope frequency to molecular
frequency that ``ft`` and the BCFTMW deconvolution path both use
internally.

The default-vs-override pattern is the same one used by Blackchirp's
GUI FID-processing menu: each named ``ft`` argument left as ``None``
falls back to the value in ``processing.csv``. Window-function and
``FtUnits`` settings are accepted as either canonical name strings
(``BlackmanHarris``, ``FtuV``, …) or the historical integer enum
values, so the loader works against acquisitions captured before and
after the v2 enum-string migration.

For combining FIDs across multiple Blackchirp experiments, see the
two module-level helpers documented in :doc:`/python/coaverage`.

API Reference
-------------

.. autoclass:: blackchirp.BCFid
   :members:
