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

Multi-experiment coaveraging
----------------------------

For combining FIDs across separate Blackchirp experiments, two
module-level helpers are exported alongside the classes:
:func:`~blackchirp.coaverage_fids` returns a :class:`BCFid` whose raw
integer data is the sample-by-sample sum of every input and whose
shot count is the sum of input shot counts, with optional
cross-correlation phase correction against a chosen reference window;
:func:`~blackchirp.coaverage_spectra` returns a shot-weighted
magnitude-spectrum coaverage as ``(x, y)`` arrays. Both refuse on any
mismatch in ``spacing``, ``size``, ``sideband``, ``probefreq``,
``vmult``, or frame count — none of those can be combined meaningfully
without an explicit policy from the caller. The C++ acquisition path
does not have an analogous cross-experiment coaverage primitive, so
the Python module is the canonical home.

API Reference
-------------

.. autoclass:: blackchirp.BCFid
   :members:

.. autofunction:: blackchirp.coaverage_fids

.. autofunction:: blackchirp.coaverage_spectra
