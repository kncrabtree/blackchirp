.. index::
   single: BCFTMW
   single: Python module; BCFTMW
   single: CP-FTMW; Python container
   single: sideband deconvolution; Python module

BCFTMW
======

``BCFTMW`` is the CP-FTMW container exposed by
:class:`~blackchirp.BCExperiment` as ``exp.ftmw`` whenever the
experiment folder contains an ``fid/`` subdirectory. It loads
``fid/fidparams.csv`` and ``fid/processing.csv`` once at construction
and serves :class:`~blackchirp.BCFid` objects on demand through
``get_fid``; the per-FID waveform data is read lazily, so opening a
large experiment is cheap.

The ``ftmw_type`` attribute carries the ``FtmwType`` value from
``objectives.csv`` and gates the API surface that is meaningful for
each acquisition mode. ``Target_Shots``, ``Target_Duration``, and
``Forever`` acquisitions store their cumulative final FID as ``0.csv``
and intermediate backups as ``1.csv``, ``2.csv``, …;
``get_differential_fid`` exposes shots collected between two backup
points by subtracting their raw integer data. ``LO_Scan``, ``DR_Scan``,
and ``Peak_Up`` acquisitions store independent segments instead, so
``get_differential_fid`` is rejected for those modes and ``num_backups``
returns zero.

``process_sideband`` performs the same sideband deconvolution that
Blackchirp's GUI applies to LO-scan acquisitions. Frequency-shifted
copies of every segment's FT are interpolated onto a common global
grid, the chosen sideband(s) are co-averaged, and the result is
returned in the requested frequency units. The algorithm and the
upper / lower / both options are described in detail on the
:doc:`CP-FTMW user-guide page </user_guide/cp-ftmw>` under
"Sideband Deconvolution".

API Reference
-------------

.. autoclass:: blackchirp.BCFTMW
   :members:
