.. index::
   single: BCLifTrace
   single: Python module; BCLifTrace
   single: LIF; per-point trace
   single: integration; LIF gates

BCLifTrace
==========

``BCLifTrace`` is the single-point LIF container served by
:meth:`BCLIF.get_trace <blackchirp.BCLIF.get_trace>`. It loads one
``lif/N.csv`` file, decodes the base-36-packed accumulated samples for
the LIF and (optional) reference channels into per-shot voltages using
the matching ``lifparams.csv`` row, and stores the result as 1-D numpy
arrays. ``has_ref`` reports whether the trace file carried a reference
column; ``xy()`` returns ``(x, lif, ref)`` when it did and
``(x, lif)`` when it did not.

The ``smooth`` and ``integrate`` methods reproduce Blackchirp's GUI
display path bit-for-bit. ``smooth`` applies an IIR low-pass filter
followed by a Savitzky-Golay smoother, with both stages controlled by
``processing.csv`` (``LowPassAlpha``, ``SavGolEnabled``,
``SavGolWindow``, ``SavGolPoly``) and overridable per call. ``integrate``
runs the same filter chain, then takes a trapezoidal sum **in
sample-index space**: integration gates are sample indices, not times,
and ``dx`` is one sample. Without a reference channel the result is in
``V·sample`` (multiply by ``self.spacing`` to get V·s); with a
reference channel the result is the dimensionless ratio of LIF to
reference integrals.

The integration-gate semantics, the reference-channel ratio behaviour,
and the on-disk layout of the trace files are described on the
:doc:`LIF Data Storage </user_guide/lif/data_storage>` user-guide page.
The :class:`~blackchirp.BCLIF` aggregating helpers
(``delay_slice``, ``laser_slice``, ``image``) call ``integrate`` on every
present scan point with a single shared processing-override surface.

API Reference
-------------

.. autoclass:: blackchirp.BCLifTrace
   :members:
