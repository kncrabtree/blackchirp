.. index::
   single: coaverage
   single: Python module; coaverage
   single: FID; multi-experiment coaverage

Coaverage
=========

For combining FIDs across separate Blackchirp experiments, the
``blackchirp`` package exports two module-level helpers alongside its
classes. They live at the package root because their inputs span more
than one :class:`~blackchirp.BCExperiment` and their outputs are not
methods on any single existing object.

:func:`~blackchirp.coaverage_fids` performs the coaverage in the time
domain. It returns a :class:`~blackchirp.BCFid` whose raw integer data
is the sample-by-sample sum of every input and whose shot count is the
sum of input shot counts; voltage data is recomputed from those sums
and the shared ``vmult``. Phase correction is optional: when both
``pc_start_us`` and ``pc_end_us`` are supplied each non-reference FID
is shifted along the time axis by the integer offset that maximises
the cross-correlation between its windowed data and that of the
reference FID. The reference defaults to the highest-shot input but
may be selected by index. A single shift per FID is applied to all
frames by default, on the assumption that the frames share a clock;
``per_frame_pc=True`` switches to one shift per frame.

:func:`~blackchirp.coaverage_spectra` performs a shot-weighted
coaverage of magnitude spectra. Each input FID is Fourier-transformed
with the same processing kwargs and the resulting magnitudes are
combined as :math:`y = \sum_i s_i\, |Y_i| / \sum_i s_i` where
:math:`s_i` is the shot count of FID :math:`i`. This avoids
time-domain alignment entirely but does not reduce the noise floor —
the Rayleigh-distributed noise mean is invariant under averaging, only
its fluctuation drops. Reach for this function when phase drift across
experiments defeats time-domain alignment; otherwise prefer
:func:`~blackchirp.coaverage_fids`.

Both entry points enforce strict compatibility between their inputs:
matching ``spacing``, ``size``, ``sideband``, ``probefreq``, ``vmult``,
and frame count. Float-valued fields are compared with exact equality;
mismatches raise ``ValueError`` rather than being silently coerced.
The C++ acquisition path does not have an analogous cross-experiment
coaverage primitive, so the Python module is the canonical home.

API Reference
-------------

.. autofunction:: blackchirp.coaverage_fids

.. autofunction:: blackchirp.coaverage_spectra
