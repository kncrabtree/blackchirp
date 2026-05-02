.. index::
   single: Ft
   single: magnitude spectrum
   single: Fourier transform; result type
   single: frequency axis; FT

Ft
==

``Ft`` is the implicitly shared value type that holds a Fourier-transform magnitude spectrum
produced by :cpp:class:`FtWorker`. Like :cpp:class:`Fid`, it uses ``QSharedDataPointer``
so that copies are cheap and a deep copy occurs only when a mutating call is made while
the data is shared.

An ``Ft`` stores a ``QVector<double>`` of uniformly-spaced magnitude values together with the
frequency-axis metadata needed to reconstruct absolute frequencies: the starting frequency
``x0`` (MHz), the bin spacing (MHz), and the local-oscillator frequency (MHz).

The absolute frequency of bin ``i`` is ``xFirst() + i * xSpacing()``. The spectrum always
runs from low to high frequency in memory, regardless of which sideband the source ``Fid``
used; :cpp:class:`FtWorker` reorders the bins during FFT output so that ``xFirst()`` returns
the lowest spectral frequency. The ``trim(fmin, fmax)`` method discards out-of-range bins
and updates ``x0`` accordingly, which is used by the sideband co-averaging path to extract
the relevant portion of each per-LO-step spectrum before stitching.

``Ft`` objects are passed as the first argument of the ``FtWorker::ftDone()`` signal and
returned directly from :cpp:func:`FtWorker::doFT`. The FTMW data-viewing widget
(``FtmwViewWidget``) receives them, updates the frequency-domain plot curves, and passes them
to the peak-finder and overlay system. The user-facing controls that govern FID preprocessing
and spectrum display are described in :doc:`/user_guide/ftmw_configuration`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: Ft
   :members:
   :undoc-members:
