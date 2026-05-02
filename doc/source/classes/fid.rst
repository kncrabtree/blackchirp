.. index::
   single: Fid
   single: FidData
   single: FidList
   single: free-induction decay
   single: co-averaging; FID accumulation

Fid
===

``Fid`` is the implicitly shared value type that carries a single
co-averaged free-induction decay through Blackchirp's acquisition and
analysis pipeline.

FTMW digitizer drivers (``FtmwScope`` subclasses) parse raw waveform
bytes into accumulated samples and stage them in a waveform buffer.
The acquisition layer drains that buffer into a ``FidList``
(``QVector<Fid>``) — one entry per LO step or sideband channel — and
hands it to :cpp:class:`FtWorker`, which produces an :cpp:class:`Ft`
magnitude spectrum. ``FtmwViewWidget`` and the blackchirp-viewer's
``ExperimentViewWidget`` share that path to drive the frequency-domain
plot, peak finder, and overlay system. The accumulation and storage
settings live on :doc:`ftmwconfig`; the RF and sideband parameters that
set ``probeFreq`` and ``sideband`` come from :doc:`rfconfig`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: Fid
   :members:
   :undoc-members:

.. doxygenclass:: FidData
   :members:
   :undoc-members:
