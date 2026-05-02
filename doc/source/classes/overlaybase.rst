.. index::
   single: OverlayBase
   single: OverlayStorage
   single: overlays; data model
   single: BCExpOverlay
   single: CatalogOverlay
   single: GenericXYOverlay
   single: OverlayBase; OverlayType discriminator

OverlayBase and OverlayStorage
==============================

``OverlayBase`` is the abstract base class for every plot overlay in
Blackchirp.  Each overlay carries a display label, a source and destination
file path, a plot panel assignment, and a set of transformations (X/Y offset,
Y scale) applied on top of the raw data.  Optional frequency-range clips
restrict the visible frequency window independently of the transformation.
Visibility is controlled by ``setEnabled()``; the enabled state is
synchronized with the ``CurveAppearance`` metadata so that the plot widget
reflects the change without an additional signal.

The ``OverlayType`` enumerator is the type discriminator that identifies each
concrete subclass:

- ``BCExperiment`` — an FT spectrum read from a Blackchirp experiment file
  (``BCExpOverlay``).
- ``Catalog`` — a spectroscopic line catalog from SPCAT, XIAM, or a compatible
  program (``CatalogOverlay``).  Optional Lorentzian or Gaussian lineshape
  convolution produces a simulated absorption profile for direct comparison
  with a measured FT spectrum.
- ``GenericXY`` — an arbitrary two-column data file (CSV, TSV, etc.)
  (``GenericXYOverlay``).

The three concrete subclasses are defined in ``overlaytypes.h``; their member
documentation appears in the API Reference section below.  The user-facing
overlay workflow is described in :doc:`/user_guide/overlays`.

OverlayStorage
--------------

``OverlayStorage`` manages the persistent collection of overlays for a single
experiment.  It maintains two separate in-memory collections: *persistent*
overlays that are written to disk under ``<experimentPath>/overlays/``, and
*preview* overlays that exist only in memory.  Write operations are dispatched
asynchronously via ``QtConcurrent``; the signals
``overlayWriteCompleted()`` and ``overlayWriteFailed()`` report outcomes on
the object's own thread.  ``waitForPendingWrites()`` coordinates shutdown with
in-flight background operations.

Preview overlays bypass all disk I/O.  ``detachPreviewOverlay()`` converts a
preview overlay to a persistent one, scheduling the initial background write.

``OverlayStorage`` inherits ``DataStorageBase`` for compatibility with the
experiment data pipeline; only ``save()`` has a non-trivial implementation.
The other ``DataStorageBase`` methods (``advance``, ``start``, ``finish``) are
no-ops.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: OverlayBase
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: BCExpOverlay
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: CatalogOverlay
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: GenericXYOverlay
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: OverlayStorage
   :members:
   :undoc-members:
