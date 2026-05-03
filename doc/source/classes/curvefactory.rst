.. index::
   single: CurveFactory
   single: CurveStorageInterface
   single: SettingsStorageWrapper
   single: OverlayMetadataStorage
   single: curve; storage backend
   single: curve; factory construction

CurveFactory and CurveStorageInterface
======================================

``CurveFactory`` is the construction entry point for all
:cpp:class:`BlackchirpPlotCurveBase` subclasses. Rather than constructing
their own storage, curve objects receive a
``std::unique_ptr<CurveStorageInterface>`` at construction time. The factory
chooses the concrete implementation — either ``SettingsStorageWrapper`` (for
standard plot curves) or ``OverlayMetadataStorage`` (for overlay curves) —
and the curve itself needs no conditional logic to support both persistence
paths.

Storage backend abstraction
---------------------------

``CurveStorageInterface`` is the polymorphic contract. It declares two pure
virtual methods — ``set(key, QVariant)`` and ``get(key, QVariant)`` — plus
type-safe template overloads that wrap them. Any class that should store
curve display settings in a different backend (e.g., a database or a network
resource) can implement this interface and pass the result to any curve
constructor.

Two concrete backends ship with the codebase:

- **SettingsStorageWrapper** — adapts :cpp:class:`SettingsStorage` (QSettings)
  to the ``CurveStorageInterface`` contract. Curves that use this backend
  persist their appearance (color, line style, thickness, marker, etc.)
  across application restarts. Standard plot curves — FID traces, FT spectra,
  auxiliary data series — all use this backend.

- **OverlayMetadataStorage** — routes all reads and writes into the metadata
  map of a :cpp:class:`OverlayBase` instance. Overlay curves use this backend
  so their appearance is saved as part of the overlay data file rather than
  globally in QSettings. This means that loading an overlay restores its
  visual appearance without touching the user's global curve settings.

``OverlayMetadataStorage`` exposes ``getOverlay()`` so that
``BlackchirpPlotCurveBase`` can retrieve the associated overlay when it needs
to identify which overlay owns the curve.

Factory methods
---------------

``CurveFactory`` provides two static templated factory methods:

- ``createStandardCurve<CurveType>(key, type, ...)`` — constructs a
  ``SettingsStorageWrapper`` and passes it to the new curve instance.
  The ``key`` parameter matches the storage key used by
  :cpp:class:`SettingsStorage`.

- ``createOverlayCurve<CurveType>(key, overlay, ...)`` — constructs an
  ``OverlayMetadataStorage`` pointing at ``overlay`` and passes it to the
  new curve instance.

Both methods accept the same optional visual defaults (line style, symbol
style, curve style) that are written to storage on first construction and
then overridden by any saved settings on subsequent construction.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: CurveStorageInterface
   :members:
   :undoc-members:

.. doxygenclass:: SettingsStorageWrapper
   :members:
   :undoc-members:

.. doxygenclass:: OverlayMetadataStorage
   :members:
   :undoc-members:

.. doxygenclass:: CurveFactory
   :members:
   :undoc-members:
