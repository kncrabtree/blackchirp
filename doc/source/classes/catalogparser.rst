.. index::
   single: CatalogParser
   single: parsers; spectroscopic catalogs
   single: CatalogData; parser interface
   single: SPCAT
   single: XIAM

CatalogParser
=============

``CatalogParser`` is the intermediate abstract base that every
spectroscopic catalog parser derives from. It extends
:cpp:class:`FileParser` with a single hook —
:cpp:func:`CatalogParser::parse`, which returns a
:cpp:class:`CatalogData` value containing the ordered transition list
(frequency, intensity, error, lower-state energy, quantum numbers) and
the source-program metadata that downstream code uses to label the
overlay. Splitting ``parse()`` onto an intermediate base lets the
catalog-specific consumers — ``CatalogOverlay``, the catalog-import
dialog, and the catalog overlay-replay path — work against this
interface alone, without coupling to either of the concrete formats.

The concrete catalog parsers shipped with Blackchirp are
:cpp:class:`SPCATParser` (Pickett SPFIT/SPCAT, ``.cat`` files) and
:cpp:class:`XIAMParser` (XIAM internal-rotation analysis, ``.xo`` and
``.out`` files). A new format with the same conceptual shape — a list of
transitions with frequency, intensity, and quantum-number assignments —
is added by deriving from this class, implementing the
:cpp:class:`FileParser` hooks plus ``parse()``, and registering an
instance with :cpp:class:`FileParserRegistry` at application startup.
Once registered, the new format becomes selectable in the catalog
overlay-import dialog with no further wiring; that workflow is described
in :doc:`/user_guide/overlays`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: CatalogParser
   :members:
   :protected-members:
   :undoc-members:
