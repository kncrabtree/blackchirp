.. index::
   single: FileParserRegistry
   single: FileParserRegistry; singleton
   single: parsers; registry
   single: file dialog; filter string
   single: SPCAT
   single: XIAM
   single: GenericXY

FileParserRegistry
==================

``FileParserRegistry`` is the process-wide singleton that owns every
:cpp:class:`FileParser` instance Blackchirp knows about. The registry
is populated during application startup — ``main()`` calls
:cpp:func:`FileParserRegistry::registerParser` once for each shipped
parser (:cpp:class:`SPCATParser`, :cpp:class:`XIAMParser`, and
:cpp:class:`GenericXYParser`) — and then services every "given this
file, find a parser that understands it" request from the rest of the
application. The overlay-import dialogs (``CatalogOverlayWidget``,
``GenericXYOverlayWidget``) and the overlay-replay code path
(``OverlayOperation``) all consume the registry directly; nothing in
Blackchirp constructs a parser instance outside of this class.

Two lookup methods cover the dispatch surface.
:cpp:func:`FileParserRegistry::findParser` returns the first registered
parser whose :cpp:func:`FileParser::canParse` returns ``true`` for the
candidate file; :cpp:func:`FileParserRegistry::findParserOfType` is a
templated variant that filters on a specific parser family — for
instance, the catalog-import dialog only wants a
:cpp:class:`CatalogParser`, even if a generic parser would also accept
the file. A registered parser keeps its position in the registry's
internal vector, so registration order determines lookup priority when
two parsers would both claim a file.

The registry is also the source of truth for ``QFileDialog`` filter
strings: :cpp:func:`FileParserRegistry::fileDialogFilter` joins each
parser's :cpp:func:`FileParser::fileExtensions` into a
``;;``-separated filter string suitable for
``QFileDialog::getOpenFileName``, with one entry per format, an "All
Catalog Files" entry combining every supported extension, and a final
"All Files" wildcard. The :cpp:func:`FileParserRegistry::parserRegistered`
signal fires once per :cpp:func:`FileParserRegistry::registerParser`
call and is intended for UI surfaces that want to refresh their format
lists when a new parser is added at runtime.

The user-facing import workflow that consumes the registry is
described in :doc:`/user_guide/overlays`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: FileParserRegistry
   :members:
   :undoc-members:
