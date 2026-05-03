.. index::
   single: FileParser
   single: parsers; abstract base
   single: file format; recognition
   single: FileParserRegistry; FileParser interface

FileParser
==========

``FileParser`` is the abstract root of Blackchirp's file-parser hierarchy.
Every class that imports an external data file — spectroscopic line
catalogs, generic two-column XY data, anything else added later — derives
from this interface. The class defines four pure-virtual hook points that
identify the format and a small set of protected helpers covering the
file-system patterns shared by every concrete parser.

Subclass authors implement :cpp:func:`FileParser::canParse` to recognize
their format (typically a suffix check followed by a structural sniff of
the first lines), :cpp:func:`FileParser::formatName` and
:cpp:func:`FileParser::formatDescription` for user-facing labels, and
:cpp:func:`FileParser::fileExtensions` for ``QFileDialog`` glob patterns.
The protected helpers — :cpp:func:`FileParser::isFileReadable`,
:cpp:func:`FileParser::hasMatchingExtension`, and
:cpp:func:`FileParser::readFileHeader` — let ``canParse`` implementations
share a single readable-file check, a case-insensitive suffix match, and
an N-line peek without each parser re-implementing them.

The actual parse method that returns data lives on the format-specific
subclass because the value type differs by family:
:cpp:class:`CatalogParser` adds a ``parse() → CatalogData`` hook for
spectroscopic line catalogs, and :cpp:class:`GenericXYParser` adds its
own ``parse() → GenericXYData`` directly. Concrete parsers register
themselves with :cpp:class:`FileParserRegistry` during application
startup; the registry then dispatches incoming files to the first parser
whose ``canParse`` returns ``true``. The user-facing import workflow
that consumes these parsers is described in :doc:`/user_guide/overlays`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: FileParser
   :members:
   :protected-members:
   :undoc-members:
