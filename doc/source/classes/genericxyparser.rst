.. index::
   single: GenericXYParser
   single: GenericXYData
   single: GenericXY; format
   single: parsers; generic XY
   single: ParseSettings
   single: ParsePreview

GenericXYParser
===============

``GenericXYParser`` is the catch-all importer for two-column XY data
files. It accepts CSV, TSV, space-delimited, and similar plain-text
formats with the suffixes ``.csv``, ``.tsv``, ``.txt``, ``.dat``,
``.data``, ``.xy``, and ``.tab``, returning the parsed point list as a
:cpp:class:`GenericXYData` value. The parser drives the *Generic XY*
overlay type described in :doc:`/user_guide/overlays`, where the user
loads an arbitrary external trace and overlays it on the FT plot.

Auto-detection covers three things in turn: the delimiter (comma, tab,
semicolon, or whitespace), the number of leading comment/header lines
(using ``#``, ``!``, and ``%`` as comment markers), and whether the
first non-comment line is a textual column-header row. The X and Y
columns default to indices ``0`` and ``1`` and can be overridden in the
:cpp:class:`GenericXYParser::ParseSettings` argument that
``parseWithSettings()`` accepts. Lines that fail to parse as numbers
are silently skipped so one malformed row does not abort the import.

The two helper structs define the import dialog's contract with the
parser. :cpp:class:`GenericXYParser::ParseSettings` carries the
resolved delimiter, header-line count, X/Y column indices, column
names, and a flag indicating whether a column-header row is present;
the dialog can edit it before calling ``parseWithSettings``.
:cpp:class:`GenericXYParser::ParsePreview` is the result of
``generatePreview()`` and combines the auto-detected settings with a
short slice of sample lines, a small set of parsed points for display,
the total data-line count, and an error message when detection fails.
``GenericXYOverlayWidget`` uses the preview to render the import
dialog's sample table before the user commits to the full parse.

A per-instance file-analysis cache keyed by ``(filePath, lastModified)``
lets the ``canParse``, ``generatePreview``, and ``parseWithSettings``
paths share a single detection result so a single file does not get
sniffed three separate times during one overlay-import session.

The parser is registered with :cpp:class:`FileParserRegistry` during
application startup. ``GenericXYParser`` derives directly from
:cpp:class:`FileParser` rather than from :cpp:class:`CatalogParser`
because its output is plain ``(x, y)`` data, not transitions with
quantum numbers; consumers that need a generic XY parser select it
through the registry's templated
:cpp:func:`FileParserRegistry::findParserOfType` helper.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: GenericXYParser
   :members:
   :protected-members:
   :undoc-members:
