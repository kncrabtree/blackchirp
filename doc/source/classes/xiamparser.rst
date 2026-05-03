.. index::
   single: XIAMParser
   single: XIAM
   single: catalog; XIAM format
   single: parsers; XIAM
   single: internal rotation

XIAMParser
==========

``XIAMParser`` reads output files produced by XIAM (eXtended Internal
Axis Method), a program for analyzing internal-rotation effects in
molecular spectra. XIAM output is whitespace-delimited and is
recognized by the ``-- B <number>`` block header followed by a
column-header line; the parser sniffs this pattern in
:cpp:func:`XIAMParser::canParse` so non-XIAM ``.out`` files are
rejected. Recognized file extensions are ``.xo`` and ``.out``.

The output format depends on the XIAM ``ints`` setting and the parser
handles both modes that ``CatalogOverlay`` needs:

- **ints=2** — a simple frequency/intensity listing, one transition per
  line, with a single block.
- **ints=3** — frequency, rigid-rotor reference, and per-symmetry-state
  splittings; the parser unrolls each split into its own transition so
  downstream consumers can treat every line as an independent record.
  Quantum numbers from the block start are inherited by the split
  lines.

XIAM emits intensities with limited decimal precision, which can lose
significant digits for weak transitions. The parser reconstructs a
higher-precision intensity from the constituent line strength,
statistical weight, population factor, and energy factor, returning
whichever value is more precise. The molecule name is read from the
XIAM file header when present and falls back to the file's base name.

The output :cpp:class:`CatalogData` carries ``sourceProgram = "XIAM"``.
The parser is registered with :cpp:class:`FileParserRegistry` during
application startup; the user-facing catalog overlay workflow is
described in :doc:`/user_guide/overlays`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: XIAMParser
   :members:
   :protected-members:
   :undoc-members:
