.. index::
   single: SPCATParser
   single: SPCAT
   single: SPFIT
   single: catalog; SPCAT format
   single: parsers; SPCAT

SPCATParser
===========

``SPCATParser`` reads catalog files written by the Pickett SPFIT/SPCAT
package, which are conventionally given the ``.cat`` suffix. SPCAT is
the de facto reference format for asymmetric-top line catalogs in
microwave and millimeter-wave spectroscopy; many other fitting
programs emit a SPCAT-compatible ``.cat`` file as their primary user
output, so this parser covers a large fraction of the catalogs that
users overlay onto a measured spectrum.

The format is fixed-width with eighty characters per line: frequency
(MHz), calculated error (MHz), log-base-10 intensity, degeneracy,
lower-state energy (cm⁻¹), species tag, format code, and quantum-number
assignments starting at column 55. The format code controls how the
quantum-number block is interpreted; ``SPCATParser`` preserves the
field grouping but strips embedded semicolons so the quantum-number
strings round-trip safely through Blackchirp's semicolon-delimited CSV
storage. Lines shorter than 80 characters are right-padded; longer
lines are truncated. Lines that do not produce a valid transition (zero
frequency or unparseable fields) are silently skipped, so a catalog
file with a trailing summary block parses without manual editing.

The output :cpp:class:`CatalogData` carries ``sourceProgram = "SPCAT"``
and a molecule name taken from the file's base name. The parser is
registered with :cpp:class:`FileParserRegistry` during application
startup; the user-facing catalog overlay workflow is described in
:doc:`/user_guide/overlays`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: SPCATParser
   :members:
   :protected-members:
   :undoc-members:
