.. index::
   single: Python module; architecture
   single: Python module; class layout
   single: Python module; schema versioning
   single: Python module; public API
   single: Python module; tests
   single: Python module; example notebooks
   single: BCExperiment; architecture
   single: BCFTMW; architecture
   single: BCFid; architecture
   single: BCLIF; architecture
   single: BCLifTrace; architecture
   single: schema; v1 and v2 readers

Python Module
=============

The ``blackchirp`` Python module under ``python/blackchirp/`` is the
read-side companion to the C++ acquisition application. It loads a
Blackchirp experiment folder from disk, decodes the on-disk CSV
schema, and reproduces the data-processing pipeline (FID Fourier
transforms, sideband deconvolution, LIF gate integration) so an
analysis script can work with the same numbers the live GUI shows.

The module is **read-only with respect to acquisition**: it does not
talk to hardware, it does not write experiment files, and it does not
depend on Qt, the hardware library, or any other piece of the C++
runtime. It runs under any Python 3.9 environment with numpy, scipy,
and pandas available, including (deliberately) on machines that are
not configured to build Blackchirp at all.

The build, test, and packaging story for the module — pyproject.toml
layout, ``python -m build``, the ``dev`` extra, the PyPI release
path — is in :doc:`/developer_guide/build_system`. The style and
docstring contract is in :doc:`/developer_guide/conventions`. This
page covers the architecture: what each class does, how the on-disk
schema flows into the class hierarchy, how schema versioning is
handled, and how the pieces fit together.

Module layout
-------------

The package is organized as one module per major class plus two
free-function modules:

================================ =================================================
Module                           Contents
================================ =================================================
``blackchirpexperiment.py``      :class:`~blackchirp.BCExperiment` — the entry point.
``bcftmw.py``                    :class:`~blackchirp.BCFTMW` — multi-FID FTMW container.
``bcfid.py``                     :class:`~blackchirp.BCFid` — single-FID container plus FT.
``bclif.py``                     :class:`~blackchirp.BCLIF` and :class:`~blackchirp.BCLifTrace` — LIF containers and aggregating helpers.
``coaverage.py``                 :func:`~blackchirp.coaverage_fids`, :func:`~blackchirp.coaverage_spectra` — multi-experiment FID combination.
``_enum_helpers.py``             Internal: parse Q_ENUM cells in either string or integer form.
``__init__.py``                  Public API surface — controls what ``from blackchirp import *`` brings in.
================================ =================================================

The leading-underscore module (``_enum_helpers``) is internal and is
not part of the public surface. Likewise, leading-underscore names
inside the public modules are implementation details that may change
without a version bump.

Class layout
------------

The class hierarchy mirrors the on-disk experiment structure:

::

   BCExperiment ── owns ──> BCFTMW ── owns ──> BCFid (one per fid/N.csv)
                  ── owns ──> BCLIF ── owns ──> BCLifTrace (one per lif/N.csv)

The relationship is composition, not inheritance. Each lower-level
class is constructed by its owner and is not intended to be
instantiated directly by user code, with one exception:
:class:`~blackchirp.BCFid` is sometimes constructed manually from a
:func:`~blackchirp.coaverage_fids` result that the user wants to
manipulate further.

BCExperiment — the entry point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~blackchirp.BCExperiment` is the top-level class users
construct. Given either a Blackchirp data-storage folder and an
experiment number, or a direct path to an experiment folder, it
locates ``version.csv``, reads the CSV separator from its first line,
and loads every top-level CSV file into a pandas ``DataFrame`` exposed
as an attribute of the same name (``header``, ``objectives``, ``log``,
``hardware``, ``clocks``, and the optional ``auxdata``, ``chirps``,
``markers``).

If the experiment contains a ``fid/`` subdirectory it constructs a
:class:`~blackchirp.BCFTMW` instance and exposes it as ``ftmw``;
likewise for ``lif/`` and :class:`~blackchirp.BCLIF`. Either
sub-container may be present, both, or neither (a hardware-only
acquisition might produce neither). The presence of CP-FTMW data
without ``clocks.csv`` is treated as a malformed experiment and
raises ``FileNotFoundError`` on construction.

The class also provides three header helpers — ``header_unique_keys``,
``header_rows``, and ``header_value`` / ``header_unit`` — that wrap
common pandas filter patterns against ``header.csv``. These exist
because ``header.csv`` is structured as ``(ObjKey, ArrayKey,
ValueKey, Value, Units)`` rows and downstream code otherwise has to
re-implement the filter chain at every call site.

BCFTMW — multi-FID FTMW data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~blackchirp.BCFTMW` represents the contents of an
experiment's ``fid/`` directory: ``fidparams.csv`` (one row per FID
with metadata), ``processing.csv`` (default FT processing settings),
and the per-FID ``N.csv`` files (raw base-36-encoded sample data,
loaded lazily by :class:`~blackchirp.BCFid`).

It exposes:

* ``get_fid(num)`` — construct a :class:`~blackchirp.BCFid` for the
  Nth FID. The default ``num=0`` covers single-FID acquisitions.
* ``get_differential_fid(start, end)`` — for ``Forever`` acquisitions
  with backups, return a :class:`~blackchirp.BCFid` whose data is the
  difference between the named backup endpoints. Multi-segment
  acquisition types (``LO_Scan``, ``DR_Scan``, ``Peak_Up``) gate this
  out because the differential math does not generalize across LO or
  DR steps.
* ``process_sideband(...)`` — sideband deconvolution for ``LO_Scan``
  acquisitions, with selectable averaging algorithm, FT range, and
  sideband choice. The result is an ``(x, y)`` numpy-array pair.

The ``ftmw_type`` field passed in at construction (read from
``objectives.csv``) is what gates the differential and sideband APIs
to the right acquisition kinds.

BCFid — single FID and its Fourier transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~blackchirp.BCFid` reads one ``fid/N.csv`` file, decodes the
base-36-packed accumulated samples into per-shot voltages using the
matching ``fidparams.csv`` row (``vmult / shots``), and stores the
result as a 2D numpy array shaped ``(size, frames)``. Single-frame
acquisitions still have shape ``(size, 1)`` so downstream code can
index uniformly.

The ``ft()`` method computes the Fourier transform of every frame
using the default settings drawn from ``processing.csv``; any of those
settings (window function, exponential filter, zero-padding factor,
start/end window, FT units) can be overridden per call via keyword
argument. Each named ``ft`` argument left as ``None`` falls back to
the value in ``processing.csv``, which is the same default-vs-override
pattern Blackchirp's GUI FID-processing menu uses.

The class deliberately keeps the *raw* base-36 data and the *decoded*
voltage array separate so that arithmetic operations
(:func:`~blackchirp.coaverage_fids` summing raw integers across
multiple FIDs, the differential-FID API subtracting two raw arrays)
can work in integer space and rescale to voltage at the end. Doing
the arithmetic in float would lose precision in the long-coaverage
limit.

BCLIF and BCLifTrace
~~~~~~~~~~~~~~~~~~~~

:class:`~blackchirp.BCLIF` is the LIF analog of
:class:`~blackchirp.BCFTMW`: it owns a ``lif/`` directory, reads
``lifparams.csv`` and ``lif/processing.csv``, and exposes per-point
trace access plus aggregating helpers (``delay_slice``,
``laser_slice``, ``image``) that integrate across the laser and delay
axes. Missing scan points are reported as ``np.nan`` (or any value
passed via the ``fill=`` argument) rather than silently zero-filled.

:class:`~blackchirp.BCLifTrace` is the single-point counterpart to
:class:`~blackchirp.BCFid`: one ``lif/N.csv`` file per
``(laser, delay)`` pair, decoded the same way, with smoothing and
integration operations that mirror the C++ ``LifTrace::processXY``
and ``LifTrace::integrate`` semantics. The integrated yields match
the GUI bit-for-bit; deviations are bugs.

Coaverage helpers
~~~~~~~~~~~~~~~~~

:func:`~blackchirp.coaverage_fids` and
:func:`~blackchirp.coaverage_spectra` live at the package root rather
than as methods on any class because their inputs span more than one
:class:`~blackchirp.BCExperiment` and their outputs are not naturally
methods on any single existing object.

* :func:`~blackchirp.coaverage_fids` performs time-domain coaverage:
  it sums raw integer data shot-for-shot across the input FIDs,
  recomputes voltages from the (shared) ``vmult`` and total shot
  count, and returns a fresh :class:`~blackchirp.BCFid`. Optional
  cross-correlation phase correction shifts each non-reference FID
  by the integer offset that maximizes correlation against the
  reference window.
* :func:`~blackchirp.coaverage_spectra` performs shot-weighted
  coaverage of magnitude spectra, returning ``(x, y)`` arrays. Used
  when phase drift defeats time-domain alignment.

Both enforce strict compatibility between their inputs (matching
``spacing``, ``size``, ``sideband``, ``probefreq``, ``vmult``, frame
count) and raise ``ValueError`` rather than silently coerce
mismatches. The C++ acquisition path has no analogous primitive, so
the Python module is the canonical home for this operation.

Schema versioning
-----------------

Blackchirp's on-disk format has changed across versions. The Python
module supports both the v1 schema (Blackchirp 1.x) and the v2 schema
(Blackchirp 2.0+) within the same loader; users do not have to know
which version produced their data.

The version is detected from the second line of ``version.csv``,
which carries the schema version that produced the file. The reader
dispatches on that value when interpreting fields whose meaning has
changed; for unchanged fields it reads the same path under either
schema.

The principal version-keyed differences:

* **Q_ENUM cells.** v1 stores enum values as integers
  (``"3"`` for ``BlackmanHarris``); v2 stores canonical names
  (``"BlackmanHarris"``). The :func:`_resolve_enum` helper in
  ``_enum_helpers.py`` accepts either form; every site that consumes
  an enum field routes through this helper so the dispatch is in
  exactly one place.
* **``hardware.csv`` column header.** v1 uses ``subKey``; v2 uses
  ``driver``. :class:`~blackchirp.BCExperiment` renames ``subKey``
  to ``driver`` on read so downstream code uses one column name
  regardless.
* **``processing.csv`` defaults.** Some processing keys have changed
  default value across schema versions. The Python module trusts the
  on-disk default and applies it; downstream code can override per
  call via keyword argument.

The principle is: **code that touches a version-specific field reads
the schema version once and dispatches in one place.** Spreading
the dispatch across a dozen call sites makes adding v3 a multi-file
change.

The shared example fixtures under ``python/example-data/`` include
both v1 (``mtbe``) and v2 (``v2-ftmw``, ``v2-lif-ref``,
``v2-lif-noref``) acquisitions specifically so the test suite can
parametrize over schema versions and catch dispatch regressions.

Public API surface
------------------

The names re-exported from ``blackchirp/__init__.py`` are the public
contract. As of the current release that is five classes —
:class:`~blackchirp.BCExperiment`, :class:`~blackchirp.BCFTMW`,
:class:`~blackchirp.BCFid`, :class:`~blackchirp.BCLIF`,
:class:`~blackchirp.BCLifTrace` — and two free functions —
:func:`~blackchirp.coaverage_fids` and
:func:`~blackchirp.coaverage_spectra`.

The recommended import style brings them all into the current
namespace::

   from blackchirp import *

The wildcard import is the only place ``import *`` is used in the
codebase; ordinary modules import what they use by name. Notebooks
also use ``from blackchirp import *`` because the wildcard form is
the documented entry point for end users.

Adding a name to the public surface is a deliberate decision with
semantic versioning consequences; removing one is a breaking change.
The recipe for graduating an internal helper to public status is in
:doc:`/developer_guide/conventions`.

Internal modules and leading-underscore names within public modules
are implementation details and may change without a version bump.
This includes:

* ``_enum_helpers`` — Q_ENUM cell parsing.
* ``_WINDOW_MAP``, ``_FT_UNITS_MAP``, ``_SIDEBAND_MAP``, etc. in
  ``bcfid.py`` — internal lookup tables for the FT processing
  pipeline.
* ``_resolve_time_scale``, ``_resolve_freq_scale_from_mhz`` in
  ``bcfid.py`` — internal unit-conversion helpers.

Example notebooks
-----------------

Two notebooks live alongside the package, not inside it:

* ``python/single-fid.ipynb`` — end-to-end CP-FTMW analysis: load an
  experiment, fetch a FID, take its FT, plot the result.
* ``python/single-lif.ipynb`` — LIF analog: fetch a trace, smooth and
  integrate, build a 2D image.

The notebooks serve two purposes:

1. **Documentation.** They are linked into the Sphinx build via
   ``nbsphinx-link`` and rendered as pages under
   :doc:`/python/example`. Their cell outputs (plots, DataFrames)
   appear in the rendered HTML, so they double as visual examples.
2. **Living tests of the public API.** They exercise the documented
   import style (``from blackchirp import *``) and the documented
   class methods. If a change breaks the notebook, the change has
   broken something user-visible.

Notebooks must be re-executed end-to-end before commit when their
substantive cells have changed; ``nbsphinx`` renders existing cell
outputs verbatim and a partially-executed notebook will render
incorrectly. Execute in an environment with the package installed
(``pip install -e python/blackchirp``) plus matplotlib and a Jupyter
kernel — typically the conda environment described by
``python/environment.yml``.

The notebooks may import matplotlib at the cell level. The package
itself does not, and matplotlib is not in the package's dependency
list; see the dependency policy in :doc:`/developer_guide/conventions`.

Test suite
----------

The pytest suite under ``python/blackchirp/tests/`` covers four broad
areas: schema loading (v1 and v2 fixtures), FID and LIF processing,
unit-conversion and enum-helper edge cases, and coaverage. Test files
follow the ``test_<feature>.py`` naming convention; the test runner
is plain ``pytest``.

Fixtures and example data
~~~~~~~~~~~~~~~~~~~~~~~~~

Test fixtures live under ``python/example-data/`` and are loaded by
``python/blackchirp/tests/conftest.py`` via a relative path. The four
fixture names are:

* ``mtbe`` — v1-style ``Forever`` acquisition with multiple backups.
* ``v2-ftmw`` — v2 ``Forever`` acquisition with multiple backups.
* ``v2-lif-ref`` — v2 LIF acquisition with a reference channel.
* ``v2-lif-noref`` — v2 LIF acquisition without a reference channel.

The same ``python/example-data/`` directory is referenced by the C++
test ``tst_experimentloading``, which keeps the on-disk schema
definitions consistent across the two languages: a fixture-format
change that breaks the C++ loader will also break the Python loader.
This is intentional. Do not add C++-only or Python-only fixtures —
when an addition is needed, write it in a form that both loaders can
read.

The ``conftest.py`` exposes per-fixture session-scoped paths and
per-test instance-scoped :class:`~blackchirp.BCExperiment` objects,
plus a ``any_exp`` parametrized fixture that runs a test against
both ``mtbe`` and ``v2-ftmw``. Tests that mutate the FID data in
place use the per-test instance so each test gets a fresh load;
tests that only read use the session-scoped path and re-construct as
needed.

Test organization
~~~~~~~~~~~~~~~~~

Tests are grouped by what they exercise, not by which class they
poke at, so the same class may be touched from several test files:

* ``test_load_v1.py``, ``test_load_v2.py`` — schema-loading round
  trips against the corresponding fixture version.
* ``test_ft_units.py``, ``test_window_dispatch.py``,
  ``test_sideband_dispatch.py`` — FID processing with explicit
  parameter overrides.
* ``test_processing_overrides.py`` — the default-vs-override
  contract on :meth:`~blackchirp.BCFid.ft`.
* ``test_differential_fid.py`` — the
  :meth:`~blackchirp.BCFTMW.get_differential_fid` API.
* ``test_lif_*.py`` — LIF loading, scan-point access, smoothing,
  integration, and the missing-point fallback.
* ``test_coaverage.py`` — both coaverage entry points and their
  compatibility checks.
* ``test_enum_helpers.py``, ``test_error_paths.py``,
  ``test_ftmw_axis_units.py`` — internal helpers and edge cases.

Adding a new test follows the standard pytest pattern: a
``test_*.py`` file under ``python/blackchirp/tests/``, function names
``test_*``, fixtures from ``conftest.py`` requested by parameter
name. New fixtures (if needed) go under ``python/example-data/``.
A bare ``pytest`` run from anywhere in the project, or
``pytest --rootdir python/blackchirp python/blackchirp/tests``,
picks up the new test automatically.

Recipes
-------

Adding a new processing helper to BCFid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new method on :class:`~blackchirp.BCFid` (e.g., a different window
function, a different filtering operation) is the simplest extension.
The pattern is:

1. Add the method to ``bcfid.py`` next to the existing methods. Use
   Google-style docstring (``Args:``, ``Returns:``, ``Raises:``,
   ``Example:``); the rendering contract is in
   :ref:`api-reference-style`.
2. If the helper introduces a new processing-settings key, add it to
   ``_PROC_INT_KEYS`` / ``_PROC_FLOAT_KEYS`` / ``_PROC_BOOL_KEYS``
   in ``bclif.py`` and the corresponding tables in ``bcfid.py`` so
   that the parsing dispatch picks up the right type.
3. Add a test under ``tests/test_<area>.py`` exercising both the
   default-from-``processing.csv`` path and an explicit-override
   path.
4. Update the relevant API page under ``doc/source/python/`` if the
   new method warrants a sentence of orientation prose; otherwise
   ``autodoc`` picks it up automatically.

Adding a new top-level class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new top-level class is a public-API change. Beyond the steps above:

1. Place the class in its own module (``bcnewthing.py`` or similar)
   following the one-class-per-module convention. Internal helpers
   for the class go in the same module with leading-underscore names.
2. Add the class to ``blackchirp/__init__.py``: an ``import`` line
   and a paragraph in the module docstring.
3. Create a class page under ``doc/source/python/<classname>.rst``
   following the structure of :doc:`/python/bcfid`.
4. Add the page to the toctree in ``doc/source/python.rst``.
5. Bump the version in ``python/blackchirp/pyproject.toml`` (a public
   API addition is a minor version bump under semver).
6. Add an entry to the next-release changelog page under
   ``doc/source/changelog/``.

Adding support for a new schema version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the C++ acquisition schema changes in a way that affects the
Python loader:

1. Identify every field whose meaning, encoding, or column header
   changed. For each, locate the existing version-keyed dispatch
   (typically in :func:`_resolve_enum` or in
   :class:`~blackchirp.BCExperiment`'s constructor) and add a
   third branch.
2. Add a fixture to ``python/example-data/`` — a small acquisition
   in the new format that the C++ side has produced.
3. Parametrize the loading tests over the new fixture by adding a
   case to ``conftest.py`` and (if the schema version warrants it) a
   new ``test_load_v<n>.py``.
4. Document the version-keyed differences in this page's *Schema
   versioning* section above. The principle of one-place dispatch
   means the documentation can match the code without surveying the
   call sites individually.

Dependencies the rest of the project does not have
--------------------------------------------------

The Python module is the only place where:

* Numpy and scipy are runtime dependencies.
* Pandas is a runtime dependency.
* Pytest is the test framework (the C++ tests use Qt-Test).
* Black is the formatter and pylint is the linter (the C++ tree has
  no enforced formatter; clang-format usage is per-developer).
* The package version is independent of the project version.

These come up in PR review when a contributor accustomed to one tree
makes assumptions about the other; the assumption is usually
incorrect. The two trees share the source repo, the ``example-data``
fixtures, and (loosely) the on-disk schema; they share nothing else.
