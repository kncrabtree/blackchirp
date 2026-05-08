.. _conventions-and-style:

.. index::
   single: coding conventions
   single: naming
   single: member-variable prefixes
   single: string literals
   single: QAnyStringView
   single: QLatin1StringView
   single: heterogeneous lookup
   single: key declaration patterns
   single: BC::Key
   single: BC::Store
   single: BC::CSV
   single: logging
   single: bcLog
   single: hwLog
   single: persistent settings
   single: SettingsStorage; conventions
   single: Python; coding conventions
   single: Python; docstrings
   single: Python; dependency policy
   single: Python; public API
   single: documentation; voice
   single: documentation; American English
   single: documentation; cross-references
   single: documentation; index entries
   single: documentation; screenshots
   single: documentation; notebooks
   single: API reference; style
   single: API reference; where prose lives
   single: API reference; Sphinx directives
   single: Doxygen comments

Conventions and Style
=====================

This page is the canonical reference for *what Blackchirp code, Python
code, and documentation prose look like* once you sit down to add or
modify a file. Three sections cover the three trees, and a final
cross-cutting section covers the API reference contract that ties C++
headers, Python source, and the Sphinx pages under :doc:`/classes` and
:doc:`/python` together.

What this page does not cover: build commands, test invocations,
dependency installation, and the rest of the *how do I run X* surface
live in :doc:`/developer_guide/build_system` and in the per-tree
``AGENTS.md`` files at the root of ``src/``, ``doc/``, and ``python/``.
This page is for the open-the-file-and-write-code half of the work.

C++
---

The C++ section assumes fluency with Qt6 (``QObject``, the metaobject
system, ``QString`` / ``QAnyStringView``) and the standard Qt
vocabulary. It is not a Qt tutorial.

Naming and indentation
~~~~~~~~~~~~~~~~~~~~~~

* Classes, structs, and enums use ``UpperCamelCase``: ``HardwareObject``,
  ``FtmwConfig``, ``CommunicationProtocol::CommType``.
* Free functions and variables use ``lowerCamelCase``: ``hwReadSettings()``,
  ``currentExperimentNum``.
* Member variables carry a one- or two-letter prefix that records what
  they hold:

  ============  ============================================
  Prefix        Meaning
  ============  ============================================
  ``d_``        Value member (``d_key``, ``d_currentShots``).
  ``p_``        Raw pointer (``p_comm``, ``p_settings``).
  ``pu_``       ``std::unique_ptr`` (``pu_worker``).
  ``ps_``       ``std::shared_ptr`` (``ps_storage``).
  ============  ============================================

  The prefix is part of the name, not a hint: code reviews catch
  ``int currentShots`` on a member where ``d_currentShots`` was
  intended. Static class members follow the same rule
  (``s_instance`` for a singleton handle).

* Indentation is four spaces, spaces only, no tabs. New files inherit
  the project's ``.editorconfig``-equivalent settings; if your editor
  produces tab characters, fix it before committing.

String literals
~~~~~~~~~~~~~~~

Blackchirp uses Qt6's user-defined string-literal suffixes everywhere
strings appear in source. Pick the form by what the call site needs.

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Form
     - Type
     - When to use
   * - ``"..."_L1``
     - ``QLatin1StringView``
     - ASCII content. **Default choice** — accepted by any
       ``QAnyStringView`` parameter without constructing a temporary
       ``QString``.
   * - ``u"..."_s``
     - ``QString``
     - Non-ASCII content (e.g., ``u"μs"_s``); also when the call site
       requires a ``QString`` and the literal contains non-ASCII
       characters.
   * - ``"..."_s``
     - ``QString``
     - Only when the call site genuinely requires a ``QString``:
       ``.arg()`` receivers, widget constructors, ``QStringList``
       initializers, ``QRegularExpression``.
   * - ``QStringLiteral(...)``
     - ``QString``
     - **Do not use in new code.** Replace existing occurrences when
       editing a file that already uses them; otherwise leave alone.

Do not reach for ``"..."_s`` just because a parameter is
``QAnyStringView``. Constructing a temporary ``QString`` defeats the
view parameter; ``"..."_L1`` is what the parameter is designed for.

The ``Qt::Literals::StringLiterals`` namespace is pulled in globally
through ``data/loghandler.h``, so any translation unit that includes
the log header (directly or transitively) gets ``_s`` and ``_L1`` for
free. Headers that are read from contexts that do not include the log
header should still spell out the constructor form for declarations
(see *Key declaration patterns* below) so the file does not depend on
the suffixes being visible.

Function signatures
~~~~~~~~~~~~~~~~~~~

* **Never pass ``QString`` by value** unless the callee takes ownership
  and moves. Implicit sharing makes copying cheap, but a value parameter
  still costs an atomic refcount round-trip per call.
* Use ``QAnyStringView`` for pure lookup, comparison, or pass-through
  functions — anything that is going to forward the string into another
  view-aware API or compare it. ``QAnyStringView`` accepts ``QString``,
  ``QStringView``, ``QLatin1StringView``, and ``const char *`` without a
  conversion.
* Use ``const QString &`` when the callee genuinely needs a ``QString``:
  it calls ``.arg()`` on it, stores it as ``QString``, or hands it to
  another API whose parameter is ``const QString &``.

In practice, the rule of thumb is: if the body would otherwise begin
``QString s = view.toString();``, the parameter should have been
``const QString &`` to begin with.

Containers
~~~~~~~~~~

For new ``std::map`` declarations keyed on ``QString``, use the
transparent comparator:

.. code-block:: cpp

   std::map<QString, MyValue, std::less<>> d_table;

The transparent ``std::less<>`` enables heterogeneous lookup: a call
to ``find("key"_L1)`` or ``find(QStringView{...})`` will not allocate a
temporary ``QString`` to perform the comparison. Retrofit existing
declarations opportunistically when editing the surrounding code.

``QHash<QString, T>`` does not need any special declaration. Qt6 ships
``qHash`` overloads that hash a ``QStringView`` to the same value as
the equivalent ``QString``, so heterogeneous lookup works on
``QHash`` directly.

Key declaration patterns
~~~~~~~~~~~~~~~~~~~~~~~~

Blackchirp uses three patterns to declare named string keys at namespace
scope. All three are ``inline``, so the symbol has external linkage with
exactly one definition in the program — there is no per-translation-unit
copy. Pick the pattern by the type the call site needs.

**Pattern A — ``inline const QString``.** Use when the key is consumed
as a ``QString`` (typically because ``.arg()`` is called on it):

.. code-block:: cpp

   inline const QString flow = "Flow%1"_s;

One heap allocation per process, not per translation unit; the cost is
the same as any other static ``QString``.

**Pattern B — ``inline constexpr QLatin1StringView``.** Use for ASCII
keys consumed by ``QAnyStringView`` parameters or stored in
``std::map<QString, T, std::less<>>`` with heterogeneous lookup. This
is the dominant pattern for hardware setting keys:

.. code-block:: cpp

   inline constexpr QLatin1StringView trigCh{"trigCh"};

True ``constexpr`` at namespace scope; zero runtime cost. In headers,
use the constructor form ``{"..."}`` rather than ``"..."_L1`` so the
declaration does not depend on a ``using namespace
Qt::Literals::StringLiterals`` being visible in the including file.

**Pattern C — ``inline constexpr QStringView``.** Same trade-offs as
Pattern B but UTF-16, so non-ASCII keys are safe:

.. code-block:: cpp

   inline constexpr QStringView us = u"μs";

The compile-error case is worth noting because the suffix syntax
suggests it should work:

.. code-block:: cpp

   // Does NOT compile — QString is not a literal type.
   inline constexpr auto k = "key"_s;

``QString`` is not ``constexpr``-eligible in any Qt version Blackchirp
targets. If a key needs to be ``QString`` at the call site, use Pattern
A; if it needs to be ``constexpr``, use Pattern B or C.

Several namespaces collect related keys so the compiler catches typos at
the call site rather than letting a stray string literal silently miss:

* ``BC::Key::`` — application-wide and hardware settings keys, declared
  in ``data/bcglobals.h`` and the per-base-class blocks of
  ``data/settings/hardwarekeys.h``. The hardware-key namespace is split
  into sub-namespaces by hardware type (``BC::Key::HW``,
  ``BC::Key::Comm``, ``BC::Key::Clock``, ``BC::Key::AWG``, …) so that a
  ``using namespace BC::Key::Clock`` pulls in only the clock keys.
* ``BC::Store::`` — persistent storage keys for serializable data
  classes (``BC::Store::RFC`` for ``RfConfig``, ``BC::Store::CC`` for
  ``ChirpConfig``, ``BC::Store::FtmwLO`` for FTMW LO scans, and so on).
  Declared in the header next to the class that owns them.
* ``BC::CSV::`` — canonical filenames and column headers for the
  semicolon-delimited CSV files produced by an experiment. Declared in
  ``data/storage/blackchirpcsv.h``.

Always reach for an existing namespace before introducing a string
literal at a call site. Adding a new key is a one-line edit to the
appropriate header.

Logging
~~~~~~~

All log output goes through the application-wide
:cpp:class:`LogHandler` singleton. Normal code calls one of the five
free functions declared in ``data/loghandler.h``:

.. code-block:: cpp

   bcLog(u"connected to %1"_s.arg(d_prettyName));     // Normal severity (default)
   bcLog(u"detail"_s, LogHandler::Debug);             // explicit severity
   bcDebug(u"protocol byte 0x%1"_s.arg(b, 2, 16));    // Debug severity
   bcWarn(u"timeout, retrying"_s);                    // Warning severity
   bcError(u"hardware failure: %1"_s.arg(err));       // Error severity
   bcHighlight(u"experiment %1 complete"_s.arg(num)); // Highlight severity

Inside :cpp:class:`HardwareObject` subclasses, prefer the four
member helpers — ``hwLog``, ``hwDebug``, ``hwWarn``, ``hwError`` —
which prepend the device key (``d_key``) automatically. The result is
that every line a driver writes is unambiguously attributed to that
driver in the in-app log and on disk; a hardware author should not have
to remember to do this manually.

Do not use ``qDebug()`` or ``emit logMessage()`` in new code. Do not
call ``LogHandler::instance()`` directly except in connection setup at
application startup; the free functions cover every other case.

Five severity levels are defined by ``LogHandler::MessageCode``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Level
     - Use for
   * - ``Error``
     - Failures requiring user action or indicating data-loss risk.
   * - ``Warning``
     - Automatically-corrected mismatches the user should know about.
   * - ``Normal``
     - Connection outcomes, experiment milestones, user-initiated state
       changes.
   * - ``Highlight``
     - Major milestones such as experiment start and end.
   * - ``Debug``
     - Hardware lifecycle, configuration loading, protocol details,
       parameter traces. Written to the debug log file only when debug
       logging is enabled.

The full API surface — singleton lifetime, on-disk file layout, the
per-experiment log lifecycle driven by ``beginExperimentLog`` and
``endExperimentLog``, the ``sendLogMessage`` and ``iconUpdate``
signals — is documented on :doc:`/classes/loghandler`.

Persistent settings
~~~~~~~~~~~~~~~~~~~

Persistent state is owned by :cpp:class:`SettingsStorage`. The class
is the single trust boundary between code that may read a value and
code that may change it.

* The ``get`` family is **public**. Anywhere in the program may
  construct a transient ``SettingsStorage`` over a group and read from
  it. UI code routinely does this to populate a widget from a hardware
  driver's persisted settings.
* The ``set`` family is **protected**. Only a class that *owns* a
  group — i.e. inherits from ``SettingsStorage`` and initializes its
  base with the group keys — may write. The split is what keeps
  persisted state from drifting behind the owner's back; a friend
  helper is the explicit escape hatch when one manager class needs to
  compose state across many groups.

Setting keys are declared statically. New code does not pass string
literals to ``get`` and ``set``; it passes a constant from
``BC::Key::`` (for hardware settings and other application-wide keys)
or ``BC::Store::`` (for serializable data classes). Adding a new key is
a one-line addition to the appropriate namespace block in the relevant
header. The compiler then catches typos at the call site, and a
``grep`` for the constant finds every reader and writer.

For hardware drivers specifically, *defaults* are not set by ad-hoc
``setDefault`` calls in the subclass constructor. They are declared
through the hardware-settings registry — ``REGISTER_HARDWARE_SETTINGS``
for a driver, ``REGISTER_HARDWARE_BASE`` for shared base-class
defaults — and applied by ``HardwareObject::applyRegisteredSettings()``
during construction. The registry is also what populates the hardware
settings dialog without the dialog ever instantiating a driver. The
full registration story and the rest of the runtime configuration model
are covered in :doc:`/developer_guide/hardware_configuration`. The
registration macros themselves and the descriptor structs they produce
are documented on :doc:`/classes/hardwareregistry`; the
``SettingsStorage`` API surface itself, including the array, group,
getter, and discard mechanisms, is documented on
:doc:`/classes/settingsstorage`.

Documentation comments
~~~~~~~~~~~~~~~~~~~~~~

Doxygen comments on C++ entities are governed by the
:ref:`api-reference-style` section of this page, which covers C++
headers and Python source files together. The same section sets out the
contract for what lives in headers versus on the rendered ``.rst``
pages under :doc:`/classes`.

Python
------

The Python section governs the standalone ``blackchirp`` PyPI module
under ``python/blackchirp/``. The module reads experiment folders from
disk and reproduces Blackchirp's data-processing pipeline (FID Fourier
transforms, sideband deconvolution, LIF gate integration) without
depending on any of the C++ runtime. The conventions below preserve
that property; deviations are not casual decisions.

Naming and formatting
~~~~~~~~~~~~~~~~~~~~~

* Module names use ``snake_case``. The package is ``blackchirp``;
  individual modules under it are ``bcfid``, ``bcftmw``, ``bclif``,
  ``coaverage``, and so on.
* Classes use ``PascalCase``: ``BCExperiment``, ``BCFid``, ``BCFTMW``.
  The ``BC`` prefix is the existing convention; new public classes
  follow it.
* Functions, methods, and variables use ``snake_case``: ``get_fid``,
  ``coaverage_fids``, ``shot_count``.
* Module-level constants use ``UPPER_SNAKE_CASE``:
  ``_WINDOW_MAP``, ``_FT_UNITS_MAP``. The leading underscore marks an
  internal-to-the-module helper.

Formatting is enforced by ``black`` with default settings (88-character
line length). Run ``black .`` from ``python/blackchirp/`` before
committing. Linting is ``pylint -E`` (errors only — warnings are not
load-bearing). Both should run clean before the PR is opened.

Imports follow `PEP 8 <https://peps.python.org/pep-0008/#imports>`_:
standard library first, third-party second (``numpy``, ``pandas``,
``scipy``), then intra-package relative imports last. Wildcard imports
are reserved for ``blackchirp/__init__.py`` and the example notebooks;
ordinary modules import what they use by name.

Docstrings
~~~~~~~~~~

Every public class and function must carry a docstring in
`Google style
<https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_,
which the Sphinx ``napoleon`` extension converts into proper RST at
documentation build time. The first line is a one-sentence summary;
subsequent paragraphs and sections describe behavior, parameters, and
return values.

Standard sections, in the order they typically appear:

* ``Args:`` — one entry per parameter, written
  ``name: description.``. Optional parameters note their default in
  the description.
* ``Returns:`` (or ``Yields:`` for generators) — describes the return
  value. Omit when the function returns ``None``.
* ``Raises:`` — exception types the caller may need to catch, with the
  condition that triggers each.
* ``Attributes:`` — for classes, public instance attributes the user
  is expected to read or write directly. Internal ``_attr`` and
  ``__attr`` are not documented here.
* ``Example:`` (or ``Examples:``) — a short executable snippet that
  exercises the typical call path. Class-level docstrings include an
  example whenever the class has a non-trivial constructor signature.
* ``Note:`` / ``Warning:`` — for caveats that the brief and parameter
  list cannot accommodate (algorithmic limitations, performance
  cliffs, schema-version dependencies).

Mathematical content uses inline RST math (``:math:`y = ax + b```)
inside the docstring; ``napoleon`` passes it through unchanged. See
the :func:`~blackchirp.coaverage_spectra` docstring for an example.

Internal helpers prefixed with ``_`` may carry a one-line docstring or
none at all. The line between "internal" and "public" is
``blackchirp/__init__.py``: anything re-exported there is public; the
rest is an implementation detail.

The header-vs-page contract that governs how docstrings render in the
:doc:`/python` API reference is in the :ref:`api-reference-style`
section below.

Dependency policy
~~~~~~~~~~~~~~~~~

The ``blackchirp`` module depends only on **numpy, scipy, and pandas**
(plus the standard library). No matplotlib, no Qt, no requests, no
``aiohttp``, no other plotting or networking libraries. The
minimal-dependency property is a deliberate design constraint: the
module must install cleanly into any scientific Python environment
without dragging heavy optional infrastructure with it.

Treat ``[project.dependencies]`` in
``python/blackchirp/pyproject.toml`` as load-bearing. Adding a runtime
dependency requires user consent and a documented justification on the
PR. Importing matplotlib at module top-level — even guarded by
``try/except ImportError`` — is not the right answer; if a downstream
caller wants plotting, it owns the matplotlib import.

The example notebooks under ``python/`` may import matplotlib at the
cell level. Notebooks have their own dependency story (they run in an
analysis environment that already has plotting libraries) and do not
push imports back into the module.

The only declared dev dependency is ``pytest``, listed under
``[project.optional-dependencies]`` as ``dev``. Add other dev tools
(coverage, ruff, etc.) only with user consent and only after
discussion.

Public API surface
~~~~~~~~~~~~~~~~~~

The names re-exported from ``blackchirp/__init__.py`` are the public
contract:

* :class:`~blackchirp.BCExperiment`
* :class:`~blackchirp.BCFTMW`
* :class:`~blackchirp.BCFid`
* :class:`~blackchirp.BCLIF`
* :class:`~blackchirp.BCLifTrace`
* :func:`~blackchirp.coaverage_fids`
* :func:`~blackchirp.coaverage_spectra`

Adding a name to that list is a deliberate decision with semantic
versioning consequences; removing one is a breaking change. New
internal helpers live in ``bcfid.py``, ``bclif.py``, etc. without being
re-exported. When a helper graduates to public status:

1. Add it to the ``from .module import name`` block in
   ``__init__.py``.
2. Add a paragraph to the module docstring at the top of ``__init__.py``
   describing the new symbol.
3. Add (or extend) a page under ``doc/source/python/`` for the symbol,
   following :ref:`api-reference-style` below.
4. Add an entry to the next-release changelog page under
   ``doc/source/changelog/``.

Internal modules (``_enum_helpers.py``, leading-underscore names) are
not part of the public surface and may change without a version bump.

Documentation
-------------

The Documentation section governs the Sphinx + Doxygen + Breathe + nbsphinx
documentation under ``doc/source/``, deployed to
https://blackchirp.readthedocs.io/. The conventions cover *prose
style*; the build pipeline and CMake target wiring are in
:doc:`/developer_guide/build_system`.

Voice and tense
~~~~~~~~~~~~~~~

User-facing prose is **present tense and impersonal**. "Blackchirp
writes the FID to disk", not "Blackchirp will write" and not "we
write". The user guide describes the program as it is, in the moment
the user is operating it.

Developer-guide prose follows the same default with one allowance:
**second person is acceptable** when giving a contributor explicit
instructions ("Add the driver header to the aggregator", "Read the
sub-bundle file before drafting the page"). Use it sparingly; most
developer prose can be written impersonally too.

Do not use **source-evolution temporal markers** in any prose: no
"Phase 2", "v1.1.0 introduced", "recently added", "now uses",
"previously did X but now does Y". Permanent version-keyed information
lives in the changelog or migration guide, not in the user guide or
developer guide. Markers describing **runtime behavior** are fine
("after the experiment completes", "before any FID is acquired",
"while connected").

The same rule applies to commit messages and code comments. The test:
would the sentence read correctly to a reader five years from now with
no knowledge of the commit that introduced it?

American English
~~~~~~~~~~~~~~~~

All prose uses American English: ``normalize`` / ``normalization``,
``behavior``, ``color``, ``visualization``, ``analyze``, ``co-averaging``,
``randomize``, ``initialize``. Match UI labels exactly when quoting
them, even if the UI label uses an American spelling — do not "correct"
a label such as "Randomize Delay Order" to British spelling in prose.

Cross-references and index entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Sphinx cross-reference roles, not raw HTML anchors. Replace any
``<page.html>``-style links opportunistically when editing a page.

* ``:doc:`/path/to/page``` — links to another documentation page.
* ``:ref:`label-name``` — links to a labeled section anchor across
  pages. Define labels with ``.. _label-name:`` immediately above the
  section header.
* ``:cpp:class:`ClassName``` and ``:cpp:func:`namespace::function```
  — link to C++ entities documented through Doxygen / Breathe.
* ``:class:`~blackchirp.BCFid``` and ``:func:`~blackchirp.coaverage_fids```
  — link to Python entities documented through ``autoclass`` /
  ``autofunction``. The leading ``~`` collapses the displayed name to
  the last component.
* ``:meth:`~blackchirp.BCFid.ft``` — link to a method.

Every new page begins with a ``.. index::`` block listing the
user-facing terms it introduces. The block sits between the optional
``.. _label:`` line and the page title. Use ``single:`` entries for
one-word terms and ``single: parent; child`` for hierarchical entries
that group multiple subtopics under one parent. Pages without an index
block are reachable but invisible from the documentation's index page.

Screenshots
~~~~~~~~~~~

UI screenshots live under
``doc/source/_static/user_guide/<page-name>/``. Reference them with
``.. figure::`` (not ``.. image::``) so the screenshot gets a caption
and fits into the page flow:

.. code-block:: rst

   .. figure:: /_static/user_guide/hardware_menu/profile_dialog.png
      :width: 80%
      :align: center

      The hardware profile dialog after creating a new profile.

Width is expressed as a percentage; the percentage depends on the
content density of the screenshot (forms and dialogs typically render
at 60–80%, full-window shots at 90–100%). Screenshots are PNG.
Hardware-specific screenshots that may need re-capture with new
firmware revisions live in the same directory as the page that uses
them.

Notebooks
~~~~~~~~~

The example notebooks under ``python/single-*.ipynb`` are referenced
from the documentation via ``nbsphinx-link``. Wrappers under
``doc/source/python/notebooks/`` carry the Sphinx-side metadata; the
notebook source itself is the artifact ``nbsphinx`` renders.

Three rules apply to any change that touches a notebook:

1. **Execute the notebook end-to-end before commit.** ``nbsphinx``
   renders the existing cell outputs verbatim; an unexecuted or
   partially-executed notebook will render with empty or stale output
   cells. Execute in an analysis environment with the published
   dependency set (``numpy``, ``scipy``, ``pandas``, ``matplotlib``,
   plus ``blackchirp`` itself).
2. **Exercise the public API only.** Notebooks demonstrate the same
   contract a downstream user has — anything imported with
   ``from blackchirp import *`` or ``import blackchirp as bc``.
   Reaching into ``_internal`` modules to make a notebook work means
   the public API is missing something; fix that first.
3. **Notebook content is illustrative, not exhaustive.** Use the
   notebook to walk through a realistic analysis session, not to
   document every parameter of every function. Reference detail lives
   on the corresponding API reference page.

.. _api-reference-style:

API reference style
-------------------

This section defines the contract between source code (C++ headers,
Python modules), the generators that read it (Doxygen for C++, Sphinx
``autodoc`` + ``napoleon`` for Python), and the rendered API pages
under :doc:`/classes` and :doc:`/python`. The contract is symmetric
across the two languages: member-level documentation lives next to the
code it describes, and the ``.rst`` page carries only orientation prose
plus the directive that pulls the generated content in.

Where prose lives
~~~~~~~~~~~~~~~~~

Member-level documentation — what a function does, what its parameters
mean, the invariants it preserves, threading notes — lives **next to
the code**: in Doxygen comments in the C++ header, or in Google-style
docstrings on the Python class or function. The corresponding ``.rst``
page (under ``doc/source/classes/`` for C++ classes, under
``doc/source/python/`` for Python entities) holds only:

* a 1–3 paragraph orientation intro that situates the class in the
  larger system, names the most relevant collaborators, and links
  outward to the user-guide and developer-guide chapters that cover
  the feature it supports,
* optional named subsections (H2, ``-`` underline) that group prose
  by topic when the orientation runs longer than three paragraphs
  (e.g. *Validation*, *System profiles*, *Registration macros*), and
* a final ``API Reference`` section (also H2, ``-`` underline) that
  contains the directive that pulls in the generated member
  documentation.

The source code is the single source of truth for member-level
documentation. Per-method ``///`` blocks (C++) or ``"""..."""``
docstrings (Python) describe what a function does, what its parameters
mean, what it returns, and which invariants it preserves; those blocks
are read by the generator, by IDE tooltips, by ``codebase-memory``,
and by any contributor opening the file. The ``.rst`` page must not
paraphrase those per-member blocks — that just creates two places to
keep in sync.

The class-level docstring or header block is governed by a different
rule: orientation prose lives on the ``.rst``, not next to the code.

* The class-level ``\brief`` (C++) or one-line summary (Python) stays
  tight: one or two sentences naming what the class is and its primary
  collaborators, optionally followed by *internals notes a code reader
  genuinely needs* — lifecycle invariants, ownership rules, threading
  contracts, configuration-flag fields, the cache or re-entrancy
  invariants a subclass author would otherwise miss.
* What does *not* belong in the class-level block: extended motivation
  prose, worked code examples (interface/implementation driver pairs,
  getter binding examples, friend-helper templates), enumerated lists
  of usage patterns, paragraph-form orientation for the class's role
  in the larger system. All of that lives on the ``.rst`` page or,
  where the topic spans multiple classes, in the developer guide.

The test for a class-level source-side sentence: would removing it
leave a contributor reading the source in isolation unable to use the
class correctly? If yes, keep it. If the sentence is structural
orientation that already appears (or could appear) on the ``.rst``
page, delete it from the source.

Doxygen comment style (C++)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Triple-slash** (``///``) on consecutive lines is preferred for new
  code. ``///<`` is used for trailing field/parameter comments.
  Multi-line ``/*! ... */`` blocks remain valid and are not converted
  for cosmetic reasons; match the surrounding file when editing.
* Use the standard tags: ``\brief``, ``\param``, ``\return``,
  ``\note``, ``\warning``, ``\sa``. Begin every documented entity
  with a single-sentence ``\brief``.
* Document every public and protected member that a subclass author
  or external caller would reach for. Default-implementation virtuals
  whose meaning is "do nothing" are still documented so subclasses
  know what they can override.
* The ``EXTRACT_ALL`` Doxygen setting means undocumented members
  still appear in the rendered output. Aim for zero undocumented
  public members in any class that has a dedicated API page.

Python docstring rendering
~~~~~~~~~~~~~~~~~~~~~~~~~~

Sphinx renders Python docstrings through two extensions configured in
``doc/source/conf.py``: ``sphinx.ext.autodoc`` introspects the module
to discover classes, methods, and attributes, and
``sphinx.ext.napoleon`` translates Google-style sections (``Args:``,
``Returns:``, ``Raises:``, ``Attributes:``, ``Example:``) into the
corresponding RST field lists before Sphinx parses the result.

The practical consequences:

* The docstring is the source of truth. Writing one Google-style
  docstring covers the in-source view (IDE tooltip, ``help()``
  output, GitHub source view) **and** the rendered API page
  simultaneously.
* Type hints in the function signature are picked up automatically by
  ``autodoc`` and rendered next to the parameter list. Repeating the
  type in the ``Args:`` description is redundant; describe the
  *meaning* of the parameter, not its type.
* RST inline markup inside the docstring works:
  ``:class:`~blackchirp.BCFid```, ``:math:`y = ax + b```,
  ``:meth:`~blackchirp.BCFTMW.get_fid```. ``napoleon`` does not
  rewrite these; it passes them through to Sphinx unchanged.
* ``:param:`` / ``:type:`` / ``:returns:`` field-list syntax works
  but is **discouraged**. Use Google-style sections instead — they
  are easier to read in source.

The ``.rst`` page for a Python class follows the same shape as a C++
class page: an orientation intro, optional named subsections, and a
final ``API Reference`` section with the autodoc directive. See
:doc:`/python/bcfid` for a representative example.

Sphinx directives
~~~~~~~~~~~~~~~~~

Pull generated content into the ``.rst`` page with the directive that
matches the entity:

**C++ (Doxygen / Breathe):**

.. code-block:: rst

   .. doxygenclass:: ClassName
      :members:
      :protected-members:
      :undoc-members:

   .. doxygenstruct:: StructName
      :members:

   .. doxygenenum:: EnumName

* Prefer ``.. doxygenclass::`` over ``.. doxygenfile::`` — one focused
  page per class, members grouped by member rather than by source
  position.
* Match the directive to the entity kind: using ``.. doxygenclass::``
  on a struct or enum produces empty output because Breathe looks up
  by exact compound kind.
* Drop ``:protected-members:`` for classes whose protected interface
  is not part of the contract (rare in this codebase — most base
  classes expose a protected hook layer that subclass authors are
  expected to override).

**Python (autodoc / napoleon):**

.. code-block:: rst

   .. autoclass:: blackchirp.BCFid
      :members:

   .. autofunction:: blackchirp.coaverage_fids

   .. automodule:: blackchirp._enum_helpers
      :members:

* Prefer ``.. autoclass::`` and ``.. autofunction::`` over
  ``.. automodule::`` for the public API. ``.. automodule::`` is
  appropriate for internal-helper modules whose entire surface is
  documentation-relevant.
* Use the fully-qualified import path (``blackchirp.BCFid``, not
  just ``BCFid``) so ``autodoc`` resolves the import unambiguously.

**Page placement.** Place every directive under the page's final
``API Reference`` section. Without this wrapper, sibling directives
appear visually nested under whatever prose subsection precedes them
in the page TOC.

**Cross-reference roles in prose:** ``:cpp:class:`` and ``:cpp:func:``
for C++ symbols; ``:class:``, ``:meth:``, ``:func:`` for Python
symbols (with the leading ``~`` to collapse the display name); ``:doc:``
for cross-page references; ``:ref:`` for labeled-section anchors.

Refresh checklist when editing a class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When changing a C++ header that has a corresponding API page:

1. Update the Doxygen comments alongside the code change.
2. Re-skim the ``.rst`` intro: if a collaborator referenced there has
   been renamed or removed, fix the prose.
3. Build the docs with ``cmake --build build --target docs`` and check
   ``doxygen.log`` for new warnings about undocumented public members
   of the touched class.

When changing a Python class or module that has a corresponding API
page:

1. Update the Google-style docstring alongside the code change. If
   the parameter list changed, the ``Args:`` section needs the same
   edits.
2. Re-skim the ``.rst`` intro: if a collaborator referenced there has
   been renamed or removed, fix the prose.
3. Run ``pytest`` from ``python/blackchirp/`` to confirm the docstring
   examples still execute (if any are doctest-style).
4. Build the docs and check the Sphinx output for new warnings about
   the touched module.

For both languages: if the change adds a new public member, ensure the
member is documented — ``EXTRACT_ALL`` (Doxygen) and ``:members:``
(autodoc) will both surface an undocumented member in the rendered
output.
