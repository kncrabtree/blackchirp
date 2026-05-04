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
   single: Doxygen comments
   single: logging
   single: bcLog
   single: hwLog
   single: persistent settings
   single: SettingsStorage; conventions

Coding Conventions
==================

This page is the quick reference for *what Blackchirp code looks like*
once you sit down to add or modify a file. It collects the conventions
that span the codebase — naming, member prefixes, string-literal
selection, function-signature policy, container choices, key declaration
patterns, Doxygen comments, logging, and persistent settings — and
points outward at the API pages that carry the per-class detail. It is
not a tutorial on Qt6; it assumes the reader is fluent with ``QObject``,
the metaobject system, ``QString``/``QAnyStringView``, and the rest of
the standard Qt vocabulary.

Documentation comments themselves have their own contract. The Doxygen
section below is short on purpose: it names the tags Blackchirp uses
and forwards to :doc:`/developer_guide/api_style`, which is the
canonical reference for *where* prose lives (header vs ``.rst``), the
Sphinx directive choices, and the refresh checklist for editing a class
that has an API page.

Naming and indentation
----------------------

* Classes, structs, and enums use ``UpperCamelCase``: ``HardwareObject``,
  ``FtmwConfig``, ``CommunicationProtocol::CommType``.
* Free functions and variables use ``lowerCamelCase``: ``readSettings()``,
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
---------------

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
(see the *Key declaration patterns* section below) so the file does
not depend on the suffixes being visible.

Function signatures
-------------------

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
----------

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
------------------------

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

Doxygen comments
----------------

Two paragraphs of style notes here; the per-class refresh procedure
lives on :doc:`/developer_guide/api_style`.

* Triple-slash (``///``) on consecutive lines is preferred for new code;
  ``///<`` for trailing field, parameter, or enumerator comments.
  Multi-line ``/*! ... */`` blocks remain valid and are not converted
  for cosmetic reasons; match the surrounding file when editing.
* Use the standard tags: ``\brief``, ``\param``, ``\return``,
  ``\note``, ``\warning``, ``\sa``. Begin every documented entity with
  a single-sentence ``\brief``. Document every public and protected
  member that a subclass author or external caller would reach for —
  including default-implementation virtuals whose body is "do nothing",
  because subclasses still need to know what they may override.

The canonical reference is :doc:`/developer_guide/api_style`. That
page sets out where prose lives (header vs ``.rst``), the Sphinx
directive choices (``.. doxygenclass::`` over ``.. doxygenfile::``,
``.. doxygenstruct::`` for structs, ``.. doxygenenum::`` for enums),
and the checklist for editing a class that has a corresponding API
page. Do not duplicate it here.

Logging
-------

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
-------------------

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
for a driver, ``REGISTER_HARDWARE_BASE`` for shared
base-class defaults — and applied by
``HardwareObject::applyRegisteredSettings()`` during construction. The
registry is also what populates the hardware settings dialog without
the dialog ever instantiating a driver. The full registration story
and the rest of the runtime configuration model are covered in
:doc:`/developer_guide/hardware_configuration`. The registration macros
themselves and the descriptor structs they produce are documented on
:doc:`/classes/hardwareregistry`; the ``SettingsStorage`` API surface
itself, including the array, group, getter, and discard mechanisms, is
documented on :doc:`/classes/settingsstorage`.
