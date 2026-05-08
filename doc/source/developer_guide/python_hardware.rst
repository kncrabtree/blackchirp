.. index::
   single: Python hardware; developer guide
   single: PythonHardwareBase; developer guide
   single: PythonProcess; developer guide
   single: JSON IPC
   single: subprocess hardware
   single: trampoline; Python
   single: relay request
   single: proxy injection
   single: scope proxy
   single: waveform push
   single: state-management patterns; A/B/C
   single: hot reload; Python script
   single: Python environment; per-profile

Python Hardware
===============

Blackchirp lets contributors and users write hardware drivers in
Python without modifying or recompiling the application. From the
contributor's side, a Python driver is a *trampoline*: a C++
class that inherits from one of the hardware base classes
(:cpp:class:`AWG`, :cpp:class:`Clock`, :cpp:class:`FlowController`,
:cpp:class:`FtmwScope`, …) and from :cpp:class:`PythonHardwareBase`
via multiple inheritance. The hardware base supplies the Qt
slot/signal API the rest of Blackchirp consumes; the mixin owns a
child Python interpreter, a :cpp:class:`PythonProcess` that talks to
it over JSON-lines IPC, and the lifecycle plumbing that ties the
two together. Each pure virtual on the hardware base is reimplemented
as a JSON method dispatch through ``pu_process->sendRequest()``.

This page documents what a contributor needs to add new C++ behavior
to the Python hardware stack: the IPC architecture, the proxy
injection model, the three state-management patterns the trampolines
fall into, and the recipes for adding a new trampoline class or a
new push-style proxy. The user-facing perspective — writing a
driver script, picking a profile, hot-reloading from the UI — lives
on :doc:`/user_guide/python_hardware` and its sub-pages, and the
class-level API is on :doc:`/classes/pythonhardwarebase` and
:doc:`/classes/pythonprocess`.

Subprocess and JSON IPC
-----------------------

A Python trampoline does not embed an interpreter. Each instance
launches a fresh ``python3`` (or per-profile environment, see
:ref:`python-hw-env-support`) under a Qt :cpp:class:`QProcess`
running ``python_hw_host.py``, which loads the user driver, injects
a small set of proxy objects onto it, and dispatches calls received
on stdin. The Python heap therefore lives in a separate OS process,
so a script crash cannot corrupt Qt state, no GIL ever touches the
Blackchirp main thread, and the C++ side has no compile-time
dependency on a particular Python version. The cost is one IPC
round trip per call, on the order of a millisecond, which is
negligible against typical instrument I/O latencies of tens to
hundreds of milliseconds.

The wire format is one compact JSON object per line in each
direction. Four message kinds travel over the channel:

- **Method calls (C++ → Python).** Carry an integer ``id`` and a
  ``method`` name; any other keys are forwarded as keyword
  arguments to the snake_case method on the user driver. Responses
  carry the same ``id`` and either ``result`` on success or
  ``error`` plus ``traceback`` on failure. The host script's
  generic dispatch means adding a new hardware-specific method
  requires no host-script change — declaring a Python method with
  the same name as the trampoline's IPC payload is enough.
- **Relay requests (Python → C++, interleaved).** The host script
  uses these to reach back through the C++ side for services it
  cannot perform itself: ``self.comm.query``/``write``/``read_bytes``
  are relayed as ``"relay": "comm_query"`` (etc.) and serviced
  against the trampoline's :cpp:class:`CommunicationProtocol`;
  ``self.settings.get``/``set`` are relayed against the
  :cpp:class:`SettingsStorage` callbacks the mixin installs.
- **Log messages (Python → C++, unsolicited).** Lines containing
  ``"log"`` and ``"level"`` are forwarded to ``bcLog()`` with the
  appropriate severity, so script output flows into the hardware
  log panel beside C++ driver output without per-trampoline wiring.
- **Waveform pushes (Python → C++, unsolicited).** Push-style
  hardware (the digitizer trampolines) sends raw shot data as
  ``"waveform": "<base64>"`` with a ``shots`` count; the bytes
  decode and surface on the
  :cpp:func:`PythonProcess::waveformReceived` signal for the
  trampoline to drain into the :cpp:class:`WaveformBuffer`.

PythonProcess push model
------------------------

:cpp:class:`PythonProcess` is the C++-side endpoint of the IPC
channel. It owns the :cpp:class:`QProcess`, holds the
:cpp:class:`CommunicationProtocol` pointer used to service relay
requests, and exposes the synchronous
:cpp:func:`PythonProcess::sendRequest` API.

Reads are event-driven. ``QProcess::readyReadStandardOutput`` is
connected to ``onReadyRead``, which appends to a buffer, splits on
``\n``, and dispatches each complete JSON line by message kind:

.. code-block:: text

   line contains "log"      → emit bcLog at the parsed severity
   line contains "waveform" → base64-decode, emit waveformReceived
   line contains "relay"    → handle relay, write response back
   line contains "id"       → store, emit responseReady

``sendRequest`` does not poll. It writes the request, sets up the
expected ``id``, and runs a nested :cpp:class:`QEventLoop` until
the matching response arrives, the subprocess dies, or the
configured timeout (30 s by default) fires:

.. code-block:: cpp

   QEventLoop loop;
   connect(this, &PythonProcess::responseReady,
           &loop, &QEventLoop::quit);
   connect(p_process, &QProcess::finished,
           &loop, [this, &loop]() {
               onReadyRead();   // drain remaining stdout
               loop.quit();
           });
   QTimer::singleShot(d_timeoutMs, &loop, &QEventLoop::quit);
   loop.exec();

Because the loop continues to process events, relay requests, log
messages, and waveform pushes are all dispatched correctly while a
method call is in flight. ``readyReadStandardOutput`` may deliver
partial lines, so the buffer accumulates bytes until a newline
appears.

**Reentrancy contract.** The nested event loop fires
``waveformReceived`` from inside ``sendRequest``. The trampoline's
slot must therefore not re-enter ``sendRequest``: the only safe
operation is to forward the bytes into the
:cpp:class:`WaveformBuffer`. ``PythonFtmwScope::onWaveformReceived``
calls :cpp:func:`FtmwScope::emitShot` and returns; that is the
shape every push-style handler must follow. Reentering
``sendRequest`` from a slot the nested loop has dispatched
serializes incorrectly with the in-flight call and is undefined.

Proxy injection
---------------

The host script attaches proxy objects to the user driver before
its :meth:`initialize` runs. The three standard proxies
(``self.comm``, ``self.settings``, ``self.log``) are injected on
every ``_init`` and are always available. Optional,
hardware-type-specific proxies are gated.

The trampoline opts in by calling
:cpp:func:`PythonProcess::setEnabledProxies` between
:cpp:func:`PythonHardwareBase::initPythonProcess` and the first
:cpp:func:`PythonProcess::sendRequest`. The chosen names ride on
the ``_init`` payload:

.. code-block:: json

   {"method": "_init", "key": "...", "model": "...",
    "proxies": ["scope"]}

On the Python side, the host script keeps a factory map and only
instantiates the proxies the C++ side requested:

.. code-block:: python

   _OPTIONAL_PROXY_FACTORIES = {
       "scope": ScopeProxy,
   }
   for name in request.get("proxies", []):
       factory = _OPTIONAL_PROXY_FACTORIES.get(name)
       if factory:
           setattr(user_obj, name, factory())

The single optional proxy that ships today is ``ScopeProxy``, which
push-streams base64-encoded waveforms to C++:

.. code-block:: python

   class ScopeProxy:
       def emit_shot(self, raw_bytes, shots=1):
           b64 = base64.b64encode(bytes(raw_bytes)).decode('ascii')
           _send_json({"waveform": b64, "shots": shots})

``_send_json`` holds an internal stdout lock, so ``emit_shot`` is
safe to call from the driver's acquisition thread.

Adding a new push-style proxy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a push channel for a new hardware kind:

1. Implement the proxy class on the Python side in
   ``python_hw_host.py``. The proxy's job is to package a payload
   and call ``_send_json`` with a unique top-level key.
2. Add the class to ``_OPTIONAL_PROXY_FACTORIES`` under that key
   (lowercase string, matching what the trampoline will request).
3. In :cpp:func:`PythonProcess::onReadyRead`, add a branch that
   matches the new key, parses the payload, and emits a
   trampoline-facing :cpp:class:`Q_SIGNAL`. Follow the
   ``waveform`` branch as the model: decode, then ``emit``.
4. In the trampoline, call
   ``pu_process->setEnabledProxies({"yourproxy"})`` from
   ``initialize`` (or the type-specific helper virtual)
   immediately after :cpp:func:`PythonHardwareBase::initPythonProcess`
   returns, and before any :cpp:func:`PythonProcess::sendRequest`
   call would force the subprocess to start.
5. Connect the new signal to a handler slot in the trampoline.
   The slot follows the reentrancy contract: it dispatches the
   payload synchronously and never re-enters ``sendRequest``.

PythonHardwareBase mixin
------------------------

:cpp:class:`PythonHardwareBase` carries the boilerplate every
trampoline needs: the owned subprocess, the lazy-start hook, the
helpers that turn :cpp:func:`HardwareObject::sleep` and the
trampoline's read-settings hook (:cpp:func:`HardwareObject::hwReadSettings`
or the per-base variant on the intermediate bases that
``final``-override ``hwReadSettings`` — ``fcReadSettings``,
``pcReadSettings``, ``tcReadSettings``, ``pgReadSettings``,
``awgReadSettings``, ``clockReadSettings``, ``ftmwReadSettings``,
``gpibReadSettings``, ``ioReadSettings``, ``lifLaserReadSettings``,
``lifScopeReadSettings``) into
IPC dispatches, and the static helpers that find the host script
and resolve a Python interpreter. The mixin's constructor takes the hardware key and
model strings and stores them; it does not need a back-pointer to
the :cpp:class:`HardwareObject`, because the IPC and the settings
relay are funneled through callbacks the trampoline installs.

The members the mixin owns and exposes to subclasses:

- ``pu_process`` — the :cpp:class:`PythonProcess` instance, owned
  by ``std::unique_ptr``. Non-null after
  ``initPythonProcess`` returns; the subprocess inside it is
  started lazily on the first ``testPythonConnection``.
- :cpp:func:`PythonHardwareBase::initPythonProcess` — constructs
  ``pu_process``, binds the comm pointer, and installs the
  settings get/set callbacks. The callbacks are typically lambdas
  capturing :cpp:func:`SettingsStorage::get` and
  :cpp:func:`SettingsStorage::set` on the trampoline. The setter
  lambda is the bridge that lets the script update persistent
  settings across the relay, since
  :cpp:func:`SettingsStorage::set` is protected and otherwise
  inaccessible from outside the owning class.
- :cpp:func:`PythonHardwareBase::testPythonConnection` — lazily
  starts the subprocess via ``startPythonProcess()``, refreshes
  the comm pointer (in case the protocol has been swapped since
  the previous call), then sends ``test_connection`` and returns
  the boolean result.
- :cpp:func:`PythonHardwareBase::startPythonProcess` — looks up
  the per-profile ``pythonScriptPath``, ``pythonClassName``, and
  ``pythonEnvPath`` on :cpp:class:`HardwareProfileManager`,
  resolves the interpreter, and delegates to
  :cpp:func:`PythonProcess::start`. Refuses to start (and sets
  :cpp:func:`PythonHardwareBase::pythonErrorString`) if the
  script path or the class name is empty rather than substituting
  a default.
- :cpp:func:`PythonHardwareBase::findHostScript` — locates
  ``python_hw_host.py`` in the application directory or in the
  ``share/blackchirp/`` install location. Returns an empty string
  if neither exists.
- :cpp:func:`PythonHardwareBase::resolvePythonExecutable` —
  probes ``envPath`` for the standard venv and conda layouts
  (``bin/python3``, ``bin/python``, ``Scripts/python.exe``) and
  falls back to the literal ``"python3"`` (resolved through
  ``PATH``) when ``envPath`` is empty or contains no interpreter.
- :cpp:func:`PythonHardwareBase::pythonSleep` and
  :cpp:func:`PythonHardwareBase::pythonReadSettings` — the
  trampoline's :cpp:func:`HardwareObject::sleep` override and its
  read-settings hook (:cpp:func:`HardwareObject::hwReadSettings`
  for trampolines that inherit the default, or the per-base variant —
  ``fcReadSettings``, ``pcReadSettings``, ``tcReadSettings``,
  ``pgReadSettings``, ``awgReadSettings``, ``clockReadSettings``,
  ``ftmwReadSettings``, ``gpibReadSettings``, ``ioReadSettings``,
  ``lifLaserReadSettings``, or ``lifScopeReadSettings`` — when the
  parent base ``final``-overrides ``hwReadSettings``) delegate to these helpers.
  ``pythonReadSettings`` deliberately sends ``read_settings`` over
  IPC rather than restarting the subprocess, because a restart
  would re-run :meth:`initialize` and disrupt connected state.
- :cpp:func:`PythonHardwareBase::pythonErrorString` — exposes the
  human-readable error from the most recent failed
  ``startPythonProcess`` or ``testPythonConnection``. Trampolines
  copy it into ``d_errorString`` on failure so the connection
  result reaches the GUI with a useful message.
- The destructor stops ``pu_process`` if the subprocess is
  running.

A concrete trampoline therefore looks like a thin layer on top of
the mixin: an ``initialize`` hook that calls ``initPythonProcess``,
a ``testConnection`` hook that calls ``testPythonConnection``,
delegations to the sleep helper and the appropriate read-settings
hook helper, and one IPC dispatch per hardware-specific virtual.

.. _python-hw-state-patterns:

Three state-management patterns
-------------------------------

How a trampoline implements its hardware-specific virtuals depends
on how its hardware base class manages config state. The three
patterns below cover every trampoline that ships today.

Pattern A — Bulk Configure
~~~~~~~~~~~~~~~~~~~~~~~~~~

The base class **inherits** from a complex config class (a
:cpp:class:`DigitizerConfig` with channel maps, trigger settings,
sample rates, multi-record state, and so on). Per-call setters do
not exist; the experiment hands the trampoline a desired config,
and the trampoline applies it in one shot via a virtual
``configure`` (or, for :cpp:class:`FtmwScope`, an override of
:cpp:func:`HardwareObject::prepareForExperiment`). The trampoline
serializes the full config to JSON, sends a ``configure`` IPC call,
parses a validated config back from the response, and copies it
onto the C++ object so any clamped or substituted values persist.
The Python side decides which keys to clamp.

Examples: :cpp:class:`PythonIOBoard` overriding
:cpp:func:`IOBoard::configure`,
:cpp:class:`PythonLifScope` overriding
:cpp:func:`LifScope::configure`. :cpp:class:`FtmwScope` does **not**
expose a ``configure`` virtual; each subclass overrides
:cpp:func:`HardwareObject::prepareForExperiment` directly.
:cpp:class:`PythonFtmwScope` follows that convention: it overrides
``prepareForExperiment`` to serialize the
:cpp:class:`FtmwDigitizerConfig` over JSON IPC, then leaves the
final ``hwPrepareForExperiment`` in :cpp:class:`FtmwScope` to
construct the :cpp:class:`WaveformBuffer` from the validated
config in the usual way. No bespoke buffer wiring is needed in
the trampoline.

Pattern B — Granular Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base class **contains** a config object as a member and
exposes per-channel or per-parameter getter/setter slots. Each
slot delegates to a ``hw*`` pure virtual that the trampoline
implements. The base class owns the polling sequence, the validity
checks, and the signal emission; the trampoline only ever sees one
value at a time.

Examples: :cpp:class:`PythonFlowController` (eight
``hw*`` reads/writes for per-channel flow and pressure, plus the
control mode), :cpp:class:`PythonPulseGenerator` (~22 setters and
readers across channel and global state),
:cpp:class:`PythonTemperatureController`,
:cpp:class:`PythonPressureController`. The trampoline does not
override the polling cadence — :cpp:func:`FlowController::poll` is
non-virtual, and the IPC round trip per ``hw*`` call is the right
granularity for the slow serial links these instruments typically
sit behind.

Pattern C — Stateless / Pass-Through
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base class has no internal config object to manage. The
trampoline receives experiment-supplied data (chirp segments and
markers for an AWG; frequency assignments for a clock) at
:cpp:func:`HardwareObject::prepareForExperiment` time, serializes
it into JSON, and hands it to the Python script. There is no
``configure`` virtual and no bulk read-back: the experiment data
is the truth, and the script programs the hardware to match.

Examples: :cpp:class:`PythonAwg` (serializes
:cpp:class:`ChirpConfig` segments, markers, RF chain parameters,
and clock assignments), :cpp:class:`PythonClock` (per-output
``hw_set_frequency``/``hw_read_frequency``).

The mapping from each ``Python*`` trampoline that ships with
Blackchirp to its pattern, default driver class name, and
hardware-specific entry points lives in the user-guide table at
:ref:`python-hardware-trampoline-overview`; the companion sections
on :doc:`/user_guide/python_hardware/per_type_capabilities` walk
through the per-method signatures.

Trampoline implementation contract
----------------------------------

A new ``Python<Type>`` trampoline follows a fixed recipe.

1. **Inherit from both bases.** Multiple-inherit from the hardware
   base class and from :cpp:class:`PythonHardwareBase`. Initialize
   the mixin in the constructor's initializer list with
   ``d_key`` and ``d_model``:

   .. code-block:: cpp

      PythonFlowController::PythonFlowController(const QString &label, QObject *parent) :
          FlowController(QString(PythonFlowController::staticMetaObject.className()),
                         label, parent),
          PythonHardwareBase(d_key, d_model)
      { d_threaded = true; save(); }

   Set ``d_threaded = true`` so the device runs on its own
   :cpp:class:`QThread`; the IPC round trips would otherwise block
   the hardware-manager thread.

2. **Wire the mixin in the type-specific initialize hook.** For a
   plain :cpp:class:`HardwareObject` subclass that is
   :cpp:func:`HardwareObject::initialize`; for hardware bases with
   a typed helper virtual it is the helper (``initializeClock``,
   ``fcInitialize``, ``initializePGen``, …):

   .. code-block:: cpp

      void PythonFlowController::fcInitialize()
      {
          initPythonProcess(p_comm,
              [this](const QString &k, const QVariant &dv) -> QVariant {
                  return get(k, dv);
              },
              [this](const QString &k, const QVariant &v) {
                  set(k, v, true);
              });
      }

   ``initPythonProcess`` does **not** start the subprocess. The
   profile-creation flow runs the registry-driven default-settings
   pass through every :cpp:class:`HardwareObject` constructor, and
   spawning a Python interpreter on every dialog open would be
   gratuitous.

3. **Drive the test-connection hook.** Call
   ``testPythonConnection(p_comm)`` from
   :cpp:func:`HardwareObject::testConnection` (or its typed helper).
   The first call starts the subprocess; subsequent calls reuse it:

   .. code-block:: cpp

      bool PythonFlowController::fcTestConnection()
      {
          if (!testPythonConnection(p_comm)) {
              d_errorString = pythonErrorString();
              return false;
          }
          return true;
      }

4. **Delegate sleep and the read-settings hook.** Override
   :cpp:func:`HardwareObject::sleep` to call
   :cpp:func:`PythonHardwareBase::pythonSleep`, and override the
   appropriate read-settings hook to call
   :cpp:func:`PythonHardwareBase::pythonReadSettings`. The hook is
   :cpp:func:`HardwareObject::hwReadSettings` for trampolines whose
   parent base does not ``final``-override it, and the per-base
   variant (``fcReadSettings``, ``pcReadSettings``, ``tcReadSettings``,
   ``pgReadSettings``, ``awgReadSettings``, ``clockReadSettings``,
   ``ftmwReadSettings``, ``gpibReadSettings``, ``ioReadSettings``,
   ``lifLaserReadSettings``, or ``lifScopeReadSettings``) for the
   intermediate bases that do.

5. **Implement the hardware-specific virtuals as IPC dispatches.**
   The pattern from :ref:`python-hw-state-patterns` chooses the
   shape:

   - **Pattern A** — implement the ``configure(...)`` virtual.
     Serialize the full config to JSON, send a ``configure``
     IPC call, deserialize the validated dict from
     ``result.config`` back onto the C++ side.
     :cpp:class:`PythonFtmwScope` overrides
     :cpp:func:`HardwareObject::prepareForExperiment` directly
     because :cpp:class:`FtmwScope` does not expose a
     ``configure`` virtual.
   - **Pattern B** — implement each ``hw*`` pure virtual as a
     standalone :cpp:func:`PythonProcess::sendRequest` call.
     The C++ side keeps the config object;
     :cpp:func:`PythonProcess::sendRequest` returns the
     ``result`` on success and an ``error``-bearing object on
     failure, so each method ends with a default-value fall-through
     for the failure case.
   - **Pattern C** — typically only an override of
     :cpp:func:`HardwareObject::prepareForExperiment` plus any
     per-output IPC calls (clock frequencies, AWG markers).

6. **Push-style hardware: enable the optional proxy.** For
   digitizer trampolines, after ``initPythonProcess`` returns:

   .. code-block:: cpp

      pu_process->setEnabledProxies({"scope"_L1});
      connect(pu_process.get(), &PythonProcess::waveformReceived,
              this, &PythonFtmwScope::onWaveformReceived);

   The handler slot honors the reentrancy contract from
   :ref:`python-hw-push-reentrancy` — it forwards the bytes to
   :cpp:func:`FtmwScope::emitShot` (FTMW) or to
   :cpp:func:`LifScope::emitWaveform` (LIF) and returns.

7. **Register the class.** :cpp:any:`REGISTER_HARDWARE_META`,
   :cpp:any:`REGISTER_HARDWARE_PROTOCOLS`, and any
   :cpp:any:`REGISTER_HARDWARE_SETTINGS` /
   :cpp:any:`REGISTER_HARDWARE_ARRAY` declarations go in the
   ``.cpp`` next to the trampoline. The protocol set is
   ``Rs232 + Tcp + Gpib + Custom + Virtual``, omitting any value
   that is meaningless for the hardware
   (:cpp:class:`PythonGpibController` omits ``Gpib`` because it
   *is* the GPIB controller). ``Custom`` is the explicit "comm is
   handled by the .py script" indicator: the Python trampolines
   do not register :cpp:any:`REGISTER_CUSTOM_COMM` field
   descriptors, and the user-facing
   :cpp:class:`CustomProtocolWidget` detects the driver
   prefix and shows a note rather than a generic empty-fields
   form.

8. **Ship a template script.** Add
   ``python_<type>_template.py`` next to the trampoline source.
   The template defines a single class — ``AwgDriver``,
   ``FtmwScopeDriver``, etc. — that works out of the box on the
   ``Virtual`` protocol and documents every method. The host
   script's generic-keyword dispatch means the template is the
   contract the user code must satisfy; no host-side change is
   needed when adding a new hardware type.

9. **CMake wires up automatically.** The hardware globs in
   ``cmake/BlackchirpHardware.cmake`` pick up
   ``python*.cpp``/``.h`` and ``python_*_template.py`` without
   editing the build file. Trampoline headers are appended to
   ``HARDWARE_IMPLEMENTATION_HEADERS`` so AUTOMOC generates the
   meta-object code for each ``Q_OBJECT`` and the static
   registration initializers are pulled out of the static library
   at link time (without that step the registrations are silently
   dropped).

.. _python-hw-push-reentrancy:

Reentrancy contract for push handlers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A push-style trampoline's
:cpp:func:`PythonProcess::waveformReceived` slot must not call
:cpp:func:`PythonProcess::sendRequest`. The signal fires from
inside the nested :cpp:class:`QEventLoop` of an in-flight
``sendRequest``; a recursive call would serialize incorrectly
with the pending response. The slot's only legitimate work is to
forward the bytes into the :cpp:class:`WaveformBuffer`.

When a method call is in flight, the C++ side is sitting inside
``loop.exec()`` waiting for the matching ``id``. The event loop
dispatches every other message kind — relay, log, waveform — to
its handler slot. If a waveform handler issued another
``sendRequest`` it would push a second nested loop on top of the
first, and the second loop's response would be the one the outer
loop unblocks on. That is undefined behavior. The trampolines
today obey the contract by writing only to the buffer.

QSettings key paths
-------------------

Every persistent setting for a hardware object — the ``commType``
chosen at profile creation, every Required/Important/Optional
value declared by :cpp:any:`REGISTER_HARDWARE_SETTINGS`, and any
custom-comm parameters — lives directly under the
``<hwType>:<label>`` ``QSettings`` group. That group **is** the
:cpp:class:`SettingsStorage` root for the
:cpp:class:`HardwareObject`. There is no driver-name
subgroup, no ``configParams`` subtree, and no two-layer fallback.
A Python trampoline reads and writes settings through the same
:cpp:func:`SettingsStorage::get` / :cpp:func:`SettingsStorage::set`
calls that any other :cpp:class:`HardwareObject` subclass uses;
the relay across ``self.settings`` reaches that root from the
script side.

Required parameters that used to flow through the older
``HwConfigParam`` system now live in the same group as ordinary
settings, declared with ``HwSettingPriority::Required``. The
settings registry is the single source of truth; see
:doc:`/developer_guide/hardware_configuration` for the full
declaration model and the create-vs-edit semantics that determine
when Required parameters are writable. A common slip is to write
``commType`` or a Required parameter under
``<hwType>:<label>/<impl>/...``: that path is wrong, it shadows
nothing on read, and the value the trampoline actually sees is
whatever default the registry pass installed.

Hot reload
----------

Reloading a script is a subprocess operation, not a
hardware-object operation. The user clicks **Reload Script** on
:cpp:class:`PythonHardwareControlWidget`, which routes the
request through
:cpp:func:`HardwareManager::reloadPythonScript` to the device's
thread; the manager runs a single lambda there that calls
:cpp:func:`PythonProcess::stop` and immediately
:cpp:func:`HardwareObject::bcTestConnection`. ``bcTestConnection``
calls back through ``startPythonProcess``, which launches a fresh
subprocess and re-runs the ``_init`` → :meth:`initialize` →
``test_connection`` handshake. Everything on the C++ side — the
:cpp:class:`SettingsStorage` cache, the
:cpp:class:`CommunicationProtocol`, the
:cpp:class:`QThread` and signal connections — is untouched; only
the subprocess is replaced. The user-facing flow and what does
*not* survive a reload are documented on
:doc:`/user_guide/python_hardware/hot_reload`. The runtime-side
of the reload signal (the
:cpp:func:`HardwareManager::pythonScriptReloadResult` reporting
hop) is on :doc:`/developer_guide/hardware_runtime`.

.. _python-hw-env-support:

Per-profile Python environment
------------------------------

Each Python profile may carry an optional ``pythonEnvPath`` field
on :cpp:class:`HardwareProfileManager` pointing at a venv or
conda environment directory.
:cpp:func:`PythonHardwareBase::resolvePythonExecutable` probes
that directory for ``bin/python3``, ``bin/python``, and
``Scripts/python.exe`` in order; an empty path falls back to
``"python3"`` (resolved through ``PATH``). This lets a script
that depends on a vendor SDK ship with the SDK installed in a
dedicated environment without polluting the system Python — the
trampoline launches the user's interpreter and the user's Python
package set without further configuration.
