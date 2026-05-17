.. index::
   single: Crash Handler
   single: Crash Reports; triage
   single: Diagnostics; in-process handler
   single: Symbol Resolution

.. _crash-handling:

Crash Handling and Triage
=========================

Blackchirp installs an in-process crash handler at startup so that a
fault in a release build leaves a diagnostic artifact in the user's
data storage location rather than dying silently. This page explains
the handler's design, the on-disk artifact format, and the workflow
for resolving an artifact's stack-trace addresses to source-code
locations.

The user-facing description of where reports are stored and what they
contain is in :doc:`/user_guide/crash_reports`. This page is for
developers who triage incoming reports.

Handler Design
--------------

The handler lives in three files under ``src/data/``:

* ``crashhandler.h`` and ``crashhandler.cpp`` — the cross-platform
  public API and shared state (per-run filename, build-identity
  header, active experiment number).
* ``crashhandler_unix.cpp`` — POSIX implementation. Installs
  ``sigaction`` handlers for ``SIGSEGV``, ``SIGABRT``, ``SIGFPE``,
  ``SIGILL``, and ``SIGBUS`` on a ``sigaltstack`` so a stack overflow
  still has stack to run on. The handler writes a text crash log via
  raw ``write(2)`` calls; addresses are walked using ``std::stacktrace``
  when the standard library provides it (libstdc++ 13+) and fall back
  to ``backtrace(3)`` otherwise. Each frame is resolved through
  ``dladdr(3)`` to a ``module+0xoffset`` pair plus the absolute
  program counter.
* ``crashhandler_win.cpp`` — Windows implementation. Installs a
  top-level ``SetUnhandledExceptionFilter`` plus auxiliary handlers
  for the C runtime (``_set_invalid_parameter_handler``,
  ``_set_purecall_handler``, ``signal(SIGABRT, ...)``). On a fault
  the filter writes a minidump via ``MiniDumpWriteDump`` and a small
  text sidecar that mirrors the POSIX log header.

The handler is the documented exception to the
:doc:`bcLog/bcDebug logging convention <conventions>`: signal-handler
context cannot allocate, lock a Qt mutex, or touch the singleton
``LogHandler``. Output goes through file descriptors opened from
non-handler context (``CrashHandler::reopen``) before the handler ever
runs.

The handler is installed once from ``main.cpp`` immediately after
``QApplication`` construction so it covers as much of startup as
practical. ``CrashHandler::reopen(savePath)`` is called once the data
storage path is known, and again whenever the user changes the data
storage path on the Application Configuration dialog.
``CrashHandler::setActiveExperiment(num)`` is called from
``AcquisitionManager`` so the handler can record which experiment was
running at the time of a crash.

On a clean exit ``CrashHandler::shutdown()`` closes the open log
descriptor and unlinks the file if it is still empty, so a normal run
does not leave behind a stray zero-byte report.

On the next startup, ``CrashHandler::collectPriorArtifacts()``
enumerates the crash directory and returns any non-empty artifacts
that do not belong to the current process. Empty zero-byte files left
behind by an externally-killed prior run (``SIGKILL``, power loss) are
unlinked as a side effect.

Artifact Layout
---------------

Each crash artifact is a small text file under
``<savePath>/log/crashes/`` named:

.. code-block:: text

   crash-<UTC yyyyMMdd-HHmmss>-<short build SHA>.log

On Windows the same basename also gets a ``.dmp`` minidump alongside
the ``.log`` sidecar. The build SHA in the filename is the same one
embedded in the log header; this makes it easy to map a report to the
corresponding companion debug-info file without having to open the
file first.

The text portion is plain ASCII and is safe to pipe through standard
Unix tools:

.. code-block:: text

   Blackchirp 2.0.0-alpha (build 5c8837ede6aa82e4cece830dfe7d59a1bfafe799)
   Qt 6.11.0
   Crashed at 2026-05-08T02:59:14Z
   Signal: SIGSEGV (11) at address 0x0
   PID: 3355474
   Active experiment: 0

   Stack trace:
     ./blackchirp(+0x1bc972) [0x5bc972]
     /lib64/libQt6Core.so.6(+0x243a2d) [0x7f620cc43a2d]
     ...

Each frame shows the module path, the offset from the module's runtime
load base in parentheses, and the absolute program counter in square
brackets.

Symbol Resolution
-----------------

Release builds include debug info (``-g`` / ``/Zi``) so the addresses
in a crash log can be resolved against the same binary the user is
running, provided the developer has access to a binary built from
the matching git commit. Stripping happens at install time in ``cmake/Packaging.cmake``; the
unstripped binary is the artifact symbol-resolution tools consume.

Linux
~~~~~

Use ``addr2line`` against the unstripped ``blackchirp`` binary or the
companion ``.debug`` file:

.. code-block:: console

   $ addr2line -e blackchirp -f -C -i 0x5bc972 0x5bcb94 0x5bccd8
   (anonymous namespace)::emitStackTrace(int)
   /home/.../src/data/crashhandler_unix.cpp:108
   ...

For the **main blackchirp executable**, pass the absolute program
counter from the bracketed ``[0x...]`` value, not the parenthesized
``+0x...`` offset. The main executable is linked as a non-PIE ``EXEC``
binary, so its declared load base is non-zero and ``addr2line``
expects the absolute VMA.

For shared libraries (Qt, glibc, Qwt), pass the parenthesized
``+0x...`` offset. Shared libraries are PIE, so the offset is relative
to the runtime load base and is what ``addr2line`` accepts directly.

macOS
~~~~~

Use ``atos`` against the matching ``.dSYM`` bundle:

.. code-block:: console

   $ atos -o blackchirp.dSYM/Contents/Resources/DWARF/blackchirp \
          -l 0x100000000 0x100123456 0x100234567

The ``-l`` flag passes the load address; for the main executable it is
typically ``0x100000000`` on Apple silicon. Read the load address from
the system crash report or the artifact's bracketed program counters.

Windows
~~~~~~~

Open the ``.dmp`` file in WinDbg or Visual Studio with the matching
``.pdb`` on the symbol path. The ``.dmp`` carries the full process
state (data segments, thread info, process thread data) so register
contents and local variables are available, not just the stack trace.

The text sidecar mirrors the POSIX log header so the same triage steps
(matching the build SHA to a stored ``.pdb``) apply.

Build SHA Lookup
----------------

The build identifier in the log header is the git commit SHA the
binary was compiled from, captured into ``BC_BUILD_VERSION`` by the
top-level ``CMakeLists.txt``. To check out the corresponding source
tree:

.. code-block:: console

   $ git checkout 5c8837ede6aa82e4cece830dfe7d59a1bfafe799
   $ cmake -B build -DCMAKE_BUILD_TYPE=Release .
   $ cmake --build build -j

The freshly-built unstripped ``blackchirp`` binary in ``build/`` can
then be passed to ``addr2line`` to resolve the crash report.

Fetching CI symbol artifacts
----------------------------

Each CI run that produces a release binary also captures companion
debug-info files into a separate ``blackchirp-symbols-<platform>``
workflow artifact. This is the preferred symbol source for crashes
against shipped builds: the artifact is built from the same commit
the user is running, with the same compiler and flags, so the
addresses in the crash log line up exactly.

Symbol artifacts are workflow artifacts, **not** release assets:
they are bulky and contain enough information to reverse-engineer
the application internals. Retention is 90 days from the workflow
run date.

The release-build workflow is ``.github/workflows/release.yml`` in
the source repository. Each platform job adds a ``Capture symbols``
step after ``cmake --build`` that runs the platform-appropriate
extraction tool — ``objcopy --only-keep-debug`` (Linux),
``dsymutil`` (macOS), or copies the ``.pdb`` files MSVC has already
written alongside the ``.exe`` (Windows) — and uploads the result
as a separate artifact. Each artifact also contains a
``symbols-manifest.json`` listing the platform tag, the workflow's
git SHA, the run ID, and a list of ``{name, sha256}`` entries so a
triager can verify they downloaded the right artifact.

To fetch:

.. code-block:: console

   $ git_sha=$(sed -n 's/.*(build \([a-f0-9]\+\)).*/\1/p' crash.log | head -1)
   $ run_id=$(gh run list --workflow=release.yml --commit=$git_sha \
                          --json databaseId --jq '.[0].databaseId')
   $ gh run download $run_id --name blackchirp-symbols-<platform>

Where ``<platform>`` is one of ``linux-deb``, ``linux-rpm``,
``linux-appimage``, ``macos-arm64``, ``macos-x86_64``, or
``windows`` (the macOS build is an ``arm64`` / ``x86_64`` matrix, so
its symbol artifact is per-architecture; match the slice the user is
running). The downloaded
files plug directly into the resolution steps above (Linux:
``addr2line -e <basename>.debug``; macOS: ``atos -o
<basename>.dSYM/Contents/Resources/DWARF/<binary>``; Windows: open
the ``.dmp`` in WinDbg with ``<basename>.pdb`` on the symbol path).
