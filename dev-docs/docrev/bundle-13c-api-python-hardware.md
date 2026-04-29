# Bundle 13c — API Reference: Python Hardware Classes

Adds API reference pages for the Python hardware C++ side.

## Scope

New pages under `doc/source/classes/`:

- `pythonprocess.rst` ← `src/hardware/python/pythonprocess.h`
  (`PythonProcess`).
- `pythonhardwarebase.rst` ←
  `src/hardware/python/pythonhardwarebase.h`
  (`PythonHardwareBase`).

Per-trampoline classes (`PythonAwg`, `PythonClock`,
`PythonFlowController`, etc.) are intentionally not given
individual pages here. They have nearly identical surface area;
documenting `PythonHardwareBase` plus the per-type capability table
in bundle 06 covers the audience that needs to know. If a future
maintainer wants per-trampoline pages, they can be added without
disturbing this bundle.

Header refresh is mandatory for these two: the trampoline contract
and the IPC protocol are non-obvious and the headers should
explain enough to read them in isolation.

## Out of scope

- Per-trampoline class pages (deferred; see above).
- The Python-side host script and proxies (those are described in
  the user guide bundle 06 and the developer guide bundle 12).

## Sources

- `dev-docs/python-hardware.md` — primary; specifically the
  Trampoline Implementation Contract section.
- `dev-docs/python-process-push-refactor.md` — for the
  `setEnabledProxies`/`waveformReceived` surface on
  `PythonProcess`.
- The two header files.

## Sphinx file deltas

**Created:**
- `doc/source/classes/pythonprocess.rst`
- `doc/source/classes/pythonhardwarebase.rst`

**Possibly modified (Doxygen comment refresh):**
- `src/hardware/python/pythonprocess.h`
- `src/hardware/python/pythonhardwarebase.h`

## Toctree delta

`classes.rst` uses `:glob:`. No edit needed.

## Acceptance criteria

- The `PythonProcess` page documents the IPC protocol shape, the
  selective proxy injection surface, and the `waveformReceived`
  push signal.
- The `PythonHardwareBase` page documents the
  `initPythonProcess` / `testPythonConnection` /
  `startPythonProcess` / `findHostScript` /
  `pythonForbiddenKeys` / `resolvePythonExecutable` surface and
  the inheritance pattern (mixin alongside the hardware base
  class).
- Both pages cross-link to the user-guide Python hardware chapter
  (bundle 06) and the developer-guide Python hardware sub-page
  (bundle 12).
