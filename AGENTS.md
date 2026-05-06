# Blackchirp — Agent Guide

Open-source data acquisition software for chirped-pulse Fourier transform
microwave (CP-FTMW) spectroscopy. Qt6/C++ application with a companion
Python analysis module and Sphinx documentation.

## Repository structure

The repository has three independently-buildable trees:

- **C++ application** — `src/` (source), `tests/` (Qt-Test integration
  tests), `cmake/` (build modules), `CMakeLists.txt`. Produces
  `blackchirp` (acquisition app) and `blackchirp-viewer`. See
  `src/AGENTS.md` for tree-specific rules (code style, hardware drivers,
  settings, logging).
- **Sphinx documentation** — `doc/source/` (RST), built via the `docs`
  CMake target. See `doc/AGENTS.md` for tree-specific rules.
- **Python module** — `python/blackchirp/` (PyPI package, pyproject.toml,
  pytest suite) plus example notebooks and fixture data under `python/`.
  See `python/AGENTS.md` for tree-specific rules.

The three trees share data formats (the on-disk semicolon-delimited CSV
schema) but otherwise have independent build, test, and lint pipelines.

`dev-docs/` contains development planning documents that are not part
of the released product. They will be purged before release. They may
be updated to reflect current progress, but they are not load-bearing
documentation — the published Sphinx docs are.

## Critical rules (apply to all trees)

### Timeless commits and comments

Code comments and commit messages must be timeless with respect to
source evolution. Omit "Phase X", "Step Y", "Task N.M",
"now/currently/previously" in the development-history sense, "added in
v1.2", "we recently changed". Markers describing **runtime program
execution** are fine and often necessary: "after `initialize()`
completes", "before the first FID arrives", "currently connected"
(describing live device state), "previously stored configuration"
(describing prior persistent state).

The test: would the sentence read correctly to a developer five years
from now with no knowledge of this commit? If yes, it is runtime/state
language; if no, it is source-evolution language and must be removed.

### Do not install dependencies without consent

The agent **must not** install dependencies, create environments, or
modify `requirements.txt` / `environment.yml` / `pyproject.toml`
dependency lists without explicit user consent for each invocation.
This includes `pip install`, `conda install`, `mamba install`,
`uv add`, `npm install`, and equivalents. If a build or test fails
because of a missing dependency, report the failure, point at the
relevant requirements file, and wait. Offering to install is fine;
acting on the offer requires the user to say so.

### Subdirectory navigation

Prefer the dedicated subdirectory flag (`-C`, `-B`, `--test-dir`,
`--rootdir`, etc.) over chaining `cd`. Never chain `cd … && …`: a
failure mid-chain leaves the shell in the wrong directory and the rest
of the chain runs against the wrong tree.

```bash
# Bad
cd build/tests && ctest

# Good
ctest --test-dir build/tests
cmake --build build/Desktop-Debug/ --target blackchirp -j$(nproc)
```

## Local configuration

Per-checkout, machine-specific conventions (conda environment names,
hardware-specific tool paths, MCP-server availability, vendor library
locations) live in `AGENTS.local.md` at the project root. The file is
gitignored. **Read it before running build or test commands** — it
records the local invocation conventions that committed guidance
deliberately omits. If it does not exist, or a required local
convention is not recorded there, ask the user before guessing.

## Building the project (all trees)

Build directories live under `build/` in the project root.

### C++ application

```bash
# Debug build (default)
cmake . -B build/Desktop-Debug/
cmake --build build/Desktop-Debug/ -j$(nproc)

# Release build (suppresses qDebug output)
cmake . -B build/Desktop-Release/ -DCMAKE_BUILD_TYPE=Release
cmake --build build/Desktop-Release/ -j$(nproc)

# Specific targets
cmake --build build/Desktop-Debug/ --target blackchirp -j$(nproc)
cmake --build build/Desktop-Debug/ --target blackchirp-viewer -j$(nproc)
```

Always use `cmake --build` rather than `make -C`: when CMake
auto-regenerates the build system mid-invocation (e.g., after editing
`CMakeLists.txt`), `make` fails with "No rule to make target
'CMakeFiles/Makefile2'", whereas `cmake --build` handles regeneration
cleanly.

A full rebuild takes ~2:30–3:00; allow at least a 300000 ms timeout
when shell-driven, with `-j$(nproc)`.

User-toggleable build options live in `cmake/BuildConfig.cmake`
(auto-created from a template on first configure; treat as read-only).
The full CMake module map and build-option matrix is in
`doc/source/developer_guide/build_system.rst`.

### C++ tests

```bash
cmake . -B build/tests
cmake --build build/tests --target tests -j$(nproc)
ctest --test-dir build/tests
```

### Documentation

```bash
cmake --build build --target docs     # Sphinx HTML + Doxygen
cmake --build build --target doxygen  # Doxygen XML/HTML only
```

Activation of an environment that satisfies the doc requirements
(`doc/source/requirements.txt`) is required first. Local activation
conventions live in `AGENTS.local.md`.

### Python module

```bash
pytest --rootdir python/blackchirp python/blackchirp/tests
```

## Dependencies

- Qt6 (core, gui, widgets, network, serialport, concurrent, test)
- Qwt (scientific plotting)
- GSL (GNU Scientific Library)
- Eigen3 (header-only linear algebra; used by Analysis/PeakFinder)
- Optional: CUDA toolkit
- Documentation: see `doc/source/requirements.txt`
- Python module: see `python/blackchirp/pyproject.toml`
