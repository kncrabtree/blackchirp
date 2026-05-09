# Contributing to Blackchirp

Discussions about installation, usage, bugs, and features happen in the
[Discord server](https://discord.gg/88CkbAKUZY) and on the
[issue tracker](https://github.com/kncrabtree/blackchirp/issues).

## Reporting a Bug

Open an issue and include:

- Operating system and version
- Compiler version (and Qt version, if a build problem)
- Blackchirp version
- Expected and observed behavior
- Relevant excerpts from log files (with error messages)

For build problems, attach the CMake configure output and your local
`cmake/BuildConfig.cmake`.

## Installation Help

Installation help is best obtained on the Discord server. Include:

- Operating system
- For source builds: CMake configure output, your
  `cmake/BuildConfig.cmake`, and any compiler error messages
- For binary installs: which package you used (DEB / RPM / DMG /
  NSIS / AppImage) and the failure mode

## Submitting a Patch or Feature

Open a GitHub issue describing the scope of the change before starting
work. This avoids overlap with in-flight work and gives the maintainers
a chance to flag architectural concerns early.

### Branch model

`master` carries released versions. `devel` is the integration branch:
all feature and bug-fix branches target `devel`, and `devel` is merged
into `master` only when a release is cut.

Workflow:

1. Fork the repository.
2. Branch your work off `devel` (`git checkout -b my-feature devel`).
3. Implement, commit, push.
4. Before opening the PR, rebase or merge the latest `devel` into your
   branch and confirm tests still pass.
5. Open the PR against `devel` (not `master`).

### Changelog gate

Every PR into `devel` must add an entry to the next-release page under
`doc/source/changelog/` (e.g., `doc/source/changelog/2.1.0.rst`).
Create the file if it does not yet exist for the upcoming version, and
add it to the toctree in `doc/source/changelog.rst`. The entry should
describe the change from a *user's* perspective; implementation detail
belongs in the commit message. Use the `:commit:` role to link the SHA
of the merged commit when known. PRs that do not update the changelog
will not be merged.

### Tests

- C++ changes: the `ctest` suite must pass. Add a new test if the
  change is testable in isolation (data classes, parsers, storage,
  hardware-key registration).
- Python changes: the `pytest` suite under `python/blackchirp/tests/`
  must pass.
- Documentation changes: the `docs` CMake target must build with no
  new Sphinx warnings.

### Commit messages

- One concern per commit. If your branch addresses two unrelated things,
  split it into two commits.
- Subject line in the imperative ("Fix off-by-one in FID averaging",
  not "Fixed" or "Fixes").
- Keep messages timeless with respect to source evolution: no
  "Phase X", "Step Y", "now/currently/previously" referring to
  development history. Imagine reading the message five years from now
  with no knowledge of this PR. Markers describing **runtime program
  execution** are fine ("after `initialize()` completes", "before the
  first FID arrives").

### Code style

The single canonical reference for all three trees is
`doc/source/developer_guide/conventions.rst` (rendered at
https://blackchirp.readthedocs.io/page/developer_guide/conventions.html),
which carries the C++, Python, and documentation conventions side by
side. Read the relevant section before contributing.

Per-tree contributor guidance for the agent-facing details (build
commands, dependency policy, public API surface, etc.) is in the
`AGENTS.md` file at the root of each tree (`src/AGENTS.md`,
`doc/AGENTS.md`, `python/AGENTS.md`).
