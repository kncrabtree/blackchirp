# Python tree — Agent Guide

The Python tree contains the `blackchirp` PyPI module, its pytest
suite, the example notebooks referenced by the documentation, and the
fixture data used by both. This file applies to all work under
`python/`. Cross-cutting rules (timeless commits, no-install-without-
consent, etc.) are in the repository-root `AGENTS.md`.

## Layout

```
python/
├── blackchirp/             # PyPI package
│   ├── pyproject.toml
│   ├── src/blackchirp/     # Module source
│   ├── tests/              # pytest suite
│   └── README.md
├── single-fid.ipynb        # Example notebook (FID processing)
├── single-lif.ipynb        # Example notebook (LIF processing)
├── example-data/           # Fixtures (used by notebooks and pytest)
├── environment.yml         # Conda environment
└── requirements.txt        # pip equivalents
```

## Test

```bash
pytest --rootdir python/blackchirp python/blackchirp/tests
```

Test fixtures live under `python/example-data/` and are loaded via a
relative path from `python/blackchirp/tests/conftest.py`. Do not
reorganize that directory without updating the fixture paths.

## Dependency policy

The `blackchirp` module **depends only on numpy, scipy, and pandas**
(plus the standard library). No matplotlib, no Qt, no requests, no
network calls, no other plotting libraries. The minimal-dependency
property is a deliberate design constraint: the module must be safe to
install in any scientific Python environment without pulling in heavy
optional infrastructure. Treat the dependency list in `pyproject.toml`
as load-bearing and do not extend it without explicit user consent.

`pytest` is the only dev dependency, declared under
`[project.optional-dependencies]`. The example notebooks may import
matplotlib at the cell level (notebooks have their own dependency
story); do not push matplotlib into the module on the back of a
notebook example.

## Style

- **Formatter:** `black` with default settings.
- **Linter:** `pylint -E` must pass (errors only — warnings are not
  load-bearing).
- **Docstrings:** required on all public classes and functions. Google
  style (per `sphinxcontrib-napoleon`) — `Args:`, `Returns:`,
  `Raises:`, `Example:` sections.

## Public API surface

The names exported by `blackchirp/__init__.py` are the public surface:

- `BCExperiment` — loads an experiment folder.
- `BCFTMW` — multi-FID FTMW data and processing.
- `BCFid` — single FID and its Fourier transform.
- `BCLIF` — LIF scan data and processing.
- `BCLifTrace` — single LIF scan-point trace.
- `coaverage_fids` — time-domain co-averaging across experiments.
- `coaverage_spectra` — magnitude-spectrum co-averaging.

Adding to this surface is a deliberate decision. Document the new
symbol in the module docstring (`blackchirp/__init__.py`) and add a
class page under `doc/source/python/`.

## Versioning and release

The version is declared in `python/blackchirp/pyproject.toml`. Bump it
in the same PR that ships the change, not in a separate "release"
commit. Do not run `python -m build`, `twine upload`, or any
publication command without explicit user consent.

## Notebook execution

The two notebooks under `python/` must be re-executed end-to-end before
commit when their substantive cells have changed. The documentation
embeds them via `nbsphinx-link` and renders cell outputs directly; an
unexecuted or partially-executed notebook will render incorrectly on
ReadTheDocs.
