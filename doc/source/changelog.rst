.. index::
   single: changelog
   single: release notes

Changelog
=========

This chapter is the per-release record of what changed in Blackchirp.
Each release has its own page under ``changelog/``, named for the
version it covers; the toctree below lists those pages newest first.
Version-keyed prose ("v1.x did X, 2.0 does Y") lives here and in the
:doc:`migration guide </migration>`. The
:doc:`User Guide </user_guide>` and
:doc:`Developer Guide </developer_guide>` describe the program in
its current state and do not carry release markers in their prose.

For releases after 2.0.0, each page addresses two audiences in
parallel. The **user** sections describe changes a Blackchirp
operator notices when they update an existing installation: new
menus and dialogs, renamed widgets, new acquisition modes,
file-format additions, and visible behavior changes. The
**developer** sections describe backend changes that do not surface
in the user interface but matter to contributors and integrators:
refactors, internal API rearrangements, threading or storage
rework, and changes to the build or test infrastructure. Each page
makes the two audiences explicit through its section headings.

The 2.0.0 page is the one exception to the dual-audience rule. The
change set between v1.x and 2.0.0 is large enough that a
developer-oriented section on the same page would drown out the
user-visible content; the 2.0.0 page therefore covers the user
perspective only. A v1.x user upgrading should also read the
:doc:`migration guide </migration/v1_to_v2>`, which walks through
the upgrade actions for each user-visible change.

Keeping the changelog updated
-----------------------------

When a release is cut, add a new page under ``changelog/`` named
``<version>.rst`` and prepend it to the toctree below so the newest
release sorts first. Group entries by topic and write them in the
past tense ("Replaced…", "Added…", "Renamed…"), aiming for one or
two lines per bullet. Cross-link each user-facing entry to the
User Guide page that documents the feature; cross-link each
developer-facing entry to the relevant developer-guide section or
API page. Pure bug fixes that do not change documented behavior
are out of scope; if a fix restores a previously-broken documented
feature, a one-line entry under the relevant section is
appropriate.

.. toctree::
   :maxdepth: 1

   changelog/2.0.0
