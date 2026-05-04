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

Each release page is organized the same way: a short summary of the
release, then a **Highlights** section surfacing the largest changes,
then component-level sections grouped by subsystem (build and
distribution, hardware configuration, hardware drivers, acquisition,
user interface, overlays, LIF, logging, file formats, tooling, and so
on — only the ones the release actually touched), then a **Bug fixes**
section with user-noticeable fixes grouped by the same subsystem
labels. Entries are written in the past tense ("Replaced…", "Added…",
"Fixed…"), aim for one or two lines, and cross-link to the User Guide
page that documents the affected feature when one exists.

When a release is cut, add a new page under ``changelog/`` named
``<version>.rst`` and prepend it to the toctree below so the newest
release sorts first.

.. toctree::
   :maxdepth: 1

   changelog/2.0.0
   changelog/1.1.0
