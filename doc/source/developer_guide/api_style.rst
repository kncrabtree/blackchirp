API Reference Style
===================

This section sets the conventions for class-level API documentation
under :doc:`/classes`. It is the contract between the headers, the
Doxygen XML they produce, and the Sphinx pages that surface them.

Where prose lives
-----------------

Member-level documentation — what a function does, what its parameters
mean, the invariants it preserves, threading notes — lives in Doxygen
comments **in the header file**. The corresponding ``.rst`` page under
``doc/source/classes/`` holds only:

* a 1–3 paragraph orientation intro that situates the class in the
  larger system, names the most relevant collaborators, and links
  outward to the user-guide and developer-guide chapters that cover
  the feature it supports,
* optional named subsections (H2, ``-`` underline) that group prose
  by topic when the orientation runs longer than three paragraphs
  (e.g. *Validation*, *System profiles*, *Registration macros*), and
* a final ``API Reference`` section (also H2, ``-`` underline) that
  contains every ``.. doxygenclass::``, ``.. doxygenstruct::``, and
  ``.. doxygenenum::`` directive on the page.

The header is the single source of truth. If the same sentence appears
in both the header and the ``.rst`` page, delete it from the ``.rst``.
Header comments are read by Doxygen, by ``codebase-memory``, by IDE
tooltips, and by any contributor opening the file — none of those
audiences ever see the ``.rst``. A ``.rst`` page that paraphrases the
header just creates two places that have to be kept in sync.

Doxygen comment style
---------------------

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

Sphinx directives
-----------------

* Prefer ``.. doxygenclass::`` over ``.. doxygenfile::`` — one focused
  page per class, members grouped by member rather than by source
  position.
* Match the directive to the entity kind: ``.. doxygenstruct::`` for
  ``struct`` declarations and ``.. doxygenenum::`` for enums (free or
  scoped). Using ``.. doxygenclass::`` for a struct silently produces
  empty output because Breathe looks up by exact compound kind.
* Place every Doxygen directive under the page's final ``API Reference``
  section. Without this wrapper, sibling directives appear visually
  nested under whatever prose subsection precedes them in the page TOC.
* Default options:

  .. code-block:: rst

     .. doxygenclass:: ClassName
        :members:
        :protected-members:
        :undoc-members:

  Drop ``:protected-members:`` for classes whose protected interface
  is not part of the contract (rare in this codebase — most base
  classes expose a protected hook layer that subclass authors are
  expected to override).

* Use ``:cpp:class:`` and ``:cpp:func:`` when cross-referencing C++
  symbols from prose. Use ``:doc:`` for cross-references between
  documentation pages. Do not embed raw ``.html`` links.

Refresh checklist when editing a class
--------------------------------------

When changing a header that has a corresponding API page:

1. Update the Doxygen comments alongside the code change.
2. Re-skim the ``.rst`` intro: if a collaborator referenced there has
   been renamed or removed, fix the prose.
3. Build the docs with
   ``cmake --build build --target docs`` and check ``doxygen.log``
   for new warnings about undocumented public members of the touched
   class.
