.. index::
   single: migration guide
   single: upgrading

Migration Guide
===============

This chapter is for users who are upgrading an existing Blackchirp
installation across a major version boundary. Each section covers
the breaking changes a user is likely to encounter, where their
on-disk data and settings live across the transition, and the
one-time migration steps required to bring an existing setup
forward.

The chapter is organized as one page per upgrade path, named for
the source and destination versions. Each page is a checklist:
every entry names the starting condition on the source version,
the end state on the destination version, and the steps to get
from one to the other. The reader is expected to walk a page top
to bottom in one sitting. Cross-links into the
:doc:`User Guide </user_guide>` cover the destination-version
features in depth, and the matching
:doc:`changelog </changelog>` page is the authoritative summary
of everything that changed.

If you are setting up Blackchirp for the first time and have no
prior version to migrate from, this chapter is not for you — start
with the :doc:`User Guide </user_guide>` instead.

.. toctree::
   :maxdepth: 1

   migration/v1_to_v2
