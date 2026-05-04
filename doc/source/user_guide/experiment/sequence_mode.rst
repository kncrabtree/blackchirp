.. index::
   single: Experiment
   single: Quick Experiment
   single: Sequence

Sequence Mode
=============

.. image:: /_static/user_guide/experiment/sequence.png
   :align: center
   :alt: Experiment sequence mode

Sequence mode runs a series of identical experiments automatically, separated by a configurable time interval.
The interval is measured from the end of one experiment to the start of the next.
The template experiment can be configured using the ``Configure Experiment`` button (which opens the standard wizard) or seeded from a past experiment using the ``Quick Experiment`` button (see :doc:`quick_experiment`).

When a sequence is running, the main user interface displays a countdown message indicating when the next acquisition will begin.
The sequence can be canceled at any time by pressing the ``Abort`` button; pressing ``Abort`` during an active acquisition in the sequence stops that experiment immediately and does not start the next one.
