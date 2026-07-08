LIF Conversion Stage
=====================

* Overview_
* Settings_
* Drivers_

Overview
--------

A LIF Conversion Stage represents one physical optical element — a doubling crystal, a mixing crystal — in the LIF frequency-conversion chain between the tunable laser and the sample. Each configured stage appears as a row in the Conversion tab's table (see :doc:`/user_guide/lif/conversion`), where its inputs are wired and, for at most one stage, its output is marked as the excitation beam. The device itself only moves to and reports a local input-beam position (in cm⁻¹); the chain's topology — which stage feeds which, and which stage is the excitation beam — is per-experiment state configured on the Conversion tab, not part of the device profile.

Settings
--------

Most LIF Conversion Stage settings are exposed in the :doc:`hardware dialog </user_guide/hwdialog>` with inline labels and tooltips, so they need no additional explanation here. A few items are worth highlighting:

* ``Conversion Operation`` sets the stage's operation — NHG (N-th harmonic), SFG, or DFG. Concrete drivers usually pin this to a fixed value that matches the physical device (a doubler is always NHG, for example), so it is not normally something you choose yourself.
* ``Harmonic Order`` sets the harmonic order N for an NHG stage; it is ignored for SFG/DFG stages. Some drivers fix this at profile creation rather than leaving it freely editable (see Sirah FCU below).
* ``Verify Move`` controls whether a move is confirmed against the stage's read-back position after every move. When enabled (the default), a mismatch beyond ``Verify Tolerance`` fails the move; when disabled, a mismatch is only logged as a warning and the move is treated as successful.
* ``Verify Tolerance (cm-1)`` sets the read-back window, in cm⁻¹, used by ``Verify Move``. Defaults to 1.0.

Drivers
-------

Virtual
.................

A dummy driver.

Fixed
.................

Represents a conversion-topology node that is not under Blackchirp's control — a crystal or compensator tuned by hand, or one you simply want represented in the chain's math without automating its motion. Moves always report success at the commanded position, so the stage participates fully in the chain with no separate place to enter its state.

Sirah FCU
.................

The Sirah Frequency Conversion Unit is a doubling-stage driver for a Sirah frequency-conversion unit — a separate Sirah instrument from the :doc:`Sirah Cobra <liflaser>` grating controller, connected on its own serial port with its own communication settings. Harmonic order defaults to 2, the common lone-doubler case, and is fixed at profile creation rather than editable afterward.

.. note::
   The driver assumes the FCU shares the Sirah Cobra's sine-bar tuning
   mechanism and binary communication protocol, and ships with the
   Cobra's grating-stage geometry as placeholder defaults for the
   doubling crystal. Both the shared-mechanism assumption and the
   geometry defaults are unverified against real hardware; calibrate
   the stage geometry against the unit's own datasheet before relying
   on this driver for production tuning.
