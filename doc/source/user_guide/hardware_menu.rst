.. index::
   single: Rf Configuration
   single: Communication
   single: Hardware
   single: GPIB
   single: RS232
   single: TCP
   single: Clock
   single: UpLO
   single: DownLO
   single: AwgRef
   single: DRClock
   single: DigRef
   single: ComRef
   single: Frequency

Hardware Menu
=============

.. image:: /_static/user_guide/hardware_menu/menu.png
   :width: 150
   :align: center
   :alt: Hardware Menu Screenshot

The hardware menu allows you to control the instrument hardware and to configure settings that are used throughout the program.

Communication
-------------

.. image:: /_static/user_guide/first_run/hwcommunication.png
   :width: 700
   :alt: Hardware communication dialog

The **Communication** option opens the hardware communication dialog that was initially displayed during the `first run <first_run.html>`_.
Here you can make changes to any communication settings as needed.
If an important piece of hardware throws an error, you may need to open this dialog and test its connection to re-establish communication.
Some tips about connecting to instruments:

- **GPIB** instruments are supported through use of a GPIB-LAN or GPIB-RS232 bridge (though only an implementation of a Prologix GPIB-LAN controller is available at present). Before attempting to communicate with a GPIB device, ensure that the connection to the controller is successful.
- For **TCP** instruments, it is strongly recommended that you use a dedicated LAN interface and configure all devices to communicate on the link-local (169.254.X.X) address space. Many instruments support 1 Gbps communication speeds, so ensure that your network adapter supports 1 Gbps and that any switches in the network have sufficient bandwidth to support gigabit communication speeds for all devices.
- **RS232** ports are typically unavailable on modern computers, but USB-RS232 adapters are commonly used to support communication. I recommend using an adapter based on an FTDI chipset, as these are fairly reliable and their firmware comes with a unique ID per device. This is important because...
- On Linux/Unix systems, by default, USB-RS232 adapters are assigned to a device handle ``/dev/ttyUSBX``, where X is a number. However, the number depends on the order in which the devices are added to the system, and upon reboot, there is no predetermined order in which they are assigned. The device IDs, then, may change over time. To avoid this, you can configure ``udev`` to recognize the serial number of an FTDI chip and assign a symbolic link that maps to the appropriate device every time it is connected. To enable this, do the following:

   1. Edit the example udev rules file (``52-serial.rules``) that is provided with the serial numbers of your devices. To find the serial number for device ``/dev/ttyUSB1``, use ``udevadm info --name=/dev/ttyUSB1``, and optionally ``| grep serialNo``.
   2. Place the file in your OS's udev rules folder. On openSUSE, the location is ``/etc/udev/rules.d/``, though it varies with Linux distribution.
   3. As root, reload the udev rules (``udevadm control --reload``). The exact command may vary depending on distrubution; see the manpage for ``udevadm`` if the command fails.
   4. As root, trigger udev to cause currently connected devices to be re-loaded with the new rules (``udevadm trigger``).

- After completing these steps, new symlinks should be available, and you can use those symlinks for the RS232 Device ID.
- For many devices, it may require two attempts to communicate, especially for serial devices that are first tested with the wrong RS232 settings. The device's internal buffer may be filled with incorrect characters that prevent it from correctly processing Blackchirp's device query. If the device responds with a readable but incorrect response, it is worth just re-testing the connection before making additional changes.

Finally, the **Test All Connections** option attempts to reconnect to all attached hardware using the current settings.


Rf Configuration
----------------

.. image:: /_static/user_guide/hardware_menu/rfconfig.png
   :width: 750
   :align: center
   :alt: Rf Configuration dialog

This dialog provides options for telling Blackchirp how your CP-FTMW spectrometer is configured.
By properly making settings here, Blackchirp can convert between AWG/chirp frequency as well as between digitizer/molecular frequency.
In addition, if any programmable clocks are present, Blackchirp can control them (critical for `LO Scan and DR Scan acquistitions <experiment/acquisition_types.html>`_).
At the top of the dialog, set the appropriate values for your upconversion and downconversion chains according to the diagram below.

.. image:: /_static/user_guide/hardware_menu/clocks.svg
   :width: 800
   :align: center
   :alt: Clock layout

The AWG is assumed to pass through a multiplier and is then mixed with the upconversion local oscillator.
Then, the output of the mixer (either the upper or lower sideband) is assumed to pass through another multiplier to create the final chirp.
Similarly, the molecular FID signal is assumed to be mixed with the downconversion local oscillator, and then frequency assignments are based on either the upper or lower sideband.
For spectrometers that directly synthesize chirps and diretly digitize FIDs (i.e., no mixers/multipliers/oscillators), enter ``1x`` for all multiplication values, ``Upper`` sideband for both mixers, and assign a FixedClock output to both the ``UpLO`` and ``DownLO`` with a frequency of 0.0 MHz.

The **Clock Configuration** box allows you to assign each logical clock to a specific output of a physical (or virtual) clock.
Additionally, if the output is passed through a frequency multiplier or divider, you can enter the appropriate multiplication/division factor.
In the ``Frequency`` column, enter the desired logical frequency, and Blackchirp will compute the required clock frequency to produce that value.
Using the values entered in the screenshot above as an example:

- The ``UpLO`` frequency is set to 11520 MHz and the multiplication factor is set to ``2x``. Blackchirp will therefore assign a frequency of 5760 MHz (11520/2) to the physical clock assigned to the ``UpLO`` role.
- The downconversion clock has a multiplication factor of 8, the physical clock assigned to ``DownLO`` will be set to 40960/8 = 5120 MHz.
- The total upconversion chain takes an AWG waveform spanning 4895-1520 MHz and generates a chirp of 26500-40000 MHz but using the lower sideband of the mixer: :math:`f_{\text{Chirp}} = 4\left(11520 - f_{\text{AWG}} \right)`.
- The downconversion chain maps offset frequencies of 960-14460 MHz to 40000-26500 MHz (assuming the digitizer bandwidth is at least 14460 MHz): :math:`f_{\text{Molecular}} = 40960 - f_{\text{FID}}`.

In homodyne systems, the same physical clock is used for both upconversion and downconversion.
To enforce this, check the ``Use common LO for up/downconversion`` box.
When checked, the ``DownLO`` row is disabled, and the settings will mirror the ``UpLO`` at all times.

Each logical clock in Blackchirp has a specific meaning within the program:

- ``UpLO`` is the upconversion local oscillator, and as described above, it is used to convert between AWG frequency and chirp frequency. In an `LO Scan <experiment/acquisition_types.html>`_, the UpLO frequency may be tuned from one step to the next.
- ``DownLO`` likewise is the downconversion local oscillator used to convert frequencies in the FID to molecular frequencies for display. It also may be varied in an `LO Scan <experiment/acquisition_types.html>`_, and its frequency can be set to mirror the ``UpLO`` by checking the ``Use common LO for up/downconversion`` box.
- ``AwgRef`` is used to set a reference frequency for a waveform generator. For example, the AD9914 DDS requires an external clock frequency to determine its sample rate, and Blackchirp reads the value from this clock setting. If the reference oscillator is programmable, then assigning a physical clock channel to this role will allow Blackchirp to control the frequency. At present, this clock is not used for most AWGs (Tektronix and Agilent models); they are set to the maximum allowed sample rate. **This behavior may change in the future, requiring that the sample rate be specified here.**
- ``DRClock`` is used to control a frequency source used in double resonance experiments. During a `DR Scan <experiment_setup.html#dr-scan>`_, this clock is tuned during each step.
- ``DigRef`` is similar to ``AwgRef``, but for the FTMW digitizer rather than the AWG. Currently, Blackchirp does not use this setting internally, so its only use is to assign a frequency to a physical clock that is a reference for the digitizer (e.g., if a programmable synthesizer sets the reference for a Spectrum Instrumentation M4i2211x8 card).
- ``ComRef`` is a common reference oscillator, such as the 10 MHz signal from a rubidium clock. Currently, Blackchirp does not use this information, but if an implementation of an Rb clock is added in the future, Blackchirp will be able to detect if the clock comes unlocked.



Hardware Control/Settings
-------------------------

Below the Rf Configuration option you will find options for each of the pieces of hardware enabled in your ``config.pri`` file.
Selecting the entry brings up a dialog where you can configure various program settings pertaining to that device, and for some devices, you can control the device.
An example dialog is shown below for the PulseGenerator.

.. image:: /_static/user_guide/hardware_menu/pgen.png
   :width: 750
   :align: center
   :alt: Pulse generator control dialog


The top half of the dialog contains controls for the Pulse Generator.
Making changes to the settings there (channel delays, repetition rate, etc.) immediately sends the appropriate commands to the hardware.
On the bottom half, you can adjust hardware settings.
These may control behavior of input widgets (e.g., ``minWidth`` and ``maxWidth`` set the limits on the width boxes), or may control how the device interacts with other parts of the program.
Changes made here are only applied if the **Ok** button is pressed.
If the dialog is closed automatically by the program, or with the "X" on the window or the **Close** button, the settings are not applied.

.. warning::
   It is strongly recommended that you do not make changes to the settings unless you understand exactly what the setting does!

   Incorrect or inappropriate settings may cause unexpected program behavior.

Explanations of the settings are provided for each piece of hardware on the `Hardware Details <hardware_details.html>`_ page.
