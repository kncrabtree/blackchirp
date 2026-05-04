GPIB Controller
===============

* Overview_
* Settings_
* Drivers_

Overview
--------

A GpibController device serves as a bridge between a piece of hardware that communicates over GPIB (General Purpose Interface Bus) and a computer. Unlike RS232 and TCP, computers do not offer native GPIB support; a custom card or external bridge is required. Because GPIB is a parallel bus, a single controller can simultaneously support many devices. Blackchirp can also drive multiple GpibController instances at once: each GPIB instrument selects which controller it lives behind through a per-device setting in the loadout, and address ownership is tracked at runtime so two devices on the same controller cannot accidentally claim the same GPIB address.

Settings
--------

The GpibController base class itself has no user-facing settings; the controller is configured entirely through its underlying communication channel (TCP host/port or RS232 port and baud rate), which is set up in the standard :ref:`hardware-menu-communication` flow.

Drivers
-------

The supported hardware is the Prologix bridge, which Blackchirp can talk to over either Ethernet or USB. A virtual controller is also registered for testing.

Virtual
.................

A stub driver that responds successfully to writes and echoes queries. Used for testing and for builds without GPIB hardware attached.

Prologix GPIB-LAN Controller
..............................................

The `Prologix GPIB-Ethernet controller <https://prologix.biz/product/gpib-ethernet-controller/>`_ plugs into one or more GPIB instruments and connects to the computer over Ethernet. Blackchirp implements it as a TcpInstrument.

Prologix GPIB-USB Controller
..............................................

The `Prologix GPIB-USB controller <https://prologix.biz/product/gpib-usb-controller/>`_ is functionally identical to the Ethernet variant but connects through an internal USB-RS232 bridge. Blackchirp implements it as an Rs232Instrument.
