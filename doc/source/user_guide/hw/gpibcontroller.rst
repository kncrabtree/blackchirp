GPIB Controller
===============

* Overview_
* Settings_
* Implementations_

Overview
--------

A GpibController device serevs as a bridge between a physical piece of hardware that communicates over GPIB (General Purpose Interface Bus) and a computer. Unlike RS232 and TCP, computers do not offer native support for GPIB and instead it requires a custom card or device. Because GPIB is a parallel bus, one GpibController can simultaneously support many devices at once. For this reason, Blackchirp presently supports only one GpibController device, and communication with all GPIB instruments is routed through this device.

Settings
--------

None


Implementations
---------------

Virtual (virtual)
.................

A dummy implementation which does nothing.

Prologix GPIB-LAN Controller (prologixgpiblan)
..............................................

The `Prologix GPIB-Ethernet controller <https://prologix.biz/product/gpib-ethernet-controller/>`_ is a dongle which plugs in to one or more GPIB instruments, and which connects to the computer through an Ethernet cable. Blackchirp implements this device as a TcpInstrument.

Prologix GPIB-USB Controller (prologixgpibusb)
..............................................

The `Prologix GPIB-Ethernet controller <https://prologix.biz/product/gpib-usb-controller/>`_ is the same as the GPIB-Ethernet controller, but connects via an internal USB-RS232 bridge. Blackchirp implements this device as an Rs232Instrument.
