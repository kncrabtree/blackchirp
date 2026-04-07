"""
Blackchirp Python GPIB Controller Driver Template

This script is loaded by the PythonGpibController C++ trampoline class. It
provides a complete virtual GPIB controller implementation that you can
customize for your hardware.

The GPIB controller in Blackchirp acts as the IEEE-488 bus controller. Other
hardware objects (clocks, pulse generators, etc.) that communicate over GPIB
call GpibController::writeCmd(), writeBinary(), and queryCmd() on the shared
controller instance. The C++ GpibController base class handles:
  - Bus arbitration via mutex locking (one device talks at a time)
  - GPIB address management (reserving/releasing addresses per device)
  - Routing writeCmd/writeBinary/queryCmd through p_comm to the physical bus

Your Python script implements the low-level address management that the base
class uses when switching between GPIB devices on the bus. These are the only
two operations that vary by hardware implementation.

Class name must match the Python Class setting in the Hardware Configuration
dialog (default: "GpibControllerDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel

Method summary:
    initialize()          -- called once on startup
    test_connection()     -- verify connection, return True on success
    read_address()        -- read current GPIB talker/listener address, return True on success
    set_address(address)  -- select a device at the given GPIB address, return True on success
"""


class GpibControllerDriver:
    """Python GPIB Controller hardware driver.

    The GpibController base class calls these methods when it needs to select
    a device on the GPIB bus before sending or receiving data. The base class
    holds a mutex and manages address bookkeeping; your code only needs to
    issue the hardware commands to assert or query the active address.

    Required methods:
        read_address()     -> bool  (True if current address was read successfully)
        set_address(address) -> bool  (True if address was selected successfully)

    Lifecycle methods:
        initialize()      -- called once on startup (via HardwareObject::initialize)
        test_connection() -- verify hardware is reachable (via HardwareObject::testConnection)
        sleep(sleeping)   -- called on hardware standby transitions
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        This is called from GpibController::initialize(). Use this to set up
        internal state and prepare any resources needed for GPIB communication.

        The comm proxy is available but the connection has not been tested yet.
        """
        self.log.log("GPIB Controller driver initialized")

        # Internal state for virtual mode: track the currently addressed device
        self._current_address = -1

    def test_connection(self):
        """Verify communication with the GPIB controller hardware.

        Called from GpibController::testConnection(). If this returns True,
        Blackchirp considers the controller ready and other hardware objects
        may begin reserving GPIB addresses.

        Returns:
            bool: True if the controller is reachable, False otherwise.

        Examples:
            # For a USB-GPIB adapter that accepts SCPI:
            # response = self.comm.query("*IDN?\\n")
            # return len(response.strip()) > 0

            # For a serial GPIB adapter (e.g. Prologix):
            # self.comm.write("++ver\\n")
            # response = self.comm.read()
            # return "Prologix" in response
        """
        self.log.log("Testing GPIB Controller connection")
        return True

    # =========================================================================
    # Address Methods
    # =========================================================================

    def read_address(self):
        """Read the currently addressed GPIB device from hardware.

        Called by GpibController::readAddress() to synchronize the C++ layer's
        address state with the actual hardware state. This is typically called
        once during initialization or after an external address change.

        The result is used to update d_currentAddress in the C++ base class.
        If your hardware does not support querying the current address, return
        True without modifying any state.

        Returns:
            bool: True if the address was read successfully (or not applicable),
                  False on communication error.

        Examples:
            # For a Prologix USB-GPIB adapter:
            # self.comm.write("++addr\\n")
            # response = self.comm.read()
            # try:
            #     self._current_address = int(response.strip())
            #     return True
            # except ValueError:
            #     return False
        """
        # Virtual: nothing to read from hardware
        return True

    def set_address(self, address):
        """Select a GPIB device at the given address.

        Called by GpibController::setAddress() before each writeCmd(),
        writeBinary(), or queryCmd() operation when the target address differs
        from the current address. After a successful set_address(), subsequent
        comm operations go to the device at this address.

        Args:
            address (int): GPIB primary address (0-30) of the device to select.

        Returns:
            bool: True if the address was selected successfully, False on error.

        Examples:
            # For a Prologix USB-GPIB adapter:
            # self.comm.write(f"++addr {address}\\n")
            # self._current_address = address
            # return True

            # For a National Instruments GPIB-USB-HS via NI-VISA:
            # Use self.comm.write with the adapter's address select command.
        """
        # Virtual: track address internally
        self._current_address = address
        return True

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode.

        Args:
            sleeping (bool): True = entering sleep, False = waking up.
        """
        if sleeping:
            self.log.debug("GPIB Controller entering sleep mode")
        else:
            self.log.debug("GPIB Controller waking from sleep mode")
