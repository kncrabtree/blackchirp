#ifndef LABJACKCONSTANTS_H
#define LABJACKCONSTANTS_H

/*!
 * \brief LabJack USB library constants
 * 
 * Constants used with the low-level LabJack USB library (liblabjackusb.so).
 * These constants are used by u3.cpp for device identification and communication.
 */

// LabJack product IDs (from LabJack SDK)
#define U3_PRODUCT_ID 0x0003
#define U6_PRODUCT_ID 0x0006
#define UE9_PRODUCT_ID 0x0009
#define U12_PRODUCT_ID 0x000C

// LabJack vendor ID
#define LJ_VENDOR_ID 0x0CD5

// USB communication constants
#define U3_PIPE_EP1_OUT 0x01
#define U3_PIPE_EP2_IN  0x82
#define U3_PIPE_EP3_IN  0x83

// Error codes that may be returned by LJUSB functions
#define LJUSB_ERROR_DEVICE_NOT_FOUND    -1
#define LJUSB_ERROR_WRITE_FAILED        -2
#define LJUSB_ERROR_READ_FAILED         -3
#define LJUSB_ERROR_INVALID_HANDLE      -4

#endif // LABJACKCONSTANTS_H