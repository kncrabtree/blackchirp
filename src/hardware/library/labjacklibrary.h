#ifndef LABJACKLIBRARY_H
#define LABJACKLIBRARY_H

#include "vendorlibrary.h"

namespace BC::Key::LabJack {
    static const QString labjackU3{"labjackU3"};  /*!< Settings key for LabJack U3 library */
}

// Forward declarations for LabJack types to avoid including vendor headers
typedef void* HANDLE;
struct U3_CALIBRATION_INFORMATION;
typedef struct U3_CALIBRATION_INFORMATION u3CalibrationInfo;

/*!
 * \brief Runtime wrapper for LabJack U3 SDK
 * 
 * This class provides dynamic loading of the LabJack U3 library without
 * compile-time dependencies. It implements the VendorLibrary interface
 * and provides function pointers for all LabJack U3 functions used by
 * BlackChirp hardware implementations.
 * 
 * The library follows the singleton pattern for global state management
 * and provides comprehensive platform-specific library discovery.
 */
class LabjackLibrary : public VendorLibrary
{
    Q_OBJECT
    
public:
    /*!
     * \brief Get singleton instance of LabJack library wrapper
     * \return Reference to singleton instance
     */
    static LabjackLibrary& instance();
    
    // VendorLibrary interface
    bool isAvailable() const override { return d_libraryLoaded; }
    QString libraryName() const override { return "LabJack U3 Library"; }
    QString errorString() const override { return d_errorString; }
    QStringList platformLibraryNames() const override;
    QStringList defaultSearchPaths() const override;
    
    // LabJack U3 function pointers - Essential functions
    typedef HANDLE (*openUSBConnection_t)(int localID);
    typedef void (*closeUSBConnection_t)(HANDLE hDevice);
    typedef long (*getCalibrationInfo_t)(HANDLE hDevice, u3CalibrationInfo *caliInfo);
    
    // LabJack U3 function pointers - Configuration functions  
    typedef long (*eTCConfig_t)(HANDLE Handle, long *aEnableTimers, long *aEnableCounters,
                                long TCPinOffset, long TimerClockBaseIndex, long TimerClockDivisor,
                                long *aTimerModes, double *aTimerValues, long Reserved1, long Reserved2);
    
    // LabJack U3 function pointers - I/O functions
    typedef long (*eAIN_t)(HANDLE Handle, u3CalibrationInfo *CalibrationInfo, long ConfigIO,
                           long *DAC1Enable, long ChannelP, long ChannelN, double *Voltage,
                           long Range, long Resolution, long Settling, long Binary,
                           long Reserved1, long Reserved2);
    typedef long (*eDI_t)(HANDLE Handle, long ConfigIO, long Channel, long *State);
    
    // Function pointer instances
    openUSBConnection_t openUSBConnection = nullptr;
    closeUSBConnection_t closeUSBConnection = nullptr;
    getCalibrationInfo_t getCalibrationInfo = nullptr;
    eTCConfig_t eTCConfig = nullptr;
    eAIN_t eAIN = nullptr;
    eDI_t eDI = nullptr;
    
protected:
    void loadFunctions() override;
    
private:
    explicit LabjackLibrary(QObject *parent = nullptr);
    ~LabjackLibrary() override = default;
    
    // Singleton instance
    static LabjackLibrary* s_instance;
    
    // Disable copy/assignment
    LabjackLibrary(const LabjackLibrary&) = delete;
    LabjackLibrary& operator=(const LabjackLibrary&) = delete;
};

#endif // LABJACKLIBRARY_H