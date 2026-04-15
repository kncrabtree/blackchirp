#ifndef LABJACKLIBRARY_H
#define LABJACKLIBRARY_H

#include "vendorlibrary.h"

namespace BC::Key::LabJack {
    inline constexpr QLatin1StringView labjackU3{"labjackU3"};  /*!< Settings key for LabJack U3 library */
}

// Forward declarations for LabJack types to avoid including vendor headers
typedef void* HANDLE;
typedef unsigned char BYTE;
typedef unsigned int UINT;

/*!
 * \brief Runtime wrapper for LabJack USB library (liblabjackusb.so)
 * 
 * This class provides dynamic loading of the low-level LabJack USB library without
 * compile-time dependencies. It wraps the LJUSB_* functions that u3.cpp uses
 * to communicate with LabJack devices at the USB protocol level.
 * 
 * The library follows the singleton pattern for global state management
 * and provides comprehensive platform-specific library discovery.
 */
class LabjackLibrary : public VendorLibrary
{
    Q_OBJECT
    
public:
    /*!
     * \brief Get singleton instance of LabJack USB library wrapper
     * \return Reference to singleton instance
     */
    static LabjackLibrary& instance();
    
    // VendorLibrary interface
    bool isAvailable() const override { return d_libraryLoaded; }
    QString libraryName() const override { return "LabJack USB Library"; }
    QString errorString() const override { return d_errorString; }
    QStringList platformLibraryNames() const override;
    QStringList defaultSearchPaths() const override;
    QString getInstallationInstructions() const override;
    
    // Low-level LabJack USB function pointers (LJUSB_* functions from liblabjackusb.so)
    // These are the actual functions that u3.cpp calls
    
    typedef float (*LJUSB_GetLibraryVersion_t)(void);
    typedef unsigned int (*LJUSB_GetDevCount_t)(unsigned long ProductID);
    typedef HANDLE (*LJUSB_OpenDevice_t)(UINT DevNum, unsigned int dwReserved, unsigned long ProductID);
    typedef void (*LJUSB_CloseDevice_t)(HANDLE hDevice);
    typedef unsigned long (*LJUSB_Write_t)(HANDLE hDevice, const BYTE *pBuff, unsigned long count);
    typedef unsigned long (*LJUSB_Read_t)(HANDLE hDevice, BYTE *pBuff, unsigned long count);
    typedef bool (*LJUSB_IsHandleValid_t)(HANDLE hDevice);
    typedef bool (*LJUSB_ResetConnection_t)(HANDLE hDevice);
    
    // Function pointer instances - these replace direct library calls in u3.cpp
    LJUSB_GetLibraryVersion_t LJUSB_GetLibraryVersion = nullptr;
    LJUSB_GetDevCount_t LJUSB_GetDevCount = nullptr;
    LJUSB_OpenDevice_t LJUSB_OpenDevice = nullptr;
    LJUSB_CloseDevice_t LJUSB_CloseDevice = nullptr;
    LJUSB_Write_t LJUSB_Write = nullptr;
    LJUSB_Read_t LJUSB_Read = nullptr;
    LJUSB_IsHandleValid_t LJUSB_IsHandleValid = nullptr;
    LJUSB_ResetConnection_t LJUSB_ResetConnection = nullptr;
    
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