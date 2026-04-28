#ifndef LABJACKLIBRARY_H
#define LABJACKLIBRARY_H

#include "vendorlibrary.h"

namespace BC::Key::LabJack {
    inline constexpr QLatin1StringView labjackU3{"labjackU3"};  /*!< Settings key for LabJack U3 library */
}

class LabjackLibrary : public VendorLibrary
{
    Q_OBJECT

public:
    static LabjackLibrary& instance();

    // VendorLibrary interface
    bool isAvailable() const override { return d_libraryLoaded; }
    QString errorString() const override { return d_errorString; }
    QString getVersionInfo() const override;
    QString getInstallationInstructions() const override;

#ifndef Q_OS_WIN
    // Linux/macOS: LJUSB transport symbols (liblabjackusb.so / .dylib)
    QString libraryName() const override { return "LabJack U3 Driver"; }
    QStringList platformLibraryNames() const override;
    QStringList defaultSearchPaths() const override;

    typedef float         (*LJUSB_GetLibraryVersion_t)(void);
    typedef unsigned int  (*LJUSB_GetDevCount_t)(unsigned long ProductID);
    typedef void*         (*LJUSB_OpenDevice_t)(unsigned int DevNum, unsigned int dwReserved, unsigned long ProductID);
    typedef void          (*LJUSB_CloseDevice_t)(void *hDevice);
    typedef unsigned long (*LJUSB_Write_t)(void *hDevice, const unsigned char *pBuff, unsigned long count);
    typedef unsigned long (*LJUSB_Read_t)(void *hDevice, unsigned char *pBuff, unsigned long count);
    typedef bool          (*LJUSB_IsHandleValid_t)(void *hDevice);
    typedef bool          (*LJUSB_ResetConnection_t)(void *hDevice);

    LJUSB_GetLibraryVersion_t LJUSB_GetLibraryVersion = nullptr;
    LJUSB_GetDevCount_t       LJUSB_GetDevCount       = nullptr;
    LJUSB_OpenDevice_t        LJUSB_OpenDevice        = nullptr;
    LJUSB_CloseDevice_t       LJUSB_CloseDevice       = nullptr;
    LJUSB_Write_t             LJUSB_Write             = nullptr;
    LJUSB_Read_t              LJUSB_Read              = nullptr;
    LJUSB_IsHandleValid_t     LJUSB_IsHandleValid     = nullptr;
    LJUSB_ResetConnection_t   LJUSB_ResetConnection   = nullptr;

#else // Q_OS_WIN
    // Windows: UD library symbols (LabJackUD.dll, 64-bit only)
    QString libraryName() const override { return "LabJack U3 Driver"; }
    QStringList platformLibraryNames() const override;
    QStringList defaultSearchPaths() const override;

    // LJ_HANDLE is long; LJ_ERROR is long.
    // __stdcall is a no-op on x86-64 Windows — bare names resolve via QLibrary::resolve.
    typedef long   (*OpenLabJack_t)(long DeviceType, long ConnectionType, const char *pAddress, long FirstFound, long *pHandle);
    typedef long   (*Close_t)(long Handle);
    typedef long   (*eAIN_t)(long Handle, long ChannelP, long ChannelN, double *Voltage, long Range, long Resolution, long Settling, long Binary, long ReservedBit, long Reserved1, double Reserved2, double Reserved3);
    typedef long   (*eDAC_t)(long Handle, long Channel, double Voltage, long Binary, long Reserved1, double Reserved2);
    typedef long   (*eDI_t)(long Handle, long Channel, long *State);
    typedef long   (*eDO_t)(long Handle, long Channel, long State);
    typedef long   (*eTCConfig_t)(long Handle, long *aEnableTimers, long *aEnableCounters, long TCPinOffset, long TimerClockBaseIndex, long TimerClockDivisor, long *aTimerModes, double *aTimerValues, long Reserved1, double Reserved2);
    typedef long   (*eTCValues_t)(long Handle, long *aReadTimers, long *aUpdateResetTimers, long *aReadCounters, long *aResetCounters, double *aTimerValues, double *aCounterValues, long Reserved1, double Reserved2);
    typedef long   (*ErrorToString_t)(long errorcode, char *pString);
    typedef double (*GetDriverVersion_t)(void);

    OpenLabJack_t    OpenLabJack    = nullptr;
    Close_t          Close          = nullptr;
    eAIN_t           eAIN           = nullptr;
    eDAC_t           eDAC           = nullptr;
    eDI_t            eDI            = nullptr;
    eDO_t            eDO            = nullptr;
    eTCConfig_t      eTCConfig      = nullptr;
    eTCValues_t      eTCValues      = nullptr;
    ErrorToString_t  ErrorToString  = nullptr;
    GetDriverVersion_t GetDriverVersion = nullptr;

#endif // Q_OS_WIN

protected:
    void loadFunctions() override;

private:
    explicit LabjackLibrary(QObject *parent = nullptr);
    ~LabjackLibrary() override = default;

    static LabjackLibrary* s_instance;

    LabjackLibrary(const LabjackLibrary&) = delete;
    LabjackLibrary& operator=(const LabjackLibrary&) = delete;
};

#endif // LABJACKLIBRARY_H
