#include "labjacklibrary.h"
#include <data/storage/settingsstorage.h>
#include <data/loghandler.h>

LabjackLibrary* LabjackLibrary::s_instance = nullptr;

LabjackLibrary::LabjackLibrary(QObject *parent)
    : VendorLibrary(BC::Key::LabJack::labjackU3, parent)
{
    loadLibrary();
}

LabjackLibrary& LabjackLibrary::instance()
{
    if (!s_instance)
        s_instance = new LabjackLibrary();
    return *s_instance;
}

#ifndef Q_OS_WIN

QStringList LabjackLibrary::platformLibraryNames() const
{
    return {
        "liblabjackusb.so"_L1,
        "liblabjackusb.so.2"_L1,
        "liblabjackusb.so.2.7.0"_L1,
        "liblabjackusb.so.2.6.0"_L1,
        "liblabjackusb.dylib"_L1,
    };
}

QStringList LabjackLibrary::defaultSearchPaths() const
{
    return {
        "/usr/local/lib"_L1,
        "/usr/lib"_L1,
        "/usr/lib64"_L1,
        "/usr/lib/x86_64-linux-gnu"_L1,
        "/lib"_L1,
        "/lib64"_L1,
    };
}

void LabjackLibrary::loadFunctions()
{
    if (!d_library.isLoaded()) {
        d_errorString = "Library not loaded"_L1;
        return;
    }

    LJUSB_GetLibraryVersion = reinterpret_cast<LJUSB_GetLibraryVersion_t>(d_library.resolve("LJUSB_GetLibraryVersion"));
    LJUSB_GetDevCount       = reinterpret_cast<LJUSB_GetDevCount_t>      (d_library.resolve("LJUSB_GetDevCount"));
    LJUSB_OpenDevice        = reinterpret_cast<LJUSB_OpenDevice_t>       (d_library.resolve("LJUSB_OpenDevice"));
    LJUSB_CloseDevice       = reinterpret_cast<LJUSB_CloseDevice_t>      (d_library.resolve("LJUSB_CloseDevice"));
    LJUSB_Write             = reinterpret_cast<LJUSB_Write_t>            (d_library.resolve("LJUSB_Write"));
    LJUSB_Read              = reinterpret_cast<LJUSB_Read_t>             (d_library.resolve("LJUSB_Read"));
    LJUSB_IsHandleValid     = reinterpret_cast<LJUSB_IsHandleValid_t>    (d_library.resolve("LJUSB_IsHandleValid"));
    LJUSB_ResetConnection   = reinterpret_cast<LJUSB_ResetConnection_t>  (d_library.resolve("LJUSB_ResetConnection"));

    if (!LJUSB_GetDevCount || !LJUSB_OpenDevice || !LJUSB_CloseDevice ||
        !LJUSB_Write || !LJUSB_Read) {
        d_errorString = "Failed to load essential LabJack USB functions"_L1;
        d_libraryLoaded = false;
        return;
    }

    d_libraryLoaded = true;
    d_errorString.clear();
}

QString LabjackLibrary::getVersionInfo() const
{
    if (LJUSB_GetLibraryVersion)
        return QString::number(LJUSB_GetLibraryVersion(), 'f', 2);
    return {};
}

QString LabjackLibrary::getInstallationInstructions() const
{
    return QString(
        "<p><b>Installing the LabJack exodriver on Linux:</b></p>"
        "<ol>"
        "<li>Install libusb: <code>sudo apt install libusb-1.0-0-dev</code></li>"
        "<li>Clone and build: <a href=\"https://github.com/labjack/exodriver\">github.com/labjack/exodriver</a></li>"
        "<li>Run <code>sudo ./install.sh</code> from the cloned directory</li>"
        "<li>If the device is not detected, check udev rules in <code>/etc/udev/rules.d/</code></li>"
        "</ol>"
        "<p><b>macOS:</b></p>"
        "<pre>brew install libusb\ngit clone https://github.com/labjack/exodriver\ncd exodriver && make install</pre>"
    );
}

#else // Q_OS_WIN

QStringList LabjackLibrary::platformLibraryNames() const
{
    return { "LabJackUD.dll"_L1 };
}

QStringList LabjackLibrary::defaultSearchPaths() const
{
    return {
        "C:/Windows/System32"_L1,
        "C:/Program Files/LabJack/Drivers"_L1,
        "C:/Program Files (x86)/LabJack/Drivers"_L1,
    };
}

void LabjackLibrary::loadFunctions()
{
    if (!d_library.isLoaded()) {
        d_errorString = "Library not loaded"_L1;
        return;
    }

    OpenLabJack    = reinterpret_cast<OpenLabJack_t>   (d_library.resolve("OpenLabJack"));
    Close          = reinterpret_cast<Close_t>         (d_library.resolve("Close"));
    eAIN           = reinterpret_cast<eAIN_t>          (d_library.resolve("eAIN"));
    eDAC           = reinterpret_cast<eDAC_t>          (d_library.resolve("eDAC"));
    eDI            = reinterpret_cast<eDI_t>           (d_library.resolve("eDI"));
    eDO            = reinterpret_cast<eDO_t>           (d_library.resolve("eDO"));
    eTCConfig      = reinterpret_cast<eTCConfig_t>     (d_library.resolve("eTCConfig"));
    eTCValues      = reinterpret_cast<eTCValues_t>     (d_library.resolve("eTCValues"));
    ErrorToString  = reinterpret_cast<ErrorToString_t> (d_library.resolve("ErrorToString"));
    GetDriverVersion = reinterpret_cast<GetDriverVersion_t>(d_library.resolve("GetDriverVersion"));

    if (!OpenLabJack || !eAIN || !eDI || !eDO || !eDAC ||
        !eTCConfig || !eTCValues || !ErrorToString) {
        d_errorString = "Failed to load essential LabJack UD functions"_L1;
        d_libraryLoaded = false;
        return;
    }

    d_libraryLoaded = true;
    d_errorString.clear();
}

QString LabjackLibrary::getVersionInfo() const
{
    if (GetDriverVersion)
        return QString::number(GetDriverVersion(), 'f', 2);
    return {};
}

QString LabjackLibrary::getInstallationInstructions() const
{
    return QString(
        "<p><b>Installing the LabJack UD driver on Windows:</b></p>"
        "<ol>"
        "<li>Download the UD installer from "
        "<a href=\"https://labjack.com/support/software/installers/ud\">labjack.com</a></li>"
        "<li>Run the installer as Administrator</li>"
        "<li><code>LabJackUD.dll</code> is placed in <code>C:\\Windows\\System32</code></li>"
        "<li>Check Device Manager under \"LabJack\" if the device is not recognised</li>"
        "</ol>"
        "<p>Only 64-bit Windows is supported.</p>"
    );
}

#endif // Q_OS_WIN
