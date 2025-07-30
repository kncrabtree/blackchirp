#include "labjacklibrary.h"

#include <QDir>
#include <QStandardPaths>

// Static instance for singleton pattern
LabjackLibrary* LabjackLibrary::s_instance = nullptr;

LabjackLibrary::LabjackLibrary(QObject *parent)
    : VendorLibrary(BC::Key::LabJack::labjackU3, parent)
{
    // Attempt to load the library on construction
    loadLibrary();
}

LabjackLibrary& LabjackLibrary::instance()
{
    if (!s_instance) {
        s_instance = new LabjackLibrary();
    }
    return *s_instance;
}

QStringList LabjackLibrary::platformLibraryNames() const
{
    QStringList names;
    
#ifdef Q_OS_LINUX
    // Linux variants
    names << "labjackusb"              // Standard Linux library name
          << "liblabjackusb.so"        // With lib prefix and extension
          << "liblabjackusb.so.1"      // With version number
          << "liblabjackusb.so.2"      // Different version
          << "liblabjackusb.so.3"      // Different version
          << "ljacklm"                 // Alternative Linux name
          << "libljacklm.so";          // Alternative with extension
          
#elif defined(Q_OS_WIN)
    // Windows variants
    names << "labjackud.dll"          // Windows UD library
          << "ljacklm.dll"            // Legacy Windows library
          << "labjackusb.dll"         // USB-specific Windows library
          << "labjack.dll";           // Generic Windows name
          
#elif defined(Q_OS_MACOS)
    // macOS variants  
    names << "liblabjackusb.dylib"    // Standard macOS library
          << "liblabjackusb.1.dylib"  // With version
          << "libljacklm.dylib"       // Alternative macOS name
          << "labjack";               // Generic name
          
#else
    // Generic Unix variants as fallback
    names << "liblabjackusb.so"
          << "liblabjackusb.so.1"
          << "labjackusb"
          << "ljacklm";
#endif

    return names;
}

QStringList LabjackLibrary::defaultSearchPaths() const
{
    QStringList paths;
    
#ifdef Q_OS_LINUX
    // Linux-specific paths
    paths << "/opt/labjack/lib"                     // Default LabJack installation
          << "/opt/labjack/lib64"                   // 64-bit variant
          << "/usr/local/labjack/lib"               // Alternative installation
          << "/usr/local/lib"                       // Manual installation
          << "/usr/lib"                             // System library
          << "/usr/lib64"                           // 64-bit system library
          << "/usr/lib/x86_64-linux-gnu"           // Ubuntu/Debian 64-bit
          << "/usr/lib/i386-linux-gnu";             // Ubuntu/Debian 32-bit
          
#elif defined(Q_OS_WIN)
    // Windows-specific paths
    paths << "C:/Program Files/LabJack/lib"        // 64-bit Program Files
          << "C:/Program Files (x86)/LabJack/lib"  // 32-bit Program Files
          << "C:/LabJack/lib"                       // Root installation
          << QDir::homePath() + "/LabJack/lib";     // User installation
          
#elif defined(Q_OS_MACOS)
    // macOS-specific paths
    paths << "/opt/labjack/lib"                     // Similar to Linux
          << "/usr/local/labjack/lib"
          << "/usr/local/lib"
          << "/opt/local/lib"                       // MacPorts
          << "/usr/local/Cellar/labjack/lib";       // Homebrew (if available)
          
#endif

    // Add environment variable paths if set
    QString labjackPath = qgetenv("LABJACK_LIB_PATH");
    if (!labjackPath.isEmpty()) {
        paths.prepend(labjackPath); // Give priority to user-specified path
    }
    
    QString labjackHome = qgetenv("LABJACK_HOME");
    if (!labjackHome.isEmpty()) {
        paths.prepend(QDir(labjackHome).absoluteFilePath("lib"));
    }
    
    return paths;
}

void LabjackLibrary::loadFunctions()
{
    // Reset all function pointers
    openUSBConnection = nullptr;
    closeUSBConnection = nullptr;
    getCalibrationInfo = nullptr;
    eTCConfig = nullptr;
    eAIN = nullptr;
    eDI = nullptr;
    
    // Load required functions
    openUSBConnection = reinterpret_cast<openUSBConnection_t>(resolveFunction("openUSBConnection"));
    closeUSBConnection = reinterpret_cast<closeUSBConnection_t>(resolveFunction("closeUSBConnection"));
    getCalibrationInfo = reinterpret_cast<getCalibrationInfo_t>(resolveFunction("getCalibrationInfo"));
    eTCConfig = reinterpret_cast<eTCConfig_t>(resolveFunction("eTCConfig"));
    eAIN = reinterpret_cast<eAIN_t>(resolveFunction("eAIN"));
    eDI = reinterpret_cast<eDI_t>(resolveFunction("eDI"));
    
    // Check if all essential functions were loaded
    QStringList missingFunctions;
    
    if (!openUSBConnection) missingFunctions << "openUSBConnection";
    if (!closeUSBConnection) missingFunctions << "closeUSBConnection";
    if (!getCalibrationInfo) missingFunctions << "getCalibrationInfo";
    if (!eAIN) missingFunctions << "eAIN";
    if (!eDI) missingFunctions << "eDI";
    
    if (!missingFunctions.isEmpty()) {
        d_libraryLoaded = false;
        d_errorString = QString("Failed to resolve essential LabJack U3 functions: %1")
                       .arg(missingFunctions.join(", "));
        return;
    }
    
    // Optional functions - warn if missing but don't fail
    QStringList optionalMissing;
    if (!eTCConfig) optionalMissing << "eTCConfig";
    
    if (!optionalMissing.isEmpty()) {
        // Note: We could emit a warning here, but for now we'll just continue
        // Optional functions missing may indicate an older library version
    }
    
    // All essential functions loaded successfully
    d_libraryLoaded = true;
}