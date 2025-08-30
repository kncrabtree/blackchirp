#include "spectrumlibrary.h"

#include <QDir>
#include <QStandardPaths>

// Static instance for singleton pattern
SpectrumLibrary* SpectrumLibrary::s_instance = nullptr;

SpectrumLibrary::SpectrumLibrary(QObject *parent)
    : VendorLibrary(BC::Key::Spectrum::spectrumM4i, parent)
{
    // Attempt to load the library on construction
    loadLibrary();
}

SpectrumLibrary& SpectrumLibrary::instance()
{
    if (!s_instance) {
        s_instance = new SpectrumLibrary();
    }
    return *s_instance;
}

QStringList SpectrumLibrary::platformLibraryNames() const
{
    QStringList names;
    
#ifdef Q_OS_LINUX
    // Linux variants
    names << "spcm_linux"           // Standard Linux library name
          << "libspcm_linux.so"     // With lib prefix and extension
          << "libspcm_linux.so.1"   // With version number
          << "libspcm_linux.so.2"   // Different version
          << "spcm"                  // Generic name
          << "libspcm.so";           // Generic with extension
          
#elif defined(Q_OS_WIN)
    // Windows variants
    names << "spcm_win64.dll"       // 64-bit Windows
          << "spcm_win32.dll"       // 32-bit Windows  
          << "spcm64.dll"           // Alternative 64-bit name
          << "spcm32.dll"           // Alternative 32-bit name
          << "spcm.dll";            // Generic Windows name
          
#elif defined(Q_OS_MACOS)
    // macOS variants
    names << "libspcm.dylib"        // Standard macOS library
          << "libspcm.1.dylib"      // With version
          << "libspcm_osx.dylib"    // macOS-specific name
          << "spcm";                // Generic name
          
#else
    // Generic Unix variants as fallback
    names << "libspcm.so"
          << "libspcm.so.1"
          << "spcm_linux"
          << "spcm";
#endif

    return names;
}

QStringList SpectrumLibrary::defaultSearchPaths() const
{
    QStringList paths;
    
#ifdef Q_OS_LINUX
    // Linux-specific paths
    paths << "/opt/spectrum/lib"                    // Default Spectrum installation
          << "/opt/spectrum/lib64"                  // 64-bit variant
          << "/usr/local/spectrum/lib"              // Alternative installation
          << "/usr/local/lib"                       // Manual installation
          << "/usr/lib"                             // System library
          << "/usr/lib64"                           // 64-bit system library
          << "/usr/lib/x86_64-linux-gnu"           // Ubuntu/Debian 64-bit
          << "/usr/lib/i386-linux-gnu";             // Ubuntu/Debian 32-bit
          
#elif defined(Q_OS_WIN)
    // Windows-specific paths
    paths << "C:/Program Files/Spectrum/lib"       // 64-bit Program Files
          << "C:/Program Files (x86)/Spectrum/lib" // 32-bit Program Files
          << "C:/Spectrum/lib"                      // Root installation
          << QDir::homePath() + "/Spectrum/lib";    // User installation
          
#elif defined(Q_OS_MACOS)
    // macOS-specific paths
    paths << "/opt/spectrum/lib"                    // Similar to Linux
          << "/usr/local/spectrum/lib"
          << "/usr/local/lib"
          << "/opt/local/lib"                       // MacPorts
          << "/usr/local/Cellar/spectrum/lib";      // Homebrew (if available)
          
#endif

    // Add environment variable paths if set
    QString spectrumPath = qgetenv("SPECTRUM_LIB_PATH");
    if (!spectrumPath.isEmpty()) {
        paths.prepend(spectrumPath); // Give priority to user-specified path
    }
    
    QString spectrumHome = qgetenv("SPECTRUM_HOME");
    if (!spectrumHome.isEmpty()) {
        paths.prepend(QDir(spectrumHome).absoluteFilePath("lib"));
    }
    
    return paths;
}

void SpectrumLibrary::loadFunctions()
{
    // Reset all function pointers
    spcm_hOpen = nullptr;
    spcm_vClose = nullptr;
    spcm_dwSetParam_i32 = nullptr;
    spcm_dwGetParam_i32 = nullptr;
    spcm_dwSetParam_i64 = nullptr;
    spcm_dwGetParam_i64 = nullptr;
    spcm_dwDefTransfer_i64 = nullptr;
    spcm_dwInvalidateBuf = nullptr;
    spcm_dwGetErrorInfo_i32 = nullptr;
    
    // Load required functions
    spcm_hOpen = reinterpret_cast<spcm_hOpen_t>(resolveFunction("spcm_hOpen"));
    spcm_vClose = reinterpret_cast<spcm_vClose_t>(resolveFunction("spcm_vClose"));
    spcm_dwSetParam_i32 = reinterpret_cast<spcm_dwSetParam_i32_t>(resolveFunction("spcm_dwSetParam_i32"));
    spcm_dwGetParam_i32 = reinterpret_cast<spcm_dwGetParam_i32_t>(resolveFunction("spcm_dwGetParam_i32"));
    spcm_dwSetParam_i64 = reinterpret_cast<spcm_dwSetParam_i64_t>(resolveFunction("spcm_dwSetParam_i64"));
    spcm_dwGetParam_i64 = reinterpret_cast<spcm_dwGetParam_i64_t>(resolveFunction("spcm_dwGetParam_i64"));
    spcm_dwDefTransfer_i64 = reinterpret_cast<spcm_dwDefTransfer_i64_t>(resolveFunction("spcm_dwDefTransfer_i64"));
    spcm_dwInvalidateBuf = reinterpret_cast<spcm_dwInvalidateBuf_t>(resolveFunction("spcm_dwInvalidateBuf"));
    spcm_dwGetErrorInfo_i32 = reinterpret_cast<spcm_dwGetErrorInfo_i32_t>(resolveFunction("spcm_dwGetErrorInfo_i32"));
    
    // Check if all essential functions were loaded
    QStringList missingFunctions;
    
    if (!spcm_hOpen) missingFunctions << "spcm_hOpen";
    if (!spcm_vClose) missingFunctions << "spcm_vClose";
    if (!spcm_dwSetParam_i32) missingFunctions << "spcm_dwSetParam_i32";
    if (!spcm_dwGetParam_i32) missingFunctions << "spcm_dwGetParam_i32";
    
    if (!missingFunctions.isEmpty()) {
        d_libraryLoaded = false;
        d_errorString = QString("Failed to resolve essential Spectrum functions: %1")
                       .arg(missingFunctions.join(", "));
        return;
    }
    
    // Optional functions - warn if missing but don't fail
    QStringList optionalMissing;
    if (!spcm_dwSetParam_i64) optionalMissing << "spcm_dwSetParam_i64";
    if (!spcm_dwGetParam_i64) optionalMissing << "spcm_dwGetParam_i64";
    if (!spcm_dwDefTransfer_i64) optionalMissing << "spcm_dwDefTransfer_i64";
    if (!spcm_dwInvalidateBuf) optionalMissing << "spcm_dwInvalidateBuf";
    if (!spcm_dwGetErrorInfo_i32) optionalMissing << "spcm_dwGetErrorInfo_i32";
    
    if (!optionalMissing.isEmpty()) {
        // Note: We could emit a warning here, but for now we'll just continue
        // Optional functions missing may indicate an older library version
    }
    
    // All essential functions loaded successfully
    d_libraryLoaded = true;
}

QString SpectrumLibrary::getVersionInfo() const
{
    if (!isAvailable() || !spcm_dwGetParam_i32) {
        return QString();
    }
    
    // Try to get version info without a device handle (nullptr)
    // This may work for driver-level version queries
    std::int32_t driverVersion = 0;
    std::int32_t kernelVersion = 0;
    
    QString versionInfo;
    
    try {
        // Get driver version (using constants from spectrumconstants.h)
        std::int32_t result = spcm_dwGetParam_i32(nullptr, 1200 /* SPC_GETDRVVERSION */, &driverVersion);
        if (result == 0 /* ERR_OK */) {
            versionInfo += QString("Driver: %1.%2.%3")
                           .arg((driverVersion >> 24) & 0xFF)
                           .arg((driverVersion >> 16) & 0xFF)
                           .arg(driverVersion & 0xFFFF);
        }
        
        // Get kernel version
        result = spcm_dwGetParam_i32(nullptr, 1210 /* SPC_GETKERNELVERSION */, &kernelVersion);
        if (result == 0 /* ERR_OK */) {
            if (!versionInfo.isEmpty()) {
                versionInfo += ", ";
            }
            versionInfo += QString("Kernel: %1.%2.%3")
                           .arg((kernelVersion >> 24) & 0xFF)
                           .arg((kernelVersion >> 16) & 0xFF)
                           .arg(kernelVersion & 0xFFFF);
        }
    } catch (...) {
        // If nullptr causes issues, fall back gracefully
        return "Available";
    }
    
    return versionInfo.isEmpty() ? "Available" : versionInfo;
}