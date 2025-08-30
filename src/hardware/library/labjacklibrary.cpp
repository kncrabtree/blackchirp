#include "labjacklibrary.h"
#include <QDebug>
#include <data/storage/settingsstorage.h>

LabjackLibrary* LabjackLibrary::s_instance = nullptr;

LabjackLibrary::LabjackLibrary(QObject *parent) 
    : VendorLibrary(BC::Key::LabJack::labjackU3, parent)
{
    // Attempt to load the library on construction (consistent with SpectrumLibrary)
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
    names << "liblabjackusb.so"
          << "liblabjackusb.so.2"
          << "liblabjackusb.so.2.7.0"
          << "liblabjackusb.so.2.6.0";
#elif defined(Q_OS_WIN)
    names << "labjackusb.dll";
#elif defined(Q_OS_MACOS)
    names << "liblabjackusb.dylib";
#else
    names << "liblabjackusb.so";
#endif
    
    return names;
}

QStringList LabjackLibrary::defaultSearchPaths() const
{
    QStringList paths;
    
#ifdef Q_OS_LINUX
    paths << "/usr/local/lib"
          << "/usr/lib"
          << "/usr/lib64"
          << "/usr/lib/x86_64-linux-gnu"
          << "/lib"
          << "/lib64";
#elif defined(Q_OS_WIN)
    paths << "C:/Windows/System32"
          << "C:/Program Files/LabJack/Drivers"
          << "C:/Program Files (x86)/LabJack/Drivers";
#elif defined(Q_OS_MACOS)
    paths << "/usr/local/lib"
          << "/usr/lib"
          << "/Library/Frameworks"
          << "/System/Library/Frameworks";
#endif
    
    return paths;
}

void LabjackLibrary::loadFunctions()
{
    if (!d_library.isLoaded()) {
        d_errorString = "Library not loaded";
        return;
    }
    
    // Load low-level LJUSB functions that u3.cpp actually uses
    LJUSB_GetLibraryVersion = reinterpret_cast<LJUSB_GetLibraryVersion_t>(d_library.resolve("LJUSB_GetLibraryVersion"));
    LJUSB_GetDevCount = reinterpret_cast<LJUSB_GetDevCount_t>(d_library.resolve("LJUSB_GetDevCount"));
    LJUSB_OpenDevice = reinterpret_cast<LJUSB_OpenDevice_t>(d_library.resolve("LJUSB_OpenDevice"));
    LJUSB_CloseDevice = reinterpret_cast<LJUSB_CloseDevice_t>(d_library.resolve("LJUSB_CloseDevice"));
    LJUSB_Write = reinterpret_cast<LJUSB_Write_t>(d_library.resolve("LJUSB_Write"));
    LJUSB_Read = reinterpret_cast<LJUSB_Read_t>(d_library.resolve("LJUSB_Read"));
    LJUSB_IsHandleValid = reinterpret_cast<LJUSB_IsHandleValid_t>(d_library.resolve("LJUSB_IsHandleValid"));
    LJUSB_ResetConnection = reinterpret_cast<LJUSB_ResetConnection_t>(d_library.resolve("LJUSB_ResetConnection"));
    
    // Verify essential functions are loaded
    if (!LJUSB_GetDevCount || !LJUSB_OpenDevice || !LJUSB_CloseDevice || 
        !LJUSB_Write || !LJUSB_Read) {
        d_errorString = "Failed to load essential LabJack USB functions";
        d_libraryLoaded = false;
        return;
    }
    
    // Test library functionality
    try {
        if (LJUSB_GetLibraryVersion) {
            float version = LJUSB_GetLibraryVersion();
            qDebug() << "LabJack USB library version:" << version;
        }
        
        d_libraryLoaded = true;
        d_errorString.clear();
        
        qDebug() << "LabJack USB library loaded successfully";
        
    } catch (...) {
        d_errorString = "Exception occurred while testing LabJack library functions";
        d_libraryLoaded = false;
    }
}

QString LabjackLibrary::getInstallationInstructions() const
{
#ifdef Q_OS_LINUX
    return QString(
        "<p><b>Installing LabJack USB Driver on Linux:</b></p>"
        "<ol>"
        "<li><b>Download:</b> Visit the <a href=\"https://labjack.com/support/software/installers/ud\">LabJack UD software page</a> and download the Linux UD installer</li>"
        "<li><b>Extract:</b> Extract the downloaded archive (typically named like <code>LabJackUD_*.tar.gz</code>)</li>"
        "<li><b>Install:</b> Run the installation script as root:"
        "<pre>cd LabJackUD_*/\nsudo ./install.sh</pre></li>"
        "<li><b>Libraries:</b> The installer will place files in:"
        "<ul>"
        "<li>USB Library: <code>/usr/local/lib/liblabjackusb.so</code></li>"
        "<li>UD Library: <code>/usr/local/lib/liblabjackud.so</code></li>"
        "<li>Headers: <code>/usr/local/include/labjack/</code></li>"
        "</ul></li>"
        "<li><b>Permissions:</b> Add udev rules for device access (done automatically by installer)</li>"
        "<li><b>Verify:</b> Check that the library is accessible:"
        "<pre>ldconfig -p | grep labjack</pre></li>"
        "</ol>"
        "<p><b>Manual Installation (Alternative):</b></p>"
        "<ul>"
        "<li>Install libusb development package: <code>sudo apt install libusb-1.0-0-dev</code></li>"
        "<li>Download source from GitHub: <a href=\"https://github.com/labjack/exodriver\">LabJack Exodriver</a></li>"
        "<li>Compile and install following the GitHub instructions</li>"
        "</ul>"
        "<p><b>Troubleshooting:</b></p>"
        "<ul>"
        "<li>If permission denied: Check udev rules in <code>/etc/udev/rules.d/</code></li>"
        "<li>If library not found: Run <code>sudo ldconfig</code> to update library cache</li>"
        "<li>Verify device detection: <code>lsusb | grep LabJack</code></li>"
        "</ul>"
    );
#elif defined(Q_OS_WIN)
    return QString(
        "<p><b>Installing LabJack USB Driver on Windows:</b></p>"
        "<ol>"
        "<li><b>Download:</b> Visit the <a href=\"https://labjack.com/support/software/installers/ud\">LabJack UD software page</a> and download the Windows UD installer</li>"
        "<li><b>Run Installer:</b> Execute the installer as Administrator (typically named like <code>LabJackUD_*.exe</code>)</li>"
        "<li><b>Installation Components:</b> The installer includes:"
        "<ul>"
        "<li>USB drivers for all LabJack devices</li>"
        "<li>LabJackUD library (32-bit and 64-bit versions)</li>"
        "<li>Example code and documentation</li>"
        "</ul></li>"
        "<li><b>Installation Path:</b> Files are typically installed to:"
        "<ul>"
        "<li>64-bit: <code>C:\\Windows\\System32\\LabJackUD.dll</code></li>"
        "<li>32-bit: <code>C:\\Windows\\SysWOW64\\LabJackUD.dll</code></li>"
        "<li>SDK: <code>C:\\Program Files\\LabJack\\</code></li>"
        "</ul></li>"
        "<li><b>Device Recognition:</b> Windows will automatically recognize LabJack devices after installation</li>"
        "</ol>"
        "<p><b>Troubleshooting:</b></p>"
        "<ul>"
        "<li>Run the installer as Administrator</li>"
        "<li>Check Device Manager for LabJack devices under \"LabJack\" category</li>"
        "<li>Ensure USB cable and device connections are secure</li>"
        "<li>Try different USB ports if device is not detected</li>"
        "</ul>"
    );
#elif defined(Q_OS_MACOS)
    return QString(
        "<p><b>Installing LabJack USB Driver on macOS:</b></p>"
        "<ol>"
        "<li><b>Download:</b> Visit the <a href=\"https://labjack.com/support/software/installers/ud\">LabJack UD software page</a> and download the macOS UD installer</li>"
        "<li><b>Install:</b> Run the installer package (.pkg file) and follow the installation wizard</li>"
        "<li><b>Installation Path:</b> Libraries are installed to:"
        "<ul>"
        "<li><code>/usr/local/lib/liblabjackusb.dylib</code></li>"
        "<li><code>/usr/local/lib/liblabjackud.dylib</code></li>"
        "<li>Headers: <code>/usr/local/include/labjack/</code></li>"
        "</ul></li>"
        "<li><b>Permissions:</b> Grant necessary permissions when prompted</li>"
        "<li><b>Verify:</b> Check library installation:"
        "<pre>ls -la /usr/local/lib/liblabjack*</pre></li>"
        "</ol>"
        "<p><b>Manual Installation (Homebrew):</b></p>"
        "<pre>brew install libusb\ngit clone https://github.com/labjack/exodriver.git\ncd exodriver\nmake install</pre>"
        "<p><b>Troubleshooting:</b></p>"
        "<ul>"
        "<li>For Apple Silicon Macs, ensure ARM64 compatibility</li>"
        "<li>Check System Preferences for security prompts</li>"
        "<li>Verify device detection: <code>system_profiler SPUSBDataType | grep LabJack</code></li>"
        "</ul>"
    );
#else
    return QString(
        "<p><b>Installing LabJack USB Driver:</b></p>"
        "<p>Please visit the <a href=\"https://labjack.com/support/software/installers/ud\">LabJack software download page</a> "
        "to download the appropriate driver package for your operating system.</p>"
        "<p>LabJack provides comprehensive installation instructions and support for all major operating systems.</p>"
        "<p>For additional help, refer to the <a href=\"https://labjack.com/support\">LabJack support documentation</a>.</p>"
    );
#endif
}