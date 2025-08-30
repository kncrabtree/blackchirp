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