#include "vendorlibrary.h"

#include <QStandardPaths>
#include <QDir>
#include <QFileInfo>
#include <QCoreApplication>

VendorLibrary::VendorLibrary(const QString& libraryKey, QObject *parent)
    : QObject(parent), SettingsStorage({BC::Key::VendorLib::vendorLibraries, libraryKey}, SettingsStorage::General), 
      d_libraryLoaded(false), d_loadingAttempted(false), d_libraryKey(libraryKey)
{
    // Connect to library destruction for cleanup
    connect(&d_library, &QObject::destroyed, this, &VendorLibrary::onLibraryDestroyed);
}

QString VendorLibrary::loadedLibraryPath() const
{
    if (d_libraryLoaded && d_library.isLoaded()) {
        return d_library.fileName();
    }
    return QString();
}

bool VendorLibrary::loadLibrary()
{
    d_loadingAttempted = true;
    d_libraryLoaded = false;
    d_attemptedPaths.clear();
    d_errorString.clear();

    // Get platform-specific library names and build complete search paths
    QStringList libraryNames = platformLibraryNames();
    QStringList searchPaths = buildSearchPathList();
    
    // Add system library paths
    QStringList systemPaths = QCoreApplication::libraryPaths();
    systemPaths << "/usr/lib" << "/usr/local/lib" << "/opt/lib";
    
    // Build comprehensive list of paths to try
    QStringList allPaths;
    
    // First try library names alone for system path searching
    allPaths << libraryNames;
    
    // Then try each search path combined with each library name
    for (const QString& searchPath : searchPaths) {
        for (const QString& libName : libraryNames) {
            QString fullPath = QDir(searchPath).absoluteFilePath(libName);
            allPaths << fullPath;
        }
    }
    
    // Finally try system paths with library names
    for (const QString& systemPath : systemPaths) {
        for (const QString& libName : libraryNames) {
            QString fullPath = QDir(systemPath).absoluteFilePath(libName);
            allPaths << fullPath;
        }
    }
    
    // Try each combination
    for (const QString& path : allPaths) {
        d_attemptedPaths << path;
        
        d_library.setFileName(path);
        
        if (d_library.load()) {
            // Library loaded successfully, now load function pointers
            loadFunctions();
            
            if (d_libraryLoaded) {
                // Functions loaded successfully - save this path for future use
                saveSuccessfulLoad(path);
                return true;
            } else {
                // Function loading failed, unload library and continue trying
                d_library.unload();
            }
        }
    }
    
    // All loading attempts failed
    if (d_errorString.isEmpty()) {
        d_errorString = QString("Failed to load library '%1'. Tried %2 paths including: %3. Last error: %4")
                       .arg(libraryName())
                       .arg(d_attemptedPaths.size())
                       .arg(d_attemptedPaths.mid(0, 5).join(", ")) // Show first 5 attempts
                       .arg(d_library.errorString());
    }
    
    return false;
}

QFunctionPointer VendorLibrary::resolveFunction(const char* functionName)
{
    if (!d_library.isLoaded()) {
        return nullptr;
    }
    
    return d_library.resolve(functionName);
}

void VendorLibrary::onLibraryDestroyed()
{
    // Library is being destroyed, clean up our state
    d_libraryLoaded = false;
    d_errorString = "Library was unloaded";
}

void VendorLibrary::setUserProvidedPath(const QString& path)
{
    set(BC::Key::VendorLib::userProvidedPath, path);
}

QString VendorLibrary::getUserProvidedPath() const
{
    return get<QString>(BC::Key::VendorLib::userProvidedPath, QString());
}

void VendorLibrary::addUserSearchPath(const QString& path)
{
    QStringList paths = getUserSearchPaths();
    if (!paths.contains(path)) {
        paths.append(path);
        setUserSearchPaths(paths);
    }
}

void VendorLibrary::setUserSearchPaths(const QStringList& paths)
{
    set(BC::Key::VendorLib::searchPaths, paths);
}

QStringList VendorLibrary::getUserSearchPaths() const
{
    return get<QStringList>(BC::Key::VendorLib::searchPaths, QStringList());
}

void VendorLibrary::setAutoDiscoveryEnabled(bool enabled)
{
    set(BC::Key::VendorLib::enableAutoDiscovery, enabled);
}

bool VendorLibrary::isAutoDiscoveryEnabled() const
{
    return get<bool>(BC::Key::VendorLib::enableAutoDiscovery, true); // Default to enabled
}

bool VendorLibrary::reloadLibrary()
{
    // Unload current library if loaded
    if (d_library.isLoaded()) {
        d_library.unload();
    }
    
    // Reset state
    d_libraryLoaded = false;
    d_loadingAttempted = false;
    
    // Attempt to load again
    return loadLibrary();
}

QStringList VendorLibrary::buildSearchPathList() const
{
    QStringList allPaths;
    
    // 1. User-provided specific path takes highest priority
    QString userPath = getUserProvidedPath();
    if (!userPath.isEmpty()) {
        allPaths << userPath;
    }
    
    // 2. Last working path (if different from user path)
    QString lastWorking = get<QString>(BC::Key::VendorLib::lastWorkingPath, QString());
    if (!lastWorking.isEmpty() && lastWorking != userPath) {
        allPaths << lastWorking;
    }
    
    // 3. User-specified search paths
    QStringList userSearchPaths = getUserSearchPaths();
    allPaths << userSearchPaths;
    
    // 4. Default search paths (if auto-discovery is enabled)
    if (isAutoDiscoveryEnabled()) {
        QStringList defaultPaths = defaultSearchPaths();
        allPaths << defaultPaths;
    }
    
    return allPaths;
}

void VendorLibrary::saveSuccessfulLoad(const QString& successfulPath)
{
    set(BC::Key::VendorLib::lastWorkingPath, successfulPath);
}