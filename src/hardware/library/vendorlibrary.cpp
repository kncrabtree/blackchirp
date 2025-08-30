#include "vendorlibrary.h"

#include <QStandardPaths>
#include <QDir>
#include <QFileInfo>
#include <QCoreApplication>

VendorLibrary::VendorLibrary(const QString& libraryKey, QObject *parent)
    : QObject(parent), SettingsStorage({BC::Key::VendorLib::vendorLibraries, libraryKey}, SettingsStorage::General), 
      d_libraryLoaded(false), d_loadingAttempted(false), d_libraryKey(libraryKey),
      d_hasUnstagedChanges(false)
{
    // Connect to library destruction for cleanup
    connect(&d_library, &QObject::destroyed, this, &VendorLibrary::onLibraryDestroyed);
    
    // Initialize both active and staged configurations from persistent storage
    d_activeUserPath = get<QString>(BC::Key::VendorLib::userProvidedPath, QString());
    d_activeSearchPaths = get<QStringList>(BC::Key::VendorLib::searchPaths, QStringList());
    d_activeAutoDiscovery = get<bool>(BC::Key::VendorLib::enableAutoDiscovery, true); // Default to enabled
    
    // Initialize staged configuration to match active (no changes pending)
    d_stagedUserPath = d_activeUserPath;
    d_stagedSearchPaths = d_activeSearchPaths;
    d_stagedAutoDiscovery = d_activeAutoDiscovery;
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
    d_activeUserPath = path;
    d_stagedUserPath = path;
    d_hasUnstagedChanges = false;
}

QString VendorLibrary::getUserProvidedPath() const
{
    return d_activeUserPath;
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
    d_activeSearchPaths = paths;
    d_stagedSearchPaths = paths;
    d_hasUnstagedChanges = false;
}

QStringList VendorLibrary::getUserSearchPaths() const
{
    return d_activeSearchPaths;
}

void VendorLibrary::setAutoDiscoveryEnabled(bool enabled)
{
    set(BC::Key::VendorLib::enableAutoDiscovery, enabled);
    d_activeAutoDiscovery = enabled;
    d_stagedAutoDiscovery = enabled;
    d_hasUnstagedChanges = false;
}

bool VendorLibrary::isAutoDiscoveryEnabled() const
{
    return d_activeAutoDiscovery;
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
    
    // 1. Active user-provided specific path takes highest priority
    QString userPath = d_activeUserPath;
    if (!userPath.isEmpty()) {
        allPaths << userPath;
    }
    
    // 2. Last working path (if different from user path)
    QString lastWorking = get<QString>(BC::Key::VendorLib::lastWorkingPath, QString());
    if (!lastWorking.isEmpty() && lastWorking != userPath) {
        allPaths << lastWorking;
    }
    
    // 3. Active user-specified search paths
    QStringList userSearchPaths = d_activeSearchPaths;
    allPaths << userSearchPaths;
    
    // 4. Default search paths (if active auto-discovery is enabled)
    if (d_activeAutoDiscovery) {
        QStringList defaultPaths = defaultSearchPaths();
        allPaths << defaultPaths;
    }
    
    return allPaths;
}

void VendorLibrary::saveSuccessfulLoad(const QString& successfulPath)
{
    set(BC::Key::VendorLib::lastWorkingPath, successfulPath);
}

// Staging setter methods (UI uses these - no immediate effect)

void VendorLibrary::setStagedUserProvidedPath(const QString& path)
{
    if (d_stagedUserPath != path) {
        d_stagedUserPath = path;
        d_hasUnstagedChanges = (d_stagedUserPath != d_activeUserPath || 
                               d_stagedSearchPaths != d_activeSearchPaths ||
                               d_stagedAutoDiscovery != d_activeAutoDiscovery);
    }
}

void VendorLibrary::setStagedSearchPaths(const QStringList& paths)
{
    if (d_stagedSearchPaths != paths) {
        d_stagedSearchPaths = paths;
        d_hasUnstagedChanges = (d_stagedUserPath != d_activeUserPath || 
                               d_stagedSearchPaths != d_activeSearchPaths ||
                               d_stagedAutoDiscovery != d_activeAutoDiscovery);
    }
}

void VendorLibrary::setStagedAutoDiscoveryEnabled(bool enabled)
{
    if (d_stagedAutoDiscovery != enabled) {
        d_stagedAutoDiscovery = enabled;
        d_hasUnstagedChanges = (d_stagedUserPath != d_activeUserPath || 
                               d_stagedSearchPaths != d_activeSearchPaths ||
                               d_stagedAutoDiscovery != d_activeAutoDiscovery);
    }
}

// Staging getter methods

QString VendorLibrary::getStagedUserProvidedPath() const
{
    return d_stagedUserPath;
}

QStringList VendorLibrary::getStagedSearchPaths() const
{
    return d_stagedSearchPaths;
}

bool VendorLibrary::isStagedAutoDiscoveryEnabled() const
{
    return d_stagedAutoDiscovery;
}

bool VendorLibrary::hasUnstagedChanges() const
{
    return d_hasUnstagedChanges;
}

// Active configuration getter methods (hardware uses these)

QString VendorLibrary::getActiveUserProvidedPath() const
{
    return d_activeUserPath;
}

QStringList VendorLibrary::getActiveSearchPaths() const
{
    return d_activeSearchPaths;
}

bool VendorLibrary::isActiveAutoDiscoveryEnabled() const
{
    return d_activeAutoDiscovery;
}

// Synchronization methods

bool VendorLibrary::applyChanges()
{
    if (!d_hasUnstagedChanges) {
        return true; // No changes to apply
    }
    
    // Save staged settings to persistent storage
    set(BC::Key::VendorLib::userProvidedPath, d_stagedUserPath);
    set(BC::Key::VendorLib::searchPaths, d_stagedSearchPaths);
    set(BC::Key::VendorLib::enableAutoDiscovery, d_stagedAutoDiscovery);
    
    // Promote staged → active
    d_activeUserPath = d_stagedUserPath;
    d_activeSearchPaths = d_stagedSearchPaths;
    d_activeAutoDiscovery = d_stagedAutoDiscovery;
    d_hasUnstagedChanges = false;
    
    // Reload library with new active settings
    return reloadLibrary();
}

void VendorLibrary::revertChanges()
{
    // Reset staged configuration to match active configuration
    d_stagedUserPath = d_activeUserPath;
    d_stagedSearchPaths = d_activeSearchPaths;
    d_stagedAutoDiscovery = d_activeAutoDiscovery;
    d_hasUnstagedChanges = false;
}