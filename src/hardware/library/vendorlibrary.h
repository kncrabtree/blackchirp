#ifndef VENDORLIBRARY_H
#define VENDORLIBRARY_H

#include <QObject>
#include <QLibrary>
#include <QString>
#include <QStringList>
#include <data/storage/settingsstorage.h>

/*!
 * \brief Abstract base class for all vendor library wrappers
 * 
 * This class provides a common interface for dynamically loading vendor libraries
 * at runtime using QLibrary. This eliminates compile-time dependencies on vendor
 * SDKs, allowing BlackChirp to be built and distributed as a single binary that
 * works with or without vendor libraries installed.
 * 
 * Each vendor library wrapper (e.g., SpectrumLibrary, LabjackLibrary) inherits
 * from this class and implements the specific function pointer loading for their
 * respective APIs.
 * 
 * The loading process follows this pattern:
 * 1. Attempt to load the library from standard system paths
 * 2. Try additional search paths specific to the vendor
 * 3. Resolve function symbols to function pointers
 * 4. Validate that all required functions were found
 * 
 * If any step fails, the library is marked as unavailable and error information
 * is stored for user feedback.
 */
/*!
 * Keys for storing vendor library configuration in SettingsStorage
 */
namespace BC::Key::VendorLib {
    static const QString vendorLibraries{"vendorLibraries"};           /*!< Root key for all vendor library settings */
    static const QString userProvidedPath{"userProvidedPath"};         /*!< User-specified library path */
    static const QString lastWorkingPath{"lastWorkingPath"};           /*!< Last successful library path */
    static const QString searchPaths{"searchPaths"};                   /*!< User-specified additional search paths */
    static const QString enableAutoDiscovery{"enableAutoDiscovery"};   /*!< Whether to enable automatic discovery */
}

class VendorLibrary : public QObject, public SettingsStorage
{
    Q_OBJECT

public:
    explicit VendorLibrary(const QString& libraryKey, QObject *parent = nullptr);
    virtual ~VendorLibrary() = default;

    /*!
     * \brief Check if the vendor library is available and loaded
     * \return true if library is loaded and all functions resolved
     */
    virtual bool isAvailable() const = 0;

    /*!
     * \brief Get error message if library loading failed
     * \return Error string describing why library is not available
     */
    virtual QString errorString() const = 0;

    /*!
     * \brief Get the base name of the library being loaded
     * \return Base library name (e.g., "spcm", "labjackusb")
     */
    virtual QString libraryName() const = 0;

    /*!
     * \brief Get platform-specific library names to try
     * \return List of library names with platform-specific variations
     * 
     * This allows handling libraries that have different names on different
     * platforms (e.g., "spcm_linux" on Linux, "spcm64.dll" on Windows)
     */
    virtual QStringList platformLibraryNames() const = 0;

    /*!
     * \brief Get default search paths for automatic discovery
     * \return List of default search paths for this library type
     * 
     * This should return platform-specific default paths where the library
     * is typically installed. User-specified paths take priority over these.
     */
    virtual QStringList defaultSearchPaths() const = 0;

    /*!
     * \brief Get library version information
     * \return Version string if available, empty string if not available or not supported
     * 
     * This method attempts to query version information from the loaded library.
     * Implementation is library-specific and may require temporary device access.
     */
    virtual QString getVersionInfo() const { return QString(); }

    /*!
     * \brief Get the full path where library was found (if loaded successfully)
     * \return Full path to loaded library, empty if not loaded
     */
    QString loadedLibraryPath() const;

    /*!
     * \brief Check if library loading was attempted
     * \return true if loading was attempted (successfully or not)
     */
    bool wasLoadingAttempted() const { return d_loadingAttempted; }

    /*!
     * \brief Set user-provided library path
     * \param path Full path to library file provided by user
     * 
     * This path takes priority over automatic discovery. If the path is
     * invalid or the library cannot be loaded, automatic discovery will
     * be used as fallback (if enabled).
     */
    void setUserProvidedPath(const QString& path);

    /*!
     * \brief Get user-provided library path
     * \return User-specified library path, empty if none set
     */
    QString getUserProvidedPath() const;

    /*!
     * \brief Add user-specified search path
     * \param path Directory path to add to search paths
     * 
     * User-specified search paths are tried before default search paths.
     */
    void addUserSearchPath(const QString& path);

    /*!
     * \brief Set user-specified search paths
     * \param paths List of directory paths to search
     */
    void setUserSearchPaths(const QStringList& paths);

    /*!
     * \brief Get user-specified search paths
     * \return List of user-specified search directories
     */
    QStringList getUserSearchPaths() const;

    /*!
     * \brief Enable or disable automatic discovery
     * \param enabled Whether to use automatic discovery as fallback
     * 
     * When disabled, only user-provided paths are used. When enabled (default),
     * automatic discovery is used if user-provided paths fail.
     */
    void setAutoDiscoveryEnabled(bool enabled);

    /*!
     * \brief Check if automatic discovery is enabled
     * \return true if automatic discovery is enabled
     */
    bool isAutoDiscoveryEnabled() const;

    /*!
     * \brief Force reload of library with current settings
     * \return true if library was successfully loaded
     * 
     * This can be called after changing user settings to attempt
     * loading with the new configuration.
     */
    bool reloadLibrary();

    // Staging API - UI uses these methods (no immediate effect)
    
    /*!
     * \brief Set staged user-provided library path
     * \param path Full path to library file provided by user
     * 
     * This modifies the staged configuration without immediately affecting
     * the active configuration used by hardware. Changes must be applied
     * via applyChanges() to take effect.
     */
    void setStagedUserProvidedPath(const QString& path);
    
    /*!
     * \brief Set staged user-specified search paths
     * \param paths List of directory paths to search
     * 
     * This modifies the staged configuration without immediately affecting
     * the active configuration used by hardware.
     */
    void setStagedSearchPaths(const QStringList& paths);
    
    /*!
     * \brief Set staged automatic discovery setting
     * \param enabled Whether to use automatic discovery as fallback
     * 
     * This modifies the staged configuration without immediately affecting
     * the active configuration used by hardware.
     */
    void setStagedAutoDiscoveryEnabled(bool enabled);
    
    // Staging query methods
    
    /*!
     * \brief Get staged user-provided library path
     * \return Staged user-specified library path, empty if none set
     */
    QString getStagedUserProvidedPath() const;
    
    /*!
     * \brief Get staged user-specified search paths
     * \return List of staged user-specified search directories
     */
    QStringList getStagedSearchPaths() const;
    
    /*!
     * \brief Check if staged automatic discovery is enabled
     * \return true if staged automatic discovery is enabled
     */
    bool isStagedAutoDiscoveryEnabled() const;
    
    /*!
     * \brief Check if there are unstaged changes
     * \return true if staged configuration differs from active configuration
     */
    bool hasUnstagedChanges() const;
    
    // Synchronization methods
    
    /*!
     * \brief Apply staged changes to active configuration
     * \return true if changes were applied successfully and library reloaded
     * 
     * This promotes the staged configuration to become the active configuration,
     * saves the settings to persistent storage, and reloads the library with
     * the new configuration.
     */
    bool applyChanges();
    
    /*!
     * \brief Revert staged changes to match active configuration
     * 
     * This discards any staged changes and resets the staged configuration
     * to match the current active configuration.
     */
    void revertChanges();
    
    // Active configuration access (hardware uses these)
    
    /*!
     * \brief Get active user-provided library path
     * \return Active user-specified library path, empty if none set
     */
    QString getActiveUserProvidedPath() const;
    
    /*!
     * \brief Get active user-specified search paths
     * \return List of active user-specified search directories
     */
    QStringList getActiveSearchPaths() const;
    
    /*!
     * \brief Check if active automatic discovery is enabled
     * \return true if active automatic discovery is enabled
     */
    bool isActiveAutoDiscoveryEnabled() const;

protected:
    /*!
     * \brief Attempt to load library using platform-specific names and paths
     * \return true if library was successfully loaded
     * 
     * This method tries each platform-specific library name in combination with
     * each search path until the library loads successfully. It handles both 
     * absolute paths and library names for system path searching.
     */
    bool loadLibrary();

    /*!
     * \brief Resolve a function symbol to a function pointer
     * \param functionName Name of function symbol to resolve
     * \return Function pointer or nullptr if not found
     * 
     * This is a convenience method for subclasses to resolve function symbols.
     * The library must be successfully loaded before calling this method.
     */
    QFunctionPointer resolveFunction(const char* functionName);

    /*!
     * \brief Called by subclasses to load their specific function pointers
     * 
     * Subclasses must implement this method to resolve all required function
     * symbols after the library has been successfully loaded. If any required
     * function cannot be resolved, this method should set d_libraryLoaded to
     * false and update d_errorString with appropriate error information.
     */
    virtual void loadFunctions() = 0;

    QLibrary d_library;               /*!< Qt library loader */
    bool d_libraryLoaded;             /*!< Whether library is loaded and ready */
    bool d_loadingAttempted;          /*!< Whether loading was attempted */
    QString d_errorString;            /*!< Error message if loading failed */
    QStringList d_attemptedPaths;     /*!< Paths that were tried during loading */
    QString d_libraryKey;             /*!< Settings key for this library */

    // Current active configuration (used by hardware)
    QString d_activeUserPath;         /*!< Active user-provided library path */
    QStringList d_activeSearchPaths;  /*!< Active user search paths */
    bool d_activeAutoDiscovery;       /*!< Active auto-discovery setting */
    
    // Staged configuration (modified by UI, not yet applied)
    QString d_stagedUserPath;         /*!< Staged user-provided library path */
    QStringList d_stagedSearchPaths;  /*!< Staged user search paths */
    bool d_stagedAutoDiscovery;       /*!< Staged auto-discovery setting */
    bool d_hasUnstagedChanges;        /*!< Track if staging differs from active */

private:
    /*!
     * \brief Build complete search path priority list
     * \return Ordered list of paths to try (user paths first, then defaults)
     */
    QStringList buildSearchPathList() const;

    /*!
     * \brief Save successful load information to settings
     * \param successfulPath Path that successfully loaded the library
     */
    void saveSuccessfulLoad(const QString& successfulPath);

private slots:
    /*!
     * \brief Handle library unloading
     * 
     * Connected to QLibrary destroyed signal to clean up state when
     * library is unloaded (e.g., during application shutdown).
     */
    void onLibraryDestroyed();
};

#endif // VENDORLIBRARY_H