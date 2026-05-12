#ifndef HARDWAREPROFILEMANAGER_H
#define HARDWAREPROFILEMANAGER_H

#include <optional>

#include <QString>
#include <QStringList>
#include <QHash>
#include <QDateTime>
#include <QByteArray>
#include <QReadWriteLock>
#include <QMutex>

#include <data/storage/settingsstorage.h>

// Forward declarations for friend classes
class HardwareProfileManagerTest;

/*!
 * \brief Portable snapshot of one hardware profile for import and export operations
 *
 * Carries the type, label, implementation, active state, timestamps, and
 * description that together identify and describe a single hardware profile.
 * Used by HardwareProfileManager::exportProfiles() and importProfiles().
 */
struct HardwareProfileData {
    QString type;           /*!< Hardware type (e.g., "FlowController") */
    QString label;          /*!< User-defined label (e.g., "mainDevice") */
    QString implementation; /*!< Implementation key (e.g., "mks647c") */
    bool active = true;     /*!< Whether profile is active */
    QDateTime created;      /*!< Profile creation timestamp */
    QDateTime modified;     /*!< Profile last modified timestamp */
    QString description;    /*!< User description of the profile */
    
    HardwareProfileData() = default;
    HardwareProfileData(const QString& t, const QString& l, const QString& i, bool a = true)
        : type(t), label(l), implementation(i), active(a), 
          created(QDateTime::currentDateTime()), modified(QDateTime::currentDateTime()) {}
};

/*!
 * \brief Singleton manager for hardware profiles.
 *
 * A profile is a (hardware type, label, implementation) triple — fixed
 * at creation — together with its persisted settings. The hardware
 * type and label form the profile's identity (e.g.
 * \c "FlowController.frontPanel"); the implementation key is stored
 * as a profile field but is immutable, so changing the implementation
 * requires creating a new profile under a new label. Profile
 * identities survive hardware reconfiguration and are the canonical
 * reference for a hardware object's settings group.
 *
 * Labels are unique within a hardware type and reusable across types.
 * Persistence rides on SettingsStorage; all operations are thread-safe
 * via an internal QReadWriteLock (multiple readers, exclusive writer).
 * Mutating operations return explicit success/failure values rather
 * than substituting defaults, so callers must handle errors.
 */
class HardwareProfileManager : public SettingsStorage
{
    // Friend classes for testing
    friend class HardwareProfileManagerTest;
    
public:
    /*!
     * \brief Get singleton instance
     * 
     * All operations are thread-safe due to internal QReadWriteLock usage.
     * 
     * \return Reference to the singleton instance
     */
    static HardwareProfileManager& instance();
    /*!
     * \brief Collision resolution strategies when labels conflict
     */
    enum CollisionAction {
        NoCollision,  /*!< No collision detected */
        Rename,       /*!< Automatically rename the new profile */
        Replace,      /*!< Replace the existing profile */
        Restore,      /*!< Keep existing profile, discard new */
        Cancel        /*!< Cancel the operation */
    };
    
    /*!
     * \brief Label validation error types
     */
    enum LabelValidationError {
        Valid,              /*!< Label is valid */
        Empty,              /*!< Label is empty or whitespace only */
        TooLong,            /*!< Label exceeds maximum length */
        InvalidCharacters,  /*!< Label contains invalid characters */
        StartsWithNumber,   /*!< Label starts with a number */
        StartsWithUnderscore, /*!< Label starts with underscore */
        ContainsDots        /*!< Label contains dots (conflicts with key format) */
    };
    
    /*!
     * \brief Destructor - saves profiles to persistent storage
     */
    virtual ~HardwareProfileManager();
    
    // ========================================================================
    // PROFILE MANAGEMENT
    // ========================================================================
    
    /*!
     * \brief Create a new hardware profile
     * 
     * Creates a new profile with the specified implementation and label.
     * If requestedLabel is empty, a default label will be auto-generated.
     * If requestedLabel conflicts with existing profile, collision resolution
     * depends on the collisionAction parameter.
     * 
     * \param type Hardware type (e.g., "FlowController", "FtmwDigitizer")
     * \param implementation Implementation key (e.g., "mks647c", "virtual")
     * \param requestedLabel Desired label for the profile (auto-generated if empty)
     * \param collisionAction How to handle label collisions (default: Rename)
     * \return Actual label used for the profile, or empty string if creation failed
     */
    QString createHardwareProfile(const QString& type, 
                                 const QString& implementation,
                                 const QString& requestedLabel = QString(),
                                 CollisionAction collisionAction = Rename);
    
    /*!
     * \brief Delete a hardware profile
     * \param type Hardware type
     * \param label Profile label to delete
     * \return True if profile was deleted successfully
     */
    bool deleteHardwareProfile(const QString& type, const QString& label);
    
    /*!
     * \brief Activate a hardware profile
     * \param type Hardware type
     * \param label Profile label to activate
     * \return True if profile was activated successfully
     */
    bool activateHardwareProfile(const QString& type, const QString& label);
    
    /*!
     * \brief Deactivate a hardware profile
     * \param type Hardware type
     * \param label Profile label to deactivate
     * \return True if profile was deactivated successfully
     */
    bool deactivateHardwareProfile(const QString& type, const QString& label);
    
    /*!
     * \brief Check if a profile exists
     * \param type Hardware type
     * \param label Profile label
     * \return True if profile exists
     */
    bool profileExists(const QString& type, const QString& label) const;
    
    /*!
     * \brief Check if a profile is active
     * \param type Hardware type
     * \param label Profile label
     * \return True if profile exists and is active
     */
    bool isProfileActive(const QString& type, const QString& label) const;
    
    // ========================================================================
    // LABEL MANAGEMENT
    // ========================================================================
    
    /*!
     * \brief Check if a label is available for use within a hardware type
     * \param type Hardware type
     * \param label Label to check
     * \return True if label is available (not already used in this type)
     */
    bool isLabelAvailable(const QString& type, const QString& label) const;
    
    /*!
     * \brief Generate a default label for a hardware type
     *
     * Picks the first available label from a fixed candidate list:
     * "Default", "Main", "Primary", "Secondary", "Backup". If all of those
     * are taken, falls back to "{type}1", "{type}2", etc.
     *
     * \param type Hardware type
     * \return Generated default label that is available
     */
    QString generateDefaultLabel(const QString& type) const;
    
    /*!
     * \brief Get all existing labels for a hardware type
     * \param type Hardware type
     * \return List of labels currently used by this hardware type
     */
    QStringList getExistingLabels(const QString& type) const;
    
    /*!
     * \brief Validate a label according to naming rules
     * \param label Label to validate
     * \return Validation result indicating if label is valid or why it's invalid
     */
    LabelValidationError validateLabel(const QString& label) const;
    
    /*!
     * \brief Check if a label is valid (convenience function)
     * \param label Label to check
     * \return True if label passes all validation rules
     */
    bool isValidLabel(const QString& label) const;
    
    /*!
     * \brief Get maximum allowed label length
     * \return Maximum number of characters allowed in a label
     */
    int getMaxLabelLength() const { return 64; }
    
    // ========================================================================
    // PROFILE QUERIES
    // ========================================================================
    
    /*!
     * \brief Get all active profiles for a hardware type
     * \param type Hardware type
     * \return List of labels for active profiles
     */
    QStringList getActiveProfiles(const QString& type) const;
    
    /*!
     * \brief Get all inactive profiles for a hardware type
     * \param type Hardware type
     * \return List of labels for inactive profiles
     */
    QStringList getInactiveProfiles(const QString& type) const;
    
    /*!
     * \brief Get all profiles (active and inactive) for a hardware type
     * \param type Hardware type
     * \return List of all labels for this hardware type
     */
    QStringList getAllProfiles(const QString& type) const;
    
    /*!
     * \brief Get implementation for a specific profile
     * \param type Hardware type
     * \param label Profile label
     * \return Implementation key, or empty string if profile doesn't exist
     */
    QString getImplementation(const QString& type, const QString& label) const;

    /*!
     * \brief Get threading override for a specific profile
     * \param type Hardware type
     * \param label Profile label
     * \return Threading override, or nullopt if no override is stored (use type-level default)
     */
    std::optional<bool> getThreaded(const QString& type, const QString& label) const;

    /*!
     * \brief Set threading override for a specific profile
     * \param type Hardware type
     * \param label Profile label
     * \param threaded Threading override value
     * \return True if successfully set
     */
    bool setThreaded(const QString& type, const QString& label, bool threaded);

    /*!
     * \brief Get Python script path for a specific profile
     * \param type Hardware type
     * \param label Profile label
     * \return Python script path, or empty string if not set or profile doesn't exist
     */
    QString getPythonScriptPath(const QString& type, const QString& label) const;

    /*!
     * \brief Set Python script path for a specific profile
     * \param type Hardware type
     * \param label Profile label
     * \param path Python script path
     * \return True if successfully set
     */
    bool setPythonScriptPath(const QString& type, const QString& label, const QString& path);

    /*!
     * \brief Get Python class name for a specific profile
     * \param type Hardware type
     * \param label Profile label
     * \return Python class name, or empty string if not set or profile doesn't exist
     */
    QString getPythonClassName(const QString& type, const QString& label) const;

    /*!
     * \brief Set Python class name for a specific profile
     * \param type Hardware type
     * \param label Profile label
     * \param name Python class name
     * \return True if successfully set
     */
    bool setPythonClassName(const QString& type, const QString& label, const QString& name);

    /*!
     * \brief Get Python environment path for a specific profile
     * \param type Hardware type
     * \param label Profile label
     * \return Path to venv/conda environment directory, or empty string if not set
     */
    QString getPythonEnvPath(const QString& type, const QString& label) const;

    /*!
     * \brief Set Python environment path for a specific profile
     * \param type Hardware type
     * \param label Profile label
     * \param path Path to venv/conda environment directory (empty = system python3)
     * \return True if successfully set
     */
    bool setPythonEnvPath(const QString& type, const QString& label, const QString& path);

    /*!
     * \brief Get all hardware types that have profiles
     * \return List of hardware types with at least one profile
     */
    QStringList getConfiguredHardwareTypes() const;
    
    // ========================================================================
    // PROFILE METADATA
    // ========================================================================
    
    /*!
     * \brief Get profile creation timestamp
     * \param type Hardware type
     * \param label Profile label
     * \return Creation timestamp, or invalid QDateTime if profile doesn't exist
     */
    QDateTime getProfileCreationTime(const QString& type, const QString& label) const;
    
    /*!
     * \brief Get profile last modified timestamp
     * \param type Hardware type
     * \param label Profile label
     * \return Last modified timestamp, or invalid QDateTime if profile doesn't exist
     */
    QDateTime getProfileLastModified(const QString& type, const QString& label) const;
    
    /*!
     * \brief Set profile description
     * \param type Hardware type
     * \param label Profile label
     * \param description User description of the profile
     * \return True if description was set successfully
     */
    bool setProfileDescription(const QString& type, const QString& label, const QString& description);
    
    /*!
     * \brief Get profile description
     * \param type Hardware type
     * \param label Profile label
     * \return Profile description, or empty string if not set or profile doesn't exist
     */
    QString getProfileDescription(const QString& type, const QString& label) const;
    
    // ========================================================================
    // COLLISION HANDLING
    // ========================================================================
    
    /*!
     * \brief Detect if creating a profile would cause a collision
     * \param type Hardware type
     * \param label Requested label
     * \param implementation Implementation key
     * \return Collision type detected
     */
    CollisionAction detectCollision(const QString& type, const QString& label, 
                                   const QString& implementation) const;
    
    /*!
     * \brief Resolve label collision by generating alternative label
     * \param type Hardware type
     * \param baseLabel Base label that caused collision
     * \return Alternative label that is available
     */
    QString resolveCollisionByRename(const QString& type, const QString& baseLabel) const;
    
    // ========================================================================
    // BULK OPERATIONS
    // ========================================================================
    
    /*!
     * \brief Activate all profiles for a hardware type
     * \param type Hardware type
     * \return True if all profiles were activated successfully
     */
    bool activateAllProfiles(const QString& type);
    
    /*!
     * \brief Deactivate all profiles for a hardware type
     * \param type Hardware type
     * \return True if all profiles were deactivated successfully
     */
    bool deactivateAllProfiles(const QString& type);
    
    /*!
     * \brief Delete all profiles for a hardware type
     * \param type Hardware type
     * \return True if all profiles were deleted successfully
     */
    bool deleteAllProfiles(const QString& type);
    
    /*!
     * \brief Clear all profiles from all hardware types
     */
    void clearAllProfiles();
    
    // ========================================================================
    // IMPORT/EXPORT FUNCTIONALITY
    // ========================================================================
    
    /*!
     * \brief Export all profiles to binary data
     * \return Serialized profile data suitable for storage or transfer
     */
    QByteArray exportProfiles() const;
    
    /*!
     * \brief Export profiles for specific hardware type
     * \param type Hardware type to export
     * \return Serialized profile data for the specified type
     */
    QByteArray exportProfiles(const QString& type) const;
    
    /*!
     * \brief Import profiles from binary data
     * \param data Serialized profile data (from exportProfiles)
     * \param collisionAction How to handle label collisions during import
     * \return True if import was successful
     */
    bool importProfiles(const QByteArray& data, CollisionAction collisionAction = Rename);
    
    /*!
     * \brief Import a single profile
     * \param profileData Profile data to import
     * \param collisionAction How to handle label collisions
     * \return True if profile was imported successfully
     */
    bool importProfile(const HardwareProfileData& profileData, 
                      CollisionAction collisionAction = Rename);
    
    /*!
     * \brief Get profile data for export/backup purposes
     * \param type Hardware type
     * \param label Profile label
     * \return Profile data structure, or invalid data if profile doesn't exist
     */
    HardwareProfileData getProfileData(const QString& type, const QString& label) const;
    
    // ========================================================================
    // PERSISTENCE MANAGEMENT
    // ========================================================================
    
    /*!
     * \brief Force save all profiles to persistent storage
     * 
     * Profiles are automatically saved when the manager is destroyed,
     * but this method allows explicit saving.
     */
    void saveProfiles();
    
    /*!
     * \brief Load profiles from persistent storage
     * 
     * Called automatically during construction, but can be used to
     * reload from storage if external changes occurred.
     */
    void loadProfiles();
    
    /*!
     * \brief Check if profiles have been modified since last save
     * \return True if there are unsaved changes
     */
    bool hasUnsavedChanges() const;

    /*!
     * \brief Check if a profile is a system-protected profile that cannot be removed
     * \param hwType Hardware type string
     * \param label Profile label
     * \return True if this is a system profile (label == "virtual" for a required hardware type)
     */
    static bool isSystemProfile(const QString& hwType, const QString& label);

    /*!
     * \brief Ensure system profiles exist for all required hardware types
     *
     * Called at application startup and when the hardware config dialog opens.
     * Creates a "virtual" profile with the appropriate virtual implementation for
     * each required hardware type that does not already have one.
     *
     * Required type -> virtual implementation mapping:
     *   FtmwDigitizer  -> VirtualFtmwDigitizer
     *   Clock      -> FixedClock
     *   LifDigitizer   -> VirtualLifDigitizer   (when LIF enabled)
     *   LifLaser   -> VirtualLifLaser   (when LIF enabled)
     */
    void ensureSystemProfiles();

private:
    // ========================================================================
    // INTERNAL DATA STRUCTURES
    // ========================================================================
    
    /*!
     * \brief Internal profile data structure
     */
    struct ProfileInfo {
        QString implementation;     /*!< Implementation key */
        bool active = true;         /*!< Whether profile is active */
        QDateTime created;          /*!< Creation timestamp */
        QDateTime modified;         /*!< Last modified timestamp */
        QString description;        /*!< User description */
        std::optional<bool> threaded; /*!< Threading override (nullopt = use type-level default) */
        QString pythonScriptPath;    /*!< Python script path (only used by Python hardware types) */
        QString pythonClassName;     /*!< Python class name (only used by Python hardware types) */
        QString pythonEnvPath;       /*!< Python environment directory (venv/conda; empty = system python3) */

        ProfileInfo() : created(QDateTime::currentDateTime()),
                       modified(QDateTime::currentDateTime()) {}
        ProfileInfo(const QString& impl, bool act = true)
            : implementation(impl), active(act),
              created(QDateTime::currentDateTime()),
              modified(QDateTime::currentDateTime()) {}
    };
    
    // Thread-safe storage: hardware type -> label -> profile info
    mutable QReadWriteLock d_profilesLock;
    QHash<QString, QHash<QString, ProfileInfo>> d_profiles;
    
    mutable QMutex d_modifiedFlagLock;
    bool d_modified = false;
    
    // ========================================================================
    // INTERNAL HELPER METHODS
    // ========================================================================
    
    /*!
     * \brief Load profiles from SettingsStorage into memory structures
     */
    void loadProfilesFromSettings();
    
    /*!
     * \brief Save profiles from memory structures to SettingsStorage
     */
    void saveProfilesToSettings();
    
    
    /*!
     * \brief Mark profiles as modified (thread-safe)
     */
    void setModified();
    
    /*!
     * \brief Internal label validation implementation
     * \param label Label to validate
     * \return Validation error or Valid
     */
    LabelValidationError validateLabelInternal(const QString& label) const;
    
    /*!
     * \brief Internal collision detection (assumes lock is held)
     * \param type Hardware type
     * \param label Requested label
     * \param implementation Implementation key
     * \return Collision type
     */
    CollisionAction detectCollisionInternal(const QString& type, const QString& label,
                                           const QString& implementation) const;
    
    /*!
     * \brief Internal profile creation (assumes lock is held)
     * \param type Hardware type
     * \param implementation Implementation key
     * \param label Profile label (must be validated and available)
     * \return True if profile was created
     */
    bool createProfileInternal(const QString& type, const QString& implementation, 
                              const QString& label);
    
    /*!
     * \brief Internal default label generation (assumes lock is held)
     * \param type Hardware type
     * \return Generated label
     */
    QString generateDefaultLabelInternal(const QString& type) const;
    
    /*!
     * \brief Update profile modification timestamp (assumes lock is held)
     * \param type Hardware type
     * \param label Profile label
     */
    void updateModificationTime(const QString& type, const QString& label);
    
    /*!
     * \brief Internal collision resolution (assumes caller holds lock)
     * \param type Hardware type  
     * \param baseLabel Base label that caused collision
     * \return Alternative label that is available
     */
    QString resolveCollisionByRenameInternal(const QString& type, const QString& baseLabel) const;
    
    // ========================================================================
    // SINGLETON INFRASTRUCTURE
    // ========================================================================
    
    /*!
     * \brief Private constructor for singleton pattern
     */
    HardwareProfileManager();
    
    /*!
     * \brief Private constructor with custom organization and application names
     * \param orgName Organization name for QSettings
     * \param appName Application name for QSettings
     */
    HardwareProfileManager(const QString& orgName, const QString& appName);
    
    // Disable copy/assignment for singleton
    HardwareProfileManager(const HardwareProfileManager&) = delete;
    HardwareProfileManager& operator=(const HardwareProfileManager&) = delete;
    
    static HardwareProfileManager* s_instance;  /*!< Singleton instance */
};

/*!
 * \brief Settings keys for hardware profile storage
 */
namespace BC::Key::HardwareProfiles {
    inline constexpr QLatin1StringView profiles{"HardwareProfiles"};      /*!< Base settings group */
    inline constexpr QLatin1StringView implementation{"implementation"};   /*!< Implementation subkey */
    inline constexpr QLatin1StringView active{"active"};                  /*!< Active state subkey */
    inline constexpr QLatin1StringView created{"created"};                /*!< Creation time subkey */
    inline constexpr QLatin1StringView modified{"modified"};              /*!< Modified time subkey */
    inline constexpr QLatin1StringView description{"description"};        /*!< Description subkey */
    inline constexpr QLatin1StringView threaded{"threaded"};              /*!< Threading override subkey */
    inline constexpr QLatin1StringView pythonScriptPath{"pythonScriptPath"}; /*!< Python script path subkey */
    inline constexpr QLatin1StringView pythonClassName{"pythonClassName"};   /*!< Python class name subkey */
    inline constexpr QLatin1StringView pythonEnvPath{"pythonEnvPath"};       /*!< Python environment directory subkey */
}

Q_DECLARE_METATYPE(HardwareProfileData)

#endif // HARDWAREPROFILEMANAGER_H