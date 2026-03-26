#ifndef RUNTIMEHARDWARECONFIG_H
#define RUNTIMEHARDWARECONFIG_H

#include <optional>

#include <QString>
#include <QStringList>
#include <QHash>
#include <QReadWriteLock>
#include <map>

#include <data/storage/settingsstorage.h>
#include <data/experiment/hardwaredatacontainer.h>

// Include all hardware headers so template methods work properly
#include <hardware/core/ftmwdigitizer/ftmwscope.h>
#include <hardware/optional/chirpsource/awg.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <hardware/optional/flowcontroller/flowcontroller.h>
#include <hardware/optional/ioboard/ioboard.h>
#include <hardware/optional/gpibcontroller/gpibcontroller.h>
#include <hardware/optional/pressurecontroller/pressurecontroller.h>
#include <hardware/optional/tempcontroller/temperaturecontroller.h>

#include <hardware/core/lifdigitizer/lifscope.h>
#include <hardware/core/liflaser/liflaser.h>

// Forward declarations for friend classes
class HardwareManager;
class RuntimeHardwareConfigTest;
class RuntimeHardwareConfigSimpleTest;
class RuntimeHardwareConfigDialog;

/*!
 * \brief Runtime hardware configuration validation result
 */
struct HardwareValidationResult {
    bool isValid = false;                    /*!< Whether configuration is valid */
    QString selectedImplementation;          /*!< Actually selected implementation */
    QStringList warnings;                    /*!< Non-fatal configuration warnings */
    QStringList errors;                      /*!< Fatal configuration errors */
    
    HardwareValidationResult() = default;
    explicit HardwareValidationResult(bool valid) : isValid(valid) {}
};

/*!
 * \brief Singleton runtime hardware configuration manager
 * 
 * This class provides thread-safe runtime hardware configuration management using
 * SettingsStorage for persistence. Access is controlled through the singleton pattern
 * with read/write locking and friend class restrictions for modifications.
 * 
 * Design principles:
 * - Singleton with global read access, controlled write access
 * - Thread-safe with QReadWriteLock (multiple readers, exclusive writer)
 * - No automatic fallbacks (explicit error handling required)
 * - SettingsStorage integration for persistence
 * - Friend class pattern for write access control
 * 
 * Usage example:
 * ```cpp
 * // Read access (anywhere in the program)
 * const auto& config = RuntimeHardwareConfig::constInstance();
 * QString ftmwImpl = config.getHardwareSelection("ftmwDigitizer");
 * bool isValid = config.isConfigurationValid();
 * 
 * // Write access (only friend classes like HardwareManager)
 * auto& config = RuntimeHardwareConfig::instance();
 * config.setHardwareSelection("ftmwDigitizer", "m4i2220x8");
 * ```
 */
class RuntimeHardwareConfig : public SettingsStorage
{
    // Friend classes that can modify configuration
    friend class HardwareManager;
    friend class RuntimeHardwareConfigTest;  /*!< Test class needs access to private methods */
    friend class RuntimeHardwareConfigDialog;  /*!< Dialog needs access for preview/accept operations */
    
public:
    /*!
     * \brief Get const singleton instance for read-only access
     * 
     * This method provides thread-safe read-only access to the configuration
     * from anywhere in the program.
     * 
     * \return Const reference to the singleton instance
     */
    static const RuntimeHardwareConfig& constInstance();
    
    /*!
     * \brief Template helper for compile-time type-safe hardware type resolution
     * 
     * Extracts the hardware type string from the class name using Qt's meta-object system.
     * This provides compile-time type safety and automatic updates when classes are renamed.
     * 
     * \tparam T Hardware class type (e.g., VirtualFtmwScope, FixedClock)
     * \return Hardware type string for use with RuntimeHardwareConfig
     * 
     * \example
     * auto labels = RuntimeHardwareConfig::constInstance().getActiveLabels(hardwareTypeOf<VirtualFtmwScope>());
     */
    template<typename T>
    static QString hardwareTypeOf() {
        return QString(T::staticMetaObject.className());
    }
    
    // ========================================================================
    // READ-ONLY OPERATIONS (Thread-safe, public access)
    // ========================================================================
    
    /*!
     * \brief Get hardware implementation for a specific label (type-safe)
     * \tparam T Hardware class type (e.g., FtmwScope, PulseGenerator)
     * \param label Hardware label (e.g., "frontPanel", "backup")
     * \return Implementation key, or empty string if not set or disabled
     */
    template<typename T>
    QString getHardwareImplementation(const QString& label) const {
        return getHardwareImplementation(hardwareTypeOf<T>(), label);
    }
    
    /*!
     * \brief Get all active labels for a hardware type (type-safe)
     * \tparam T Hardware class type (e.g., FtmwScope, PulseGenerator)
     * \return List of labels for currently active hardware devices
     */
    template<typename T>
    QStringList getActiveLabels() const {
        return getActiveLabels(hardwareTypeOf<T>());
    }
    
    /*!
     * \brief Get all active hardware keys for a hardware type (type-safe)
     * \tparam T Hardware class type (e.g., FtmwScope, PulseGenerator)
     * \return List of full hwType.label keys for currently active hardware devices
     */
    template<typename T>
    QStringList getActiveKeys() const {
        return getActiveKeys(hardwareTypeOf<T>());
    }
    
    /*!
     * \brief Get current hardware configuration as map
     * 
     * Returns the hardware configuration for experiment validation and UI code.
     * 
     * \return Map of hardware type key to implementation key (enabled hardware only)
     */
    std::map<QString, QString> getCurrentHardware() const;
    
    /*!
     * \brief Create hardware data container for experiment classes
     * 
     * Creates a BC::Data::HardwareDataContainer populated with both hardware
     * selections and type keys. This provides a clean interface between the
     * hardware layer and data layer without creating circular dependencies.
     * 
     * The container includes:
     * - All active hardware selections in "type.label" -> "implementation" format
     * - Hardware type keys populated using template methods for type safety
     * - Convenience methods for type-safe hardware access
     * 
     * This method is used by application layer code when creating new experiments
     * to pass hardware information to data layer classes like Experiment.
     * 
     * \return HardwareDataContainer with populated hardware map and type keys
     */
    BC::Data::HardwareDataContainer createHardwareDataContainer() const;
    
    /*!
     * \brief Static method to validate any hardware configuration map
     * 
     * Checks all hardware selections in the provided configuration map against available implementations.
     * Does NOT perform automatic fallbacks - reports errors that must be handled explicitly.
     * This method can be used to validate arbitrary configuration maps without needing the singleton state.
     * 
     * \param hardwareMap Hardware configuration map ("hwType.label" -> "implementation")
     * \return List of validation error messages (empty list means configuration is valid)
     */
    static QStringList validateHardwareConfiguration(const std::map<QString, QString>& hardwareMap);
    
    /*!
     * \brief Validate entire hardware configuration
     * 
     * Checks all configured hardware selections against available implementations.
     * Does NOT perform automatic fallbacks - reports errors that must be handled explicitly.
     * 
     * \return Map of hardware type to validation result
     */
    QHash<QString, HardwareValidationResult> validateConfiguration() const;
    
    
    /*!
     * \brief Static method to check if hardware configuration is valid
     * 
     * \param hardwareMap Hardware configuration map ("hwType.label" -> "implementation")
     * \return True if configuration is valid (no validation errors)
     */
    static bool isHardwareConfigurationValid(const std::map<QString, QString>& hardwareMap);
    
    /*!
     * \brief Check if entire configuration is valid
     * \return True if all configured hardware is available and valid
     */
    bool isConfigurationValid() const;
    
    /*!
     * \brief Get all configured hardware types
     * \return List of hardware type keys that have selections
     */
    QStringList getConfiguredHardwareTypes() const;
    
    /*!
     * \brief Check if hardware type is required
     *
     * Checks the HardwareRegistry to determine if a hardware type
     * is marked as required for operation.
     *
     * \param hardwareType Hardware type key
     * \return True if hardware type is required
     */
    static bool isHardwareRequired(const QString& hardwareType);

    /*!
     * \brief Check if hardware type is a LIF hardware type
     *
     * Returns true for hardware types associated with LIF (LifScope, LifLaser).
     * Used to filter LIF hardware from the runtime config when LIF is disabled.
     *
     * \param hardwareType Hardware type key
     * \return True if hardware type is a LIF type
     */
    static bool isLifHardwareType(const QString& hardwareType);
    
    /*!
     * \brief Get list of missing required hardware
     * \return List of required hardware types that are not configured or unavailable
     */
    QStringList getMissingRequiredHardware() const;
    
    /*!
     * \brief Get validation errors for all hardware
     * \return List of all validation errors across all hardware types
     */
    QStringList getAllValidationErrors() const;
    
    /*!
     * \brief Get validation warnings for all hardware
     * \return List of all validation warnings across all hardware types
     */
    QStringList getAllValidationWarnings() const;

    /*!
     * \brief Activate system profiles for required hardware types that have no active config
     *
     * Called at startup after ensureSystemProfiles(). For each required hardware type
     * with no active entry, activates the "virtual" system profile.
     */
    void activateMissingSystemProfiles();

    /*!
     * \brief Get the threading override for a hardware key
     * \param hwKey Full hardware key (e.g., "FtmwScope.virtual")
     * \return Threading override, or nullopt if none stored (use type-level default from constructor)
     */
    std::optional<bool> getThreaded(const QString& hwKey) const;

private:
    // ========================================================================
    // SINGLETON MANAGEMENT
    // ========================================================================

    /*!
     * \brief Get mutable singleton instance for write access
     *
     * This method is private and only accessible to friend classes.
     * It provides write access to the configuration.
     *
     * \return Mutable reference to the singleton instance
     */
    static RuntimeHardwareConfig& instance();
    
    /*!
     * \brief Private constructor for singleton pattern
     */
    RuntimeHardwareConfig();
    
    /*!
     * \brief Private destructor
     */
    ~RuntimeHardwareConfig() = default;
    
    // Disable copy/assignment for singleton
    RuntimeHardwareConfig(const RuntimeHardwareConfig&) = delete;
    RuntimeHardwareConfig& operator=(const RuntimeHardwareConfig&) = delete;
    
    // ========================================================================
    // INTERNAL READ OPERATIONS (String-based, used by template functions)
    // ========================================================================
    
    /*!
     * \brief Get hardware implementation for a specific label (string-based)
     * \param hardwareType Hardware type key (e.g., "FlowController")
     * \param label Hardware label (e.g., "frontPanel", "backup")
     * \return Implementation key, or empty string if not set or disabled
     */
    QString getHardwareImplementation(const QString& hardwareType, const QString& label) const;
    
    /*!
     * \brief Get all active labels for a hardware type (string-based)
     * \param hardwareType Hardware type key
     * \return List of labels for currently active hardware devices
     */
    QStringList getActiveLabels(const QString& hardwareType) const;
    
    /*!
     * \brief Get all active hardware keys for a hardware type (string-based)
     * \param hardwareType Hardware type key
     * \return List of full hwType.label keys for currently active hardware devices
     */
    QStringList getActiveKeys(const QString& hardwareType) const;
    
    // ========================================================================
    // WRITE OPERATIONS (Friend class access only)
    // ========================================================================
    
    /*!
     * \brief Set hardware selection for a specific label
     * \param hardwareType Hardware type key (e.g., "FlowController") 
     * \param label Hardware label (e.g., "frontPanel", "backup")
     * \param implementation Implementation key (e.g., "mks647c")
     * \return True if selection was set successfully
     */
    bool setHardwareSelection(const QString& hardwareType, 
                             const QString& label,
                             const QString& implementation);
    
    
    /*!
     * \brief Remove hardware selection for a specific label
     * \param hardwareType Hardware type key
     * \param label Hardware label
     * \return True if selection was removed
     */
    bool removeHardwareSelection(const QString& hardwareType, const QString& label);
    
    /*!
     * \brief Clear all hardware selections
     */
    void clearConfiguration();
    
    /*!
     * \brief Apply complete hardware configuration (for dialog accept operations)
     * \param config Hardware configuration map (hwType.label -> implementation)
     * \return True if configuration was applied successfully
     */
    bool applyConfiguration(const std::map<QString, QString>& config);

    /*!
     * \brief Set threading override for a hardware key
     * \param hwKey Full hardware key (e.g., "FtmwScope.virtual")
     * \param threaded Threading override value
     */
    void setThreaded(const QString& hwKey, bool threaded);
    
    
    
    
    // ========================================================================
    // PRIVATE IMPLEMENTATION
    // ========================================================================
    
    /*!
     * \brief Hardware selection entry for a specific device
     */
    struct HardwareSelection {
        QString type;               /*!< Hardware type (stored for easy filtering) */
        QString implementation;     /*!< Selected implementation key */
        std::optional<bool> threaded; /*!< Threading override (nullopt = use type-level default) */
    };
    
    static RuntimeHardwareConfig* s_instance;  /*!< Singleton instance */
    mutable QReadWriteLock d_configLock;       /*!< Thread-safe access control */
    
    QHash<QString, HardwareSelection> d_activeHardware; /*!< "type.label" key -> selection */
    
    /*!
     * \brief Internal validation helper that assumes caller holds lock
     * \param hardwareType Hardware type key
     * \param label Hardware label
     * \param selection Hardware selection to validate
     * \return Validation result
     */
    HardwareValidationResult validateHardwareSelectionInternal(const QString& hardwareType, 
                                                               const QString& label,
                                                               const HardwareSelection& selection) const;
    
    /*!
     * \brief Internal helper to get missing required hardware (assumes caller holds lock)
     * \return List of missing required hardware types
     */
    QStringList getMissingRequiredHardwareInternal() const;
    
    /*!
     * \brief Sync with HardwareProfileManager - load active profiles
     * 
     * Updates the runtime configuration to match active profiles from
     * HardwareProfileManager. Called during initialization and when
     * profiles are activated/deactivated.
     */
    void syncWithProfiles();
    
    /*!
     * \brief Activate profile in HardwareProfileManager
     * \param hardwareType Hardware type key
     * \param label Hardware label 
     */
    void activateProfile(const QString& hardwareType, const QString& label);
    
    /*!
     * \brief Deactivate profile in HardwareProfileManager
     * \param hardwareType Hardware type key
     * \param label Hardware label 
     */
    void deactivateProfile(const QString& hardwareType, const QString& label);
};

/*!
 * \brief Settings keys for runtime hardware configuration
 */
namespace BC::Key::RuntimeHw {
    static const QString runtimeHw{"runtimeHardware"};     /*!< Base settings key */
    static const QString selection{"selection"};           /*!< Selection subkey */
    static const QString enabled{"enabled"};               /*!< Enabled state subkey */
}

#endif // RUNTIMEHARDWARECONFIG_H