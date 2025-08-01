#ifndef RUNTIMEHARDWARECONFIG_H
#define RUNTIMEHARDWARECONFIG_H

#include <QString>
#include <QStringList>
#include <QHash>
#include <QReadWriteLock>
#include <map>

#include <data/storage/settingsstorage.h>

// Forward declarations for friend classes
class HardwareManager;
class RuntimeHardwareConfigTest;
class RuntimeHardwareConfigSimpleTest;

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
    friend class RuntimeHardwareConfigSimpleTest;  /*!< Simple test class needs access to private methods */
    // TODO: Add HardwareSettingsDialog and other authorized writers as needed
    
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
    
    // ========================================================================
    // READ-ONLY OPERATIONS (Thread-safe, public access)
    // ========================================================================
    
    /*!
     * \brief Get hardware selection for a hardware type
     * \param hardwareType Hardware type key (e.g., "ftmwDigitizer")
     * \return Implementation key, or empty string if not set or disabled
     */
    QString getHardwareSelection(const QString& hardwareType) const;
    
    /*!
     * \brief Check if hardware type is enabled
     * \param hardwareType Hardware type key
     * \return True if hardware type is enabled
     */
    bool isHardwareEnabled(const QString& hardwareType) const;
    
    /*!
     * \brief Get current hardware configuration as map
     * 
     * Returns the hardware configuration in the same format used by
     * HardwareManager::currentHardware() for compatibility with existing
     * experiment validation and UI code.
     * 
     * \return Map of hardware type key to implementation key (enabled hardware only)
     */
    std::map<QString, QString> getCurrentHardware() const;
    
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
     * \brief Validate specific hardware type configuration
     * \param hardwareType Hardware type key
     * \return Validation result for the specified hardware type
     */
    HardwareValidationResult validateHardwareType(const QString& hardwareType) const;
    
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
    bool isHardwareRequired(const QString& hardwareType) const;
    
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
    // WRITE OPERATIONS (Friend class access only)
    // ========================================================================
    
    /*!
     * \brief Set hardware selection for a hardware type
     * \param hardwareType Hardware type key (e.g., "ftmwDigitizer")
     * \param implementation Implementation key (e.g., "m4i2220x8")
     * \param enabled Whether this hardware should be enabled
     * \return True if selection was set successfully
     */
    bool setHardwareSelection(const QString& hardwareType, 
                             const QString& implementation,
                             bool enabled = true);
    
    /*!
     * \brief Enable or disable a hardware type
     * \param hardwareType Hardware type key
     * \param enabled Whether to enable the hardware
     * \return True if setting was applied successfully
     */
    bool setHardwareEnabled(const QString& hardwareType, bool enabled);
    
    /*!
     * \brief Remove hardware selection for a hardware type
     * \param hardwareType Hardware type key
     * \return True if selection was removed
     */
    bool removeHardwareSelection(const QString& hardwareType);
    
    /*!
     * \brief Clear all hardware selections
     */
    void clearConfiguration();
    
    /*!
     * \brief Setup default configuration based on available hardware
     * 
     * Queries the HardwareRegistry and sets up a default configuration
     * with the first available implementation for each hardware type.
     * This is used for initial setup or when starting fresh.
     */
    void setupDefaultConfiguration();
    
    /*!
     * \brief Load configuration from SettingsStorage
     * 
     * Reads the saved hardware configuration from persistent storage
     * into the in-memory data structures. Called automatically during
     * initialization.
     */
    void loadFromSettings();
    
    /*!
     * \brief Save current configuration to SettingsStorage
     * 
     * Writes the current in-memory hardware configuration to persistent
     * storage. Should be called when configuration changes need to be
     * persisted (e.g., when settings dialog closes, program exit).
     */
    void saveToSettings();
    
    // ========================================================================
    // PRIVATE IMPLEMENTATION
    // ========================================================================
    
    /*!
     * \brief Hardware configuration entry
     */
    struct HardwareConfig {
        QString implementation;     /*!< Selected implementation key */
        bool enabled = true;        /*!< Whether hardware is enabled */
    };
    
    static RuntimeHardwareConfig* s_instance;  /*!< Singleton instance */
    mutable QReadWriteLock d_configLock;       /*!< Thread-safe access control */
    
    QHash<QString, HardwareConfig> d_hardwareConfig; /*!< In-memory hardware configuration */
    
    /*!
     * \brief Internal validation helper that assumes caller holds lock
     * \param hardwareType Hardware type key
     * \param config Hardware configuration to validate
     * \return Validation result
     */
    HardwareValidationResult validateHardwareTypeInternal(const QString& hardwareType, const HardwareConfig& config) const;
    
    /*!
     * \brief Internal helper to get missing required hardware (assumes caller holds lock)
     * \return List of missing required hardware types
     */
    QStringList getMissingRequiredHardwareInternal() const;
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