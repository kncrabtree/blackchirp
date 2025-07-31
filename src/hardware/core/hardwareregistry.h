#ifndef HARDWAREREGISTRY_H
#define HARDWAREREGISTRY_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QHash>
#include <QMutex>
#include <functional>
#include <memory>

class HardwareObject;
class VendorLibrary;

/*!
 * \brief Hardware availability status
 */
enum class HardwareAvailability {
    Available,        /*!< Hardware is available and can be instantiated */
    Unavailable,      /*!< Hardware is not available (missing dependencies, libraries, etc.) */
    Unknown           /*!< Availability has not been checked yet */
};

/*!
 * \brief Hardware registration information
 */
struct HardwareRegistration {
    QString key;                                               /*!< Hardware type key (e.g., "ftmwDigitizer") */
    QString subKey;                                            /*!< Implementation key (e.g., "m4i2220x8") */
    QString prettyName;                                        /*!< Display name for UI */
    QString description;                                       /*!< Description of the hardware */
    QStringList dependencies;                                  /*!< Required vendor libraries or other dependencies */
    std::function<HardwareObject*()> factory;                 /*!< Factory function to create hardware instance */
    std::function<bool()> availabilityCheck;                  /*!< Function to check if hardware is available */
    HardwareAvailability availability = HardwareAvailability::Unknown; /*!< Cached availability status */
    bool isRequired = false;                                   /*!< Whether this hardware is required for operation */
    
    // Constructor
    HardwareRegistration() = default;
    HardwareRegistration(const QString& k, const QString& sk, const QString& name,
                        const QString& desc, const QStringList& deps,
                        std::function<HardwareObject*()> fact,
                        std::function<bool()> availCheck,
                        bool required = false)
        : key(k), subKey(sk), prettyName(name), description(desc),
          dependencies(deps), factory(fact), availabilityCheck(availCheck),
          isRequired(required) {}
};

/*!
 * \brief Hardware Registry for runtime hardware management
 * 
 * The HardwareRegistry provides a centralized system for registering hardware
 * implementations and checking their runtime availability. This enables dynamic
 * hardware configuration without compile-time dependencies on vendor libraries.
 * 
 * Key features:
 * - Runtime availability checking for hardware implementations
 * - Dependency validation (vendor libraries, drivers, etc.)
 * - Factory pattern for hardware instantiation
 * - Thread-safe access to hardware registry
 * - Categorization of hardware by type (required vs optional)
 * 
 * Usage example:
 * ```cpp
 * // Register hardware
 * HardwareRegistry::instance().registerHardware(
 *     "ftmwDigitizer", "m4i2220x8", "Spectrum M4i.2220-x8",
 *     "High-speed digitizer for FTMW spectroscopy",
 *     {"spectrum"}, // dependency on SpectrumLibrary
 *     []() -> HardwareObject* { return new M4i2220x8(); },
 *     []() -> bool { return SpectrumLibrary::instance().isAvailable(); },
 *     true // required hardware
 * );
 * 
 * // Check availability
 * if (HardwareRegistry::instance().isHardwareAvailable("ftmwDigitizer", "m4i2220x8")) {
 *     auto hw = HardwareRegistry::instance().createHardware("ftmwDigitizer", "m4i2220x8");
 * }
 * ```
 */
class HardwareRegistry : public QObject
{
    Q_OBJECT
    
public:
    /*!
     * \brief Get singleton instance of HardwareRegistry
     * \return Reference to the single HardwareRegistry instance
     */
    static HardwareRegistry& instance();
    
    /*!
     * \brief Register a hardware implementation
     * \param key Hardware type key (e.g., "ftmwDigitizer")
     * \param subKey Implementation key (e.g., "m4i2220x8")
     * \param prettyName Display name for UI
     * \param description Description of the hardware
     * \param dependencies List of required dependencies (vendor libraries, etc.)
     * \param factory Factory function to create hardware instances
     * \param availabilityCheck Function to check if hardware is available
     * \param required Whether this hardware is required for operation
     * \return True if registration was successful
     */
    bool registerHardware(const QString& key, const QString& subKey, const QString& prettyName,
                          const QString& description, const QStringList& dependencies,
                          std::function<HardwareObject*()> factory,
                          std::function<bool()> availabilityCheck,
                          bool required = false);
    
    /*!
     * \brief Check if a specific hardware implementation is available
     * \param key Hardware type key
     * \param subKey Implementation key
     * \return True if hardware is available and can be instantiated
     */
    bool isHardwareAvailable(const QString& key, const QString& subKey);
    
    /*!
     * \brief Create an instance of the specified hardware
     * \param key Hardware type key
     * \param subKey Implementation key
     * \return Pointer to created hardware object, or nullptr if unavailable
     */
    HardwareObject* createHardware(const QString& key, const QString& subKey);
    
    /*!
     * \brief Get list of all registered hardware implementations for a type
     * \param key Hardware type key
     * \return List of implementation keys for the specified hardware type
     */
    QStringList getAvailableImplementations(const QString& key);
    
    /*!
     * \brief Get list of all registered hardware types
     * \return List of hardware type keys
     */
    QStringList getRegisteredHardwareTypes();
    
    /*!
     * \brief Get registration information for specific hardware
     * \param key Hardware type key
     * \param subKey Implementation key
     * \return Hardware registration information, or nullptr if not found
     */
    const HardwareRegistration* getRegistration(const QString& key, const QString& subKey);
    
    /*!
     * \brief Refresh availability status for all registered hardware
     * 
     * This function re-runs availability checks for all registered hardware
     * implementations. Useful when vendor libraries may have been installed
     * or uninstalled since program startup.
     */
    void refreshAvailability();
    
    /*!
     * \brief Check if all required hardware is available
     * \return True if all required hardware implementations are available
     */
    bool allRequiredHardwareAvailable();
    
    /*!
     * \brief Get list of unavailable required hardware
     * \return List of unavailable required hardware descriptions
     */
    QStringList getUnavailableRequiredHardware();
    
    /*!
     * \brief Get default implementation for a hardware type
     * 
     * Returns the first available implementation for the specified hardware type,
     * prioritizing required hardware implementations.
     * 
     * \param key Hardware type key
     * \return Implementation key of default implementation, or empty string if none available
     */
    QString getDefaultImplementation(const QString& key);

signals:
    /*!
     * \brief Emitted when hardware availability status changes
     * \param key Hardware type key
     * \param subKey Implementation key
     * \param available New availability status
     */
    void hardwareAvailabilityChanged(const QString& key, const QString& subKey, bool available);
    
    /*!
     * \brief Emitted when hardware is successfully registered
     * \param key Hardware type key
     * \param subKey Implementation key
     */
    void hardwareRegistered(const QString& key, const QString& subKey);

private:
    explicit HardwareRegistry(QObject *parent = nullptr);
    ~HardwareRegistry() = default;
    
    // Singleton - disable copy/assignment
    HardwareRegistry(const HardwareRegistry&) = delete;
    HardwareRegistry& operator=(const HardwareRegistry&) = delete;
    
    /*!
     * \brief Generate a unique registry key for hardware type and implementation
     * \param key Hardware type key
     * \param subKey Implementation key
     * \return Unique registry key
     */
    QString makeRegistryKey(const QString& key, const QString& subKey) const;
    
    static HardwareRegistry* s_instance;
    QHash<QString, HardwareRegistration> d_registrations;  /*!< All registered hardware */
    mutable QMutex d_registryMutex;                        /*!< Thread safety for registry access */
};

#endif // HARDWAREREGISTRY_H