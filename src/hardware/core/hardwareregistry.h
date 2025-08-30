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
 * \brief Hardware registration information
 */
struct HardwareRegistration {
    QString key;                                               /*!< Hardware type key (e.g., "ftmwDigitizer") */
    QString subKey;                                            /*!< Implementation key (e.g., "m4i2220x8") */
    QString description;                                       /*!< Description of the hardware */
    std::function<HardwareObject*(const QString&)> factory;   /*!< Factory function to create hardware instance with label */
    QStringList libraryDependencies;                          /*!< List of vendor libraries this hardware depends on */
    
    // Constructor
    HardwareRegistration() = default;
    HardwareRegistration(const QString& k, const QString& sk, const QString& desc, 
                        std::function<HardwareObject*(const QString&)> fact)
        : key(k), subKey(sk), description(desc), factory(fact), libraryDependencies({}) {}
};

/*!
 * \brief Hardware Registry for runtime hardware management
 * 
 * The HardwareRegistry provides a centralized catalog of hardware implementations
 * and factory functions for creating them. This is a pure registry system that
 * does NOT handle availability checking, dependencies, or fallback logic.
 * 
 * Key features:
 * - Simple registration of hardware implementations
 * - Factory pattern for hardware instantiation
 * - Thread-safe access to hardware registry
 * - Pure catalog - no runtime state or availability tracking
 * 
 * Responsibilities:
 * - DOES: Register implementations and create instances
 * - DOES NOT: Check availability, validate dependencies, or provide fallbacks
 * 
 * Usage example:
 * ```cpp
 * // Register hardware
 * HardwareRegistry::instance().registerHardware(
 *     "ftmwDigitizer", "m4i2220x8", "Spectrum M4i.2220-x8",
 *     "High-speed digitizer for FTMW spectroscopy",
 *     []() -> HardwareObject* { return new M4i2220x8(); }
 * );
 * 
 * // Create hardware (may return nullptr if factory fails)
 * auto hw = HardwareRegistry::instance().createHardware("ftmwDigitizer", "m4i2220x8");
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
     * \param description Description of the hardware (should include manufacturer and model)
     * \param factory Factory function to create hardware instances with label
     * \return True if registration was successful
     */
    bool registerHardware(const QString& key, const QString& subKey, const QString& description,
                          std::function<HardwareObject*(const QString&)> factory);
    
    /*!
     * \brief Create an instance of the specified hardware
     * \param key Hardware type key
     * \param subKey Implementation key
     * \param label Label to use for hardware instance identification
     * \return Pointer to created hardware object, or nullptr if factory fails
     */
    HardwareObject* createHardware(const QString& key, const QString& subKey, const QString& label);
    
    /*!
     * \brief Get list of all registered hardware implementations for a type
     * \param key Hardware type key
     * \return List of implementation keys for the specified hardware type
     */
    QStringList getImplementations(const QString& key);
    
    /*!
     * \brief Get list of all registered hardware types
     * \return List of hardware type keys
     */
    QStringList getHardwareTypes();
    
    /*!
     * \brief Get registration information for specific hardware
     * \param key Hardware type key
     * \param subKey Implementation key
     * \return Hardware registration information, or nullptr if not found
     */
    const HardwareRegistration* getRegistration(const QString& key, const QString& subKey);
    
    /*!
     * \brief Check if hardware implementation is registered
     * \param key Hardware type key
     * \param subKey Implementation key
     * \return True if hardware is registered in the catalog
     */
    bool isRegistered(const QString& key, const QString& subKey);

    /*!
     * \brief Check if hardware type supports multiple instances
     * \param hardwareType Hardware type key (e.g., "Clock", "FtmwScope")
     * \return True if hardware type can have multiple labeled instances, false for single-instance types
     * 
     * Single-instance types: FtmwScope, Awg, LifLaser, LifScope
     * Multi-instance types: Clock, PulseGenerator, FlowController, PressureController, TemperatureController, IOBoard, GPIBController
     */
    static bool isMultiInstanceType(const QString& hardwareType);

    /*!
     * \brief Get list of vendor libraries that a hardware implementation depends on
     * \param implementationName Hardware implementation key (e.g., "m4i2220x8", "labjacku3")
     * \return List of library names that the hardware depends on ("SpectrumLibrary", "LabjackLibrary", etc.)
     * 
     * This method identifies which vendor libraries a hardware implementation requires.
     * Hardware objects using these libraries must be destroyed before library changes
     * and recreated after library changes to prevent crashes from invalid function pointers.
     */
    QStringList getLibraryDependencies(const QString& implementationName) const;

    /*!
     * \brief Get list of hardware implementations that depend on a specific library
     * \param libraryName Library name (e.g., "SpectrumLibrary", "LabjackLibrary")
     * \return List of hardware implementation keys that depend on the specified library
     */
    QStringList getHardwareDependingOnLibrary(const QString& libraryName) const;

    /*!
     * \brief Check if hardware implementation uses specific library
     * \param implementationName Hardware implementation key
     * \param libraryName Library name
     * \return True if hardware implementation depends on the specified library
     */
    bool hardwareUsesLibrary(const QString& implementationName, const QString& libraryName) const;

    /*!
     * \brief Add library dependency to existing hardware registration
     * \param key Hardware type key
     * \param subKey Implementation key
     * \param libraryName Library name to add as dependency
     * \param libraryGetter Function to get library instance
     * \return True if dependency was added successfully
     * 
     * This method allows hardware implementations to register their library dependencies
     * after the initial hardware registration. Used by the REGISTER_LIBRARY macro.
     */
    bool addLibraryDependency(const QString& key, const QString& subKey, const QString& libraryName,
                              std::function<VendorLibrary*()> libraryGetter);

    /*!
     * \brief Get all libraries that have unstaged changes
     * \return List of library names that have unstaged changes
     * 
     * This method checks all registered library instances for unstaged changes,
     * providing a generic way to detect library configuration changes.
     */
    QStringList getLibrariesWithChanges() const;

signals:
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
    QHash<QString, std::function<VendorLibrary*()>> d_libraryGetters;  /*!< Library instance getters by name */
    mutable QMutex d_registryMutex;                        /*!< Thread safety for registry access */
};

#endif // HARDWAREREGISTRY_H