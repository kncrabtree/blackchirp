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

/*!
 * \brief Hardware registration information
 */
struct HardwareRegistration {
    QString key;                                               /*!< Hardware type key (e.g., "ftmwDigitizer") */
    QString subKey;                                            /*!< Implementation key (e.g., "m4i2220x8") */
    QString prettyName;                                        /*!< Display name for UI */
    QString description;                                       /*!< Description of the hardware */
    std::function<HardwareObject*()> factory;                 /*!< Factory function to create hardware instance */
    
    // Constructor
    HardwareRegistration() = default;
    HardwareRegistration(const QString& k, const QString& sk, const QString& name,
                        const QString& desc, std::function<HardwareObject*()> fact)
        : key(k), subKey(sk), prettyName(name), description(desc), factory(fact) {}
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
     * \param prettyName Display name for UI
     * \param description Description of the hardware
     * \param factory Factory function to create hardware instances
     * \return True if registration was successful
     */
    bool registerHardware(const QString& key, const QString& subKey, const QString& prettyName,
                          const QString& description, std::function<HardwareObject*()> factory);
    
    /*!
     * \brief Create an instance of the specified hardware
     * \param key Hardware type key
     * \param subKey Implementation key
     * \return Pointer to created hardware object, or nullptr if factory fails
     */
    HardwareObject* createHardware(const QString& key, const QString& subKey);
    
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
    mutable QMutex d_registryMutex;                        /*!< Thread safety for registry access */
};

#endif // HARDWAREREGISTRY_H