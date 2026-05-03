#ifndef HARDWAREREGISTRY_H
#define HARDWAREREGISTRY_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QHash>
#include <QMap>
#include <QMutex>
#include <functional>

#include <data/storage/settingsstorage.h>
#include <hardware/core/communication/communicationprotocol.h>

class HardwareObject;
class VendorLibrary;

/*!
 * \brief Describes a configuration parameter that must be set before hardware construction
 *
 * Used by Python hardware trampolines (and potentially other dynamic implementations)
 * to declare parameters that need UI input when a profile is created. The QVariant
 * defaultValue determines both the default and the widget type (int → QSpinBox,
 * bool → QCheckBox, double → QDoubleSpinBox, QString → QLineEdit).
 */
struct HwConfigParam {
    QString key;           /*!< SettingsStorage key (written before object construction) */
    QString label;         /*!< Display label for the UI widget */
    QVariant defaultValue; /*!< Type-aware default value */
    QVariant minimum;      /*!< Optional minimum for numeric types (invalid QVariant = no limit) */
    QVariant maximum;      /*!< Optional maximum for numeric types (invalid QVariant = no limit) */
};

/*!
 * \brief Priority level for hardware settings
 *
 * Controls visibility in the profile creation dialog and HWDialog:
 * - Required: Must be set before construction. Shown prominently.
 *   May not be editable after profile creation.
 * - Important: Has a sensible default but the user should review it.
 *   Shown in the main settings area.
 * - Optional: Rarely needs changing. Shown under a collapsible
 *   "Advanced Settings" section.
 */
enum class HwSettingPriority {
    Required,  ///< Must be set before construction; may not be edited after profile creation
    Important, ///< Has a sensible default but the user should review it
    Optional   ///< Rarely needs changing; rendered under a collapsible *Advanced Settings* group
};

/*!
 * \brief Input type for a custom communication parameter
 */
enum class CustomCommType { String, Int, FilePath };

/*!
 * \brief Descriptor for one user-visible field in a Custom-protocol device
 *
 * Registered statically at program startup via REGISTER_CUSTOM_COMM.
 * For String fields, \c bound holds the maximum length (int).
 * For Int fields, \c bound holds the minimum value (int) and \c bound2 holds the
 * maximum value (int).
 * For FilePath fields, the bound fields are unused.
 */
struct CustomCommDef {
    QString key;              ///< SettingsStorage key (written to BC::Key::Comm::custom group)
    QString label;            ///< User-facing display label
    QString description;      ///< Explanatory tooltip/help text
    CustomCommType type;      ///< Widget type to render
    QVariant bound;           ///< Type-dependent lower bound / max string length
    QVariant bound2;          ///< Type-dependent upper bound (Int only)
};

/*!
 * \brief Scalar setting definition with metadata
 *
 * Registered statically at program startup. The defaultValue's QVariant
 * type determines the UI widget (int -> QSpinBox, double -> QDoubleSpinBox,
 * bool -> QCheckBox, QString -> QLineEdit).
 */
struct HwSettingDef {
    QString key;              ///< SettingsStorage key
    QString label;            ///< User-facing display label
    QString description;      ///< Explanatory tooltip/help text
    QVariant defaultValue;    ///< Type-aware default value
    QVariant minimum;         ///< Optional min for numeric types (invalid = no limit)
    QVariant maximum;         ///< Optional max for numeric types (invalid = no limit)
    HwSettingPriority priority = HwSettingPriority::Optional;
};

/*!
 * \brief Array setting definition with metadata
 *
 * Describes an array-type setting (e.g., sampleRates). The entries vector
 * holds the default array contents; each entry is a SettingsMap.
 */
struct HwArraySettingDef {
    QString key;              ///< SettingsStorage array key
    QString label;            ///< User-facing display label
    QString description;      ///< Explanatory tooltip/help text
    std::vector<SettingsStorage::SettingsMap> entries;  ///< Default entries
    HwSettingPriority priority = HwSettingPriority::Optional;
};

/*!
 * \brief Hardware registration information
 */
struct HardwareRegistration {
    QString key;                                               /*!< Hardware type key (e.g., "ftmwDigitizer") */
    QString subKey;                                            /*!< Implementation key (e.g., "m4i2220x8") */
    QString description;                                       /*!< Description of the hardware */
    std::function<HardwareObject*(const QString&)> factory;   /*!< Factory function to create hardware instance with label */
    QStringList inheritanceChain;                             /*!< Class names from direct base up to (not including) QObject */
    QStringList libraryDependencies;                          /*!< List of vendor libraries this hardware depends on */
    QVector<CommunicationProtocol::CommType> supportedProtocols; /*!< Communication protocols supported by this hardware */
    QVector<HwSettingDef> settingDefs;                         /*!< Registered setting definitions with metadata */
    QMap<QString, HwArraySettingDef> arraySettingDefs;          /*!< Registered array setting definitions */
    QVector<CustomCommDef> customCommDefs;                      /*!< Registered custom communication parameter definitions */

    /*! \brief Construct an empty registration with all fields at their default values */
    HardwareRegistration() = default;
    /*!
     * \brief Construct a registration with the core metadata fields
     * \param k Hardware type key (e.g., "ftmwDigitizer")
     * \param sk Implementation key (e.g., "m4i2220x8")
     * \param desc Human-readable description of the hardware
     * \param fact Factory function that constructs an instance given a label string
     * \param chain Class names from the implementation's direct base up to (not including) QObject
     */
    HardwareRegistration(const QString& k, const QString& sk, const QString& desc,
                        std::function<HardwareObject*(const QString&)> fact,
                        const QStringList& chain = {})
        : key(k), subKey(sk), description(desc), factory(fact), inheritanceChain(chain),
          libraryDependencies({}), supportedProtocols({CommunicationProtocol::Virtual}) {}
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
     * \param inheritanceChain Class names from the implementation's direct base up to (not including) QObject
     * \return True if registration was successful
     */
    bool registerHardware(const QString& key, const QString& subKey, const QString& description,
                          std::function<HardwareObject*(const QString&)> factory,
                          const QStringList& inheritanceChain = {});
    
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
     * \brief Add supported protocols to existing hardware registration
     * \param key Hardware type key
     * \param subKey Implementation key
     * \param protocols List of supported communication protocols
     * \return True if protocols were added successfully
     */
    bool addSupportedProtocols(const QString& key, const QString& subKey,
                               const QVector<CommunicationProtocol::CommType>& protocols);

    /*!
     * \brief Get supported protocols for a hardware implementation
     * \param key Hardware type key
     * \param subKey Implementation key
     * \return List of supported communication protocols, or empty if not registered
     */
    QVector<CommunicationProtocol::CommType> getSupportedProtocols(
        const QString& key, const QString& subKey) const;

    /*!
     * \brief Add setting definitions to an existing hardware registration
     * \param key Hardware type key
     * \param subKey Implementation key
     * \param settings List of setting definitions with metadata
     * \return True if settings were added successfully
     */
    bool addSettingDefs(const QString& key, const QString& subKey,
                        const QVector<HwSettingDef>& settings);

    /*!
     * \brief Get setting definitions for a hardware implementation
     * \param key Hardware type key
     * \param subKey Implementation key
     * \return List of setting definitions, or empty if none registered
     */
    QVector<HwSettingDef> getSettingDefs(const QString& key, const QString& subKey) const;

    /*!
     * \brief Add array setting metadata to an existing hardware registration
     * \param key Hardware type key
     * \param subKey Implementation key
     * \param arrayKey The array setting key
     * \param label User-facing display label
     * \param description Explanatory tooltip/help text
     * \param priority Visibility priority level
     * \return True if array setting definition was added successfully
     */
    bool addArraySettingDef(const QString& key, const QString& subKey,
                            const QString& arrayKey, const QString& label,
                            const QString& description, HwSettingPriority priority);

    /*!
     * \brief Add one entry to an array setting
     * \param key Hardware type key
     * \param subKey Implementation key
     * \param arrayKey The array setting key (must already be registered via addArraySettingDef)
     * \param entry Map of sub-key/value pairs for this entry
     * \return True if entry was added successfully
     */
    bool addArraySettingEntry(const QString& key, const QString& subKey,
                              const QString& arrayKey,
                              const SettingsStorage::SettingsMap& entry);

    /*!
     * \brief Get all array setting definitions for a hardware implementation
     * \param key Hardware type key
     * \param subKey Implementation key
     * \return Map of array key to array setting definitions, merged with inherited base class definitions
     */
    QMap<QString, HwArraySettingDef> getArraySettingDefs(
        const QString& key, const QString& subKey) const;

    /*!
     * \brief Register scalar setting definitions for a base (non-instantiable) hardware class
     * \param className The class name (e.g., "HardwareObject", "Clock")
     * \param settings List of setting definitions with metadata
     * \return True if successfully stored
     *
     * Unlike addSettingDefs, this does not require a prior registerHardware call.
     * Settings are merged into getSettingDefs results for any implementation whose
     * inheritanceChain contains className.
     */
    bool addBaseSettingDefs(const QString& className, const QVector<HwSettingDef>& settings);

    /*!
     * \brief Register array setting metadata for a base hardware class
     * \param className The class name (e.g., "HardwareObject", "Clock")
     * \param arrayKey The array setting key
     * \param label User-facing display label
     * \param description Explanatory tooltip/help text
     * \param priority Visibility priority level
     * \return True if successfully stored
     */
    bool addBaseArraySettingDef(const QString& className, const QString& arrayKey,
                                const QString& label, const QString& description,
                                HwSettingPriority priority);

    /*!
     * \brief Add one entry to a base class array setting
     * \param className The class name
     * \param arrayKey The array setting key (must already be registered via addBaseArraySettingDef)
     * \param entry Map of sub-key/value pairs for this entry
     * \return True if successfully stored
     */
    bool addBaseArraySettingEntry(const QString& className, const QString& arrayKey,
                                  const SettingsStorage::SettingsMap& entry);

    /*!
     * \brief Add custom communication parameter definitions to an existing hardware registration
     * \param key Hardware type key
     * \param subKey Implementation key
     * \param defs List of custom communication field definitions
     * \return True if definitions were added successfully
     */
    bool addCustomCommDefs(const QString& key, const QString& subKey,
                           const QVector<CustomCommDef>& defs);

    /*!
     * \brief Get custom communication parameter definitions for a hardware implementation
     *
     * Returns the implementation's own definitions merged with any definitions registered
     * for base classes in its inheritanceChain (base-class defs appended after the
     * implementation's own, innermost ancestor first).
     *
     * \param key Hardware type key
     * \param subKey Implementation key
     * \return List of custom communication field definitions, or empty if none registered
     */
    QVector<CustomCommDef> getCustomCommDefs(const QString& key, const QString& subKey) const;

    /*!
     * \brief Register custom communication parameter definitions for a base hardware class
     * \param className The class name (e.g., "HardwareObject", "CustomInstrument")
     * \param defs List of custom communication field definitions
     * \return True if successfully stored
     */
    bool addBaseCustomCommDefs(const QString& className, const QVector<CustomCommDef>& defs);

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
    QHash<QString, HardwareRegistration> d_registrations;              /*!< All registered hardware implementations */
    QHash<QString, QVector<HwSettingDef>> d_baseSettingDefs;           /*!< Base class scalar settings, keyed by class name */
    QHash<QString, QMap<QString, HwArraySettingDef>> d_baseArrayDefs;  /*!< Base class array settings, keyed by class name */
    QHash<QString, QVector<CustomCommDef>> d_baseCustomCommDefs;       /*!< Base class custom comm defs, keyed by class name */
    QHash<QString, std::function<VendorLibrary*()>> d_libraryGetters;  /*!< Library instance getters by name */
    mutable QMutex d_registryMutex;                                    /*!< Thread safety for registry access */
};

#endif // HARDWAREREGISTRY_H