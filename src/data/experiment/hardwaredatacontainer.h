#ifndef HARDWAREDATACONTAINER_H
#define HARDWAREDATACONTAINER_H

#include <QString>
#include <QStringList> 
#include <QHash>
#include <QVariant>
#include <map>

/*!
 * \file hardwaredatacontainer.h
 * \brief Pure data structure for hardware information without dependencies on hardware layer
 * 
 * This header provides a clean data container that can be used by experiment classes
 * without requiring dependencies on hardware layer classes like RuntimeHardwareConfig.
 * The container is populated externally by the application layer using RuntimeHardwareConfig
 * and maintains clean architectural separation.
 */

namespace BC::Data {

/*!
 * \brief Hardware type enumeration for robust type identification
 * 
 * This enum provides a stable way to identify hardware types that doesn't
 * depend on string constants that might change during migration.
 */
enum class HardwareType {
    Unknown = 0,
    IOBoard,
    PulseGenerator, 
    FlowController,
    PressureController,
    TemperatureController,
    FtmwScope,
    Clock,
    AWG,
    GPIBController,
    LifScope,
    LifLaser
    // Add other types as needed during migration
};

/*!
 * \brief Pure data container for hardware configuration information
 * 
 * This structure holds both hardware selection mappings and hardware type keys needed
 * by data layer classes like Experiment. It serves as an interface between the hardware
 * layer (RuntimeHardwareConfig) and data layer without creating circular dependencies.
 * 
 * Design principles:
 * - Pure data structure with no dependencies on hardware classes
 * - Populated externally by application layer using RuntimeHardwareConfig
 * - Supports both new experiments (with type keys) and loaded experiments (without type keys)
 * - Maintains backward compatibility with old hardware.csv format
 * - Thread-safe for read operations (no internal state modification)
 * 
 * Usage patterns:
 * 1. New experiments: Application layer populates both typeKeys and hardwareMap from RuntimeHardwareConfig
 * 2. Loaded experiments: Only hardwareMap populated from storage, typeKeys remain empty
 * 3. Type-safe access: Use provided convenience methods with populated type keys
 * 4. Legacy access: Direct hardwareMap access for backward compatibility
 */
struct HardwareDataContainer {
    
    // ========================================================================
    // HARDWARE TYPE KEYS (Populated for new experiments only)
    // ========================================================================
    
    /*!
     * \brief Hardware type key names for type-safe access
     * 
     * These keys are populated by RuntimeHardwareConfig for new experiments to enable
     * type-safe hardware access without hardcoded strings. For loaded experiments,
     * these remain empty and hardwareMap is accessed directly.
     */
    struct TypeKeys {
        QString ftmwScope;              /*!< FTMW digitizer type key (e.g., "FtmwScope") */
        QString clock;                  /*!< Clock type key (e.g., "Clock") */
        QString awg;                    /*!< AWG type key (e.g., "AWG") */
        QString pulseGenerator;         /*!< Pulse generator type key (e.g., "PulseGenerator") */
        QString flowController;         /*!< Flow controller type key (e.g., "FlowController") */
        QString ioBoard;                /*!< IO board type key (e.g., "IOBoard") */
        QString gpibController;         /*!< GPIB controller type key (e.g., "GPIBController") */
        QString pressureController;     /*!< Pressure controller type key (e.g., "PressureController") */
        QString temperatureController;  /*!< Temperature controller type key (e.g., "TemperatureController") */
        QString lifScope;               /*!< LIF digitizer type key (e.g., "LifScope") */
        QString lifLaser;               /*!< LIF laser type key (e.g., "LifLaser") */
        
        /*!
         * \brief Check if type keys are populated
         * \return True if at least some type keys are set (indicates new experiment)
         */
        bool isPopulated() const {
            return !ftmwScope.isEmpty() || !clock.isEmpty() || !awg.isEmpty() ||
                   !pulseGenerator.isEmpty() || !flowController.isEmpty() || !ioBoard.isEmpty() ||
                   !gpibController.isEmpty() || !pressureController.isEmpty() || !temperatureController.isEmpty() ||
                   !lifScope.isEmpty() || !lifLaser.isEmpty();
        }
        
        /*!
         * \brief Get all non-empty type keys as a list
         * \return List of populated hardware type keys
         */
        QStringList getAllTypeKeys() const {
            QStringList keys;
            if (!ftmwScope.isEmpty()) keys << ftmwScope;
            if (!clock.isEmpty()) keys << clock;
            if (!awg.isEmpty()) keys << awg;
            if (!pulseGenerator.isEmpty()) keys << pulseGenerator;
            if (!flowController.isEmpty()) keys << flowController;
            if (!ioBoard.isEmpty()) keys << ioBoard;
            if (!gpibController.isEmpty()) keys << gpibController;
            if (!pressureController.isEmpty()) keys << pressureController;
            if (!temperatureController.isEmpty()) keys << temperatureController;
            return keys;
        }
    } typeKeys;
    
    // ========================================================================
    // HARDWARE SELECTIONS MAP (Always populated)
    // ========================================================================
    
    /*!
     * \brief Individual hardware selection entry
     */
    struct HardwareEntry {
        QString implementation;     /*!< Implementation key (e.g., "mks647c", "virtual") */
        HardwareType type;         /*!< Hardware type enum for robust identification */
        
        HardwareEntry() : type(HardwareType::Unknown) {}
        HardwareEntry(const QString& impl, HardwareType hwType) 
            : implementation(impl), type(hwType) {}
    };
    
    /*!
     * \brief Hardware selections mapping
     * 
     * Format: "type.label" -> HardwareEntry{implementation, type_enum}
     * Examples:
     * - "FlowController.frontPanel" -> {"mks647c", HardwareType::FlowController}
     * - "FtmwScope.main" -> {"virtualftmwscope", HardwareType::FtmwScope}
     * - "Clock.reference" -> {"fixedclock", HardwareType::Clock}
     */
    QHash<QString, HardwareEntry> hardwareMap;
    
    // ========================================================================
    // HARDWARE TYPE ENUM MAPPING
    // ========================================================================
    
    /*!
     * \brief Map legacy hardware type string to enum value (LEGACY COMPATIBILITY ONLY)
     * \param legacyTypeString Legacy hardware type string from old experiments (e.g., "FlowController")
     * \return Hardware type enum, or HardwareType::Unknown if not recognized
     * 
     * NOTE: This is ONLY for loading legacy experiments. New code should use QMetaObject classnames.
     */
    static HardwareType legacyStringToHardwareType(const QString& legacyTypeString) {
        static const QHash<QString, HardwareType> legacyTypeMap = {
            {"IOBoard", HardwareType::IOBoard},
            {"PulseGenerator", HardwareType::PulseGenerator},
            {"FlowController", HardwareType::FlowController},
            {"PressureController", HardwareType::PressureController},
            {"TemperatureController", HardwareType::TemperatureController},
            {"FtmwScope", HardwareType::FtmwScope},
            {"FtmwDigitizer", HardwareType::FtmwScope},  // pre-label-era name
            {"Clock", HardwareType::Clock},
            {"AWG", HardwareType::AWG},
            {"GPIBController", HardwareType::GPIBController},
            {"GpibController", HardwareType::GPIBController},  // current className
            {"LifScope", HardwareType::LifScope},
            {"LifLaser", HardwareType::LifLaser}
        };
        return legacyTypeMap.value(legacyTypeString, HardwareType::Unknown);
    }
    
    /*!
     * \brief Extract hardware type from hardware key (works with both legacy and new formats)
     * \param key Hardware key (e.g., "FlowController.frontPanel" or "FlowController.0")
     * \return Hardware type enum
     */
    static HardwareType extractHardwareType(const QString& key) {
        auto parts = key.split('.');
        if (parts.isEmpty()) return HardwareType::Unknown;
        return legacyStringToHardwareType(parts.first());
    }
    
    // ========================================================================
    // CONVENIENCE METHODS (Type-safe access using populated type keys)
    // ========================================================================
    
    /*!
     * \brief Get hardware selections for a specific type (type-safe)
     * \param typeKey Hardware type key (from populated typeKeys)
     * \return Map of label -> implementation for the specified type
     * 
     * Example:
     * auto flowSelections = container.getSelectionsForType(container.typeKeys.flowController);
     * // Returns: {"frontPanel" -> "mks647c", "backup" -> "virtual"}
     */
    QHash<QString, QString> getSelectionsForType(const QString& typeKey) const {
        QHash<QString, QString> result;
        if (typeKey.isEmpty()) return result;
        
        QString prefix = typeKey + ".";
        for (auto it = hardwareMap.begin(); it != hardwareMap.end(); ++it) {
            if (it.key().startsWith(prefix)) {
                QString label = it.key().mid(prefix.length());
                result[label] = it.value().implementation;
            }
        }
        return result;
    }
    
    /*!
     * \brief Get hardware selections for a specific hardware type enum
     * \param hwType Hardware type enum
     * \return Map of full_key -> implementation for the specified type
     * 
     * Example:
     * auto flowSelections = container.getSelectionsForType(HardwareType::FlowController);
     * // Returns: {"FlowController.frontPanel" -> "mks647c", "FlowController.backup" -> "virtual"}
     */
    QHash<QString, QString> getSelectionsForType(HardwareType hwType) const {
        QHash<QString, QString> result;
        for (auto it = hardwareMap.begin(); it != hardwareMap.end(); ++it) {
            if (it.value().type == hwType) {
                result[it.key()] = it.value().implementation;
            }
        }
        return result;
    }
    
    /*!
     * \brief Get FTMW scope selections (convenience method)
     * \return Map of label -> implementation for FTMW scopes
     */
    QHash<QString, QString> getFtmwScopeSelections() const {
        return getSelectionsForType(typeKeys.ftmwScope);
    }
    
    /*!
     * \brief Get clock selections (convenience method)
     * \return Map of label -> implementation for clocks
     */
    QHash<QString, QString> getClockSelections() const {
        return getSelectionsForType(typeKeys.clock);
    }
    
    /*!
     * \brief Get AWG selections (convenience method)
     * \return Map of label -> implementation for AWGs
     */
    QHash<QString, QString> getAwgSelections() const {
        return getSelectionsForType(typeKeys.awg);
    }
    
    /*!
     * \brief Get pulse generator selections (convenience method)
     * \return Map of label -> implementation for pulse generators
     */
    QHash<QString, QString> getPulseGeneratorSelections() const {
        return getSelectionsForType(typeKeys.pulseGenerator);
    }
    
    /*!
     * \brief Get flow controller selections (convenience method)
     * \return Map of label -> implementation for flow controllers
     */
    QHash<QString, QString> getFlowControllerSelections() const {
        return getSelectionsForType(typeKeys.flowController);
    }
    
    /*!
     * \brief Get IO board selections (convenience method)
     * \return Map of label -> implementation for IO boards
     */
    QHash<QString, QString> getIoBoardSelections() const {
        return getSelectionsForType(typeKeys.ioBoard);
    }
    
    /*!
     * \brief Get GPIB controller selections (convenience method)
     * \return Map of label -> implementation for GPIB controllers
     */
    QHash<QString, QString> getGpibControllerSelections() const {
        return getSelectionsForType(typeKeys.gpibController);
    }
    
    /*!
     * \brief Get pressure controller selections (convenience method)
     * \return Map of label -> implementation for pressure controllers
     */
    QHash<QString, QString> getPressureControllerSelections() const {
        return getSelectionsForType(typeKeys.pressureController);
    }
    
    /*!
     * \brief Get temperature controller selections (convenience method)
     * \return Map of label -> implementation for temperature controllers
     */
    QHash<QString, QString> getTemperatureControllerSelections() const {
        return getSelectionsForType(typeKeys.temperatureController);
    }
    
    // ========================================================================
    // UTILITY METHODS
    // ========================================================================
    
    /*!
     * \brief Check if any hardware is configured
     * \return True if hardwareMap contains any entries
     */
    bool hasAnyHardware() const {
        return !hardwareMap.isEmpty();
    }
    
    /*!
     * \brief Get all configured hardware types (from map keys)
     * \return List of hardware type keys that have selections
     */
    QStringList getConfiguredTypes() const {
        QStringList types;
        for (auto it = hardwareMap.begin(); it != hardwareMap.end(); ++it) {
            QString type = it.key().split('.').first();
            if (!types.contains(type)) {
                types << type;
            }
        }
        return types;
    }
    
    /*!
     * \brief Get hardware selection for specific type and label
     * \param type Hardware type key
     * \param label Hardware label
     * \return Implementation key, or empty string if not found
     */
    QString getImplementation(const QString& type, const QString& label) const {
        QString key = type + "." + label;
        return hardwareMap.value(key, HardwareEntry()).implementation;
    }
    
    /*!
     * \brief Convert to legacy format for backward compatibility
     * \return Map compatible with old std::map<QString,QString,std::less<>> format
     */
    std::map<QString, QString, std::less<>> toLegacyMap() const {
        std::map<QString, QString, std::less<>> legacy;
        for (auto it = hardwareMap.begin(); it != hardwareMap.end(); ++it) {
            legacy[it.key()] = it.value().implementation;
        }
        return legacy;
    }
    
    /*!
     * \brief Create from legacy format for backward compatibility
     * \param legacyMap Hardware map in old std::map format
     * \return HardwareDataContainer with populated hardwareMap (no type keys)
     */
    static HardwareDataContainer fromLegacyMap(const std::map<QString, QString, std::less<>>& legacyMap) {
        HardwareDataContainer container;
        for (const auto& [key, impl] : legacyMap) {
            HardwareType hwType = extractHardwareType(key);
            container.hardwareMap[key] = HardwareEntry(impl, hwType);
        }
        // typeKeys remain empty for legacy data
        return container;
    }
    
    /*!
     * \brief Merge hardware selections from another container
     * \param other Container to merge from
     * \param overwrite Whether to overwrite existing entries
     */
    void mergeFrom(const HardwareDataContainer& other, bool overwrite = false) {
        for (auto it = other.hardwareMap.begin(); it != other.hardwareMap.end(); ++it) {
            if (overwrite || !hardwareMap.contains(it.key())) {
                hardwareMap[it.key()] = it.value(); // it.value() is already a HardwareEntry
            }
        }
        
        // Merge type keys if target doesn't have them populated
        if (!typeKeys.isPopulated() && other.typeKeys.isPopulated()) {
            typeKeys = other.typeKeys;
        }
    }
    
    // ========================================================================
    // SERIALIZATION SUPPORT (for CSV format with backward compatibility)
    // ========================================================================
    
    /*!
     * \brief Save hardware configuration to hardware.csv file
     * \param filePath Full path to hardware.csv file to write
     * \return True if save was successful, false otherwise
     */
    bool saveToFile(const QString& filePath) const;
    
    /*!
     * \brief Load hardware configuration from hardware.csv file with backward compatibility
     * \param filePath Full path to hardware.csv file to read
     * \return HardwareDataContainer with loaded data, or empty container if load failed
     *
     * Three on-disk formats are accepted so historical experiments load
     * unchanged:
     * - 1 column: a single hardware-type root key (predates multiple-
     *   hardware support; the loader synthesises a \c "<type>.default" key
     *   with implementation \c "virtual").
     * - 2 columns: \c key and the driver class identifier. Header row's
     *   second label may be either the historical \c "subKey" or the
     *   current \c "driver"; the reader is positional so either works.
     * - 3 columns: \c key, driver, and a redundant numeric hardware-type
     *   cell from a transitional format. The third cell is silently
     *   ignored — the \c HardwareType is recovered from the key prefix
     *   instead.
     *
     * The original key format is preserved from the file for display and
     * comparison purposes. Legacy keys will naturally fail hardware
     * validation against current label-based configurations, which is the
     * desired behavior.
     */
    static HardwareDataContainer loadFromFile(const QString& filePath);

    /*!
     * \brief Check if hardware key uses legacy index format
     * \param key Hardware key to check (e.g., "FlowController.0" vs "FlowController.frontPanel")
     * \return True if key appears to be in legacy index format
     * 
     * This is useful for display purposes to show users when they're viewing
     * legacy experiment data.
     */
    static bool isLegacyKey(const QString& key) {
        auto parts = key.split('.');
        if (parts.size() != 2) return false;
        
        bool ok;
        parts.last().toInt(&ok);
        return ok; // If second part converts to int, it's likely legacy format
    }
};

} // namespace BC::Data

#endif // HARDWAREDATACONTAINER_H