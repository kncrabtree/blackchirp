#include "runtimehardwareconfig.h"
#include "hardwareregistry.h"
#include "hardwareprofilemanager.h"

// Hardware class includes for template type resolution
#include "hw_base.h"
#include <data/storage/applicationconfigmanager.h>

#include <QReadLocker>
#include <QWriteLocker>
#include <QDebug>

// Static member definitions
RuntimeHardwareConfig* RuntimeHardwareConfig::s_instance = nullptr;

RuntimeHardwareConfig::RuntimeHardwareConfig()
    : SettingsStorage(BC::Key::RuntimeHw::runtimeHw)
{
    qDebug() << "Initializing RuntimeHardwareConfig...";
    syncWithProfiles();
}

const RuntimeHardwareConfig& RuntimeHardwareConfig::constInstance()
{
    if (!s_instance) {
        s_instance = new RuntimeHardwareConfig();
    }
    return *s_instance;
}

RuntimeHardwareConfig& RuntimeHardwareConfig::instance()
{
    if (!s_instance) {
        s_instance = new RuntimeHardwareConfig();
    }
    return *s_instance;
}

// ============================================================================
// READ-ONLY OPERATIONS
// ============================================================================

QString RuntimeHardwareConfig::getHardwareImplementation(const QString& hardwareType, const QString& label) const
{
    QReadLocker locker(&d_configLock);
    
    QString key = BC::Key::hwKey(hardwareType, label);
    auto it = d_activeHardware.find(key);
    
    if (it != d_activeHardware.end()) {
        return it->implementation;
    }
    
    return QString(); // Not found
}

QStringList RuntimeHardwareConfig::getActiveLabels(const QString& hardwareType) const
{
    QReadLocker locker(&d_configLock);
    
    QStringList activeLabels;
    for (auto it = d_activeHardware.cbegin(); it != d_activeHardware.cend(); ++it) {
        if (it->type == hardwareType) {
            auto [parsedType, label] = BC::Key::parseKey(it.key());
            activeLabels.append(label);
        }
    }
    
    return activeLabels;
}

QStringList RuntimeHardwareConfig::getActiveKeys(const QString& hardwareType) const
{
    QReadLocker locker(&d_configLock);
    
    QStringList activeKeys;
    for (auto it = d_activeHardware.cbegin(); it != d_activeHardware.cend(); ++it) {
        if (it->type == hardwareType) {
            activeKeys.append(it.key());
        }
    }
    
    return activeKeys;
}

std::map<QString, QString> RuntimeHardwareConfig::getCurrentHardware() const
{
    QReadLocker locker(&d_configLock);
    
    std::map<QString, QString> hardware;
    
    for (auto it = d_activeHardware.cbegin(); it != d_activeHardware.cend(); ++it) {
        const QString& hwKey = it.key(); // Already in "type.label" format
        const HardwareSelection& selection = it.value();
        
        if (!selection.implementation.isEmpty()) {
            hardware[hwKey] = selection.implementation;
        }
    }
    
    return hardware;
}

BC::Data::HardwareDataContainer RuntimeHardwareConfig::createHardwareDataContainer() const
{
    QReadLocker locker(&d_configLock);
    
    BC::Data::HardwareDataContainer container;
    
    // Populate hardware map with all active hardware selections
    for (auto it = d_activeHardware.cbegin(); it != d_activeHardware.cend(); ++it) {
        const QString& hwKey = it.key(); // Already in "type.label" format
        const HardwareSelection& selection = it.value();
        
        if (!selection.implementation.isEmpty()) {
            // Extract hardware type from the selection.type field or from the key
            auto keyParts = hwKey.split('.');
            QString typeString = keyParts.isEmpty() ? hwKey : keyParts.first();
            BC::Data::HardwareType hwType = BC::Data::HardwareDataContainer::legacyStringToHardwareType(typeString);
            
            container.hardwareMap[hwKey] = BC::Data::HardwareDataContainer::HardwareEntry(selection.implementation, hwType);
        }
    }
    
    // Populate type keys using hardwareTypeOf template method for type safety
    // All hardware types are always compiled in with runtime hardware configuration
    container.typeKeys.ftmwScope = hardwareTypeOf<FtmwScope>();
    container.typeKeys.clock = hardwareTypeOf<Clock>();
    container.typeKeys.awg = hardwareTypeOf<AWG>();
    container.typeKeys.pulseGenerator = hardwareTypeOf<PulseGenerator>();
    container.typeKeys.flowController = hardwareTypeOf<FlowController>();
    container.typeKeys.ioBoard = hardwareTypeOf<IOBoard>();
    container.typeKeys.gpibController = hardwareTypeOf<GpibController>();
    container.typeKeys.pressureController = hardwareTypeOf<PressureController>();
    container.typeKeys.temperatureController = hardwareTypeOf<TemperatureController>();
    container.typeKeys.lifScope = hardwareTypeOf<LifScope>();
    container.typeKeys.lifLaser = hardwareTypeOf<LifLaser>();
    
    return container;
}

QStringList RuntimeHardwareConfig::validateHardwareConfiguration(const std::map<QString, QString>& hardwareMap)
{
    QStringList errors;
    
    // Check each hardware selection in the map
    for (auto it = hardwareMap.begin(); it != hardwareMap.end(); ++it) {
        const QString& hwKey = it->first;  // "type.label" format
        const QString& implementation = it->second;
        
        // Parse hardware type and label from key
        auto [hardwareType, label] = BC::Key::parseKey(hwKey);
        
        if (hardwareType.isEmpty() || label.isEmpty()) {
            errors << QString("Invalid hardware key format: '%1'").arg(hwKey);
            continue;
        }
        
        if (implementation.isEmpty()) {
            errors << QString("No implementation selected for hardware '%1.%2'")
                         .arg(hardwareType, label);
            continue;
        }
        
        // Check if implementation is registered
        HardwareRegistry& registry = HardwareRegistry::instance();
        if (!registry.isRegistered(hardwareType, implementation)) {
            errors << QString("Implementation '%1' for hardware '%2.%3' is not registered")
                         .arg(implementation, hardwareType, label);
            continue;
        }
    }
    
    // Check for missing required hardware types using the static isHardwareRequired method
    HardwareRegistry& registry = HardwareRegistry::instance();
    QStringList allTypes = registry.getHardwareTypes();
    
    for (const QString& hwType : allTypes) {
        if (!isHardwareRequired(hwType)) {
            continue; // Not a required type
        }
        
        // Check if we have any valid hardware of this required type
        bool hasValidHardware = false;
        for (auto it = hardwareMap.begin(); it != hardwareMap.end(); ++it) {
            auto [configHwType, label] = BC::Key::parseKey(it->first);
            if (configHwType == hwType && !it->second.isEmpty()) {
                // Check if implementation is valid
                if (registry.isRegistered(configHwType, it->second)) {
                    hasValidHardware = true;
                    break;
                }
            }
        }
        
        if (!hasValidHardware) {
            errors << QString("Required hardware type '%1' is not configured or has no valid implementation")
                         .arg(hwType);
        }
    }
    
    // Check for duplicate labels within the same hardware type
    QHash<QString, QStringList> labelsByType;
    for (auto it = hardwareMap.begin(); it != hardwareMap.end(); ++it) {
        auto [hwType, label] = BC::Key::parseKey(it->first);
        if (!hwType.isEmpty() && !label.isEmpty()) {
            labelsByType[hwType].append(label);
        }
    }
    
    for (auto typeIt = labelsByType.begin(); typeIt != labelsByType.end(); ++typeIt) {
        const QString& hwType = typeIt.key();
        QStringList& labels = typeIt.value();
        
        // Check for duplicate labels
        QSet<QString> seenLabels;
        for (const QString& label : labels) {
            if (seenLabels.contains(label)) {
                errors << QString("Duplicate label '%1' for hardware type '%2'")
                             .arg(label, hwType);
            } else {
                seenLabels.insert(label);
            }
        }
    }
    
    return errors;
}

QHash<QString, HardwareValidationResult> RuntimeHardwareConfig::validateConfiguration() const
{
    QReadLocker locker(&d_configLock);
    
    QHash<QString, HardwareValidationResult> results;
    
    for (auto it = d_activeHardware.cbegin(); it != d_activeHardware.cend(); ++it) {
        const QString& hwKey = it.key(); // "type.label" format
        const HardwareSelection& selection = it.value();
        auto [type, label] = BC::Key::parseKey(hwKey);
        
        results[hwKey] = validateHardwareSelectionInternal(type, label, selection);
    }
    
    return results;
}


bool RuntimeHardwareConfig::isHardwareConfigurationValid(const std::map<QString, QString>& hardwareMap)
{
    QStringList errors = validateHardwareConfiguration(hardwareMap);
    return errors.isEmpty();
}

bool RuntimeHardwareConfig::isConfigurationValid() const
{
    QReadLocker locker(&d_configLock);
    
    // Convert internal hardware configuration to map format
    std::map<QString, QString> currentConfig;
    for (auto it = d_activeHardware.cbegin(); it != d_activeHardware.cend(); ++it) {
        const QString& hwKey = it.key();
        const HardwareSelection& selection = it.value();
        if (!selection.implementation.isEmpty()) {
            currentConfig[hwKey] = selection.implementation;
        }
    }
    
    // Use the static validation method
    return isHardwareConfigurationValid(currentConfig);
}

QStringList RuntimeHardwareConfig::getConfiguredHardwareTypes() const
{
    QReadLocker locker(&d_configLock);
    
    QSet<QString> types;
    for (auto it = d_activeHardware.cbegin(); it != d_activeHardware.cend(); ++it) {
        types.insert(it->type);
    }
    
    return QStringList(types.begin(), types.end());
}

bool RuntimeHardwareConfig::isHardwareRequired(const QString& hardwareType)
{
    // Define required hardware types using type-safe hardwareTypeOf<T>()
    // These are hardware types that must be configured for the system to operate
    static const QStringList coreRequiredTypes = {
        hardwareTypeOf<FtmwScope>(),  // Required for FTMW spectroscopy
        hardwareTypeOf<Clock>()       // Required for timing
    };
    
    // Check core required types
    if (coreRequiredTypes.contains(hardwareType)) {
        return true;
    }
    
    // Check LIF-specific required types if LIF is enabled
    if (ApplicationConfigManager::instance().isLifEnabled()) {
        static const QStringList lifRequiredTypes = {
            hardwareTypeOf<LifScope>(),   // Required for LIF data acquisition
            hardwareTypeOf<LifLaser>()    // Required for LIF laser control
        };
        
        if (lifRequiredTypes.contains(hardwareType)) {
            return true;
        }
    }
    
    return false;
}

QStringList RuntimeHardwareConfig::getMissingRequiredHardware() const
{
    QReadLocker locker(&d_configLock);
    return getMissingRequiredHardwareInternal();
}

QStringList RuntimeHardwareConfig::getAllValidationErrors() const
{
    QReadLocker locker(&d_configLock);
    
    QStringList allErrors;
    auto results = validateConfiguration();
    
    for (auto it = results.cbegin(); it != results.cend(); ++it) {
        allErrors.append(it.value().errors);
    }
    
    return allErrors;
}

QStringList RuntimeHardwareConfig::getAllValidationWarnings() const
{
    QReadLocker locker(&d_configLock);
    
    QStringList allWarnings;
    auto results = validateConfiguration();
    
    for (auto it = results.cbegin(); it != results.cend(); ++it) {
        allWarnings.append(it.value().warnings);
    }
    
    return allWarnings;
}

// ============================================================================
// WRITE OPERATIONS (Friend class access only)
// ============================================================================

bool RuntimeHardwareConfig::setHardwareSelection(const QString& hardwareType, 
                                                 const QString& label,
                                                 const QString& implementation)
{
    QWriteLocker locker(&d_configLock);
    
    if (hardwareType.isEmpty() || label.isEmpty()) {
        qWarning() << "Cannot set hardware selection: hardware type or label is empty";
        return false;
    }
    
    // Validate that the implementation exists in the registry if not empty
    if (!implementation.isEmpty()) {
        HardwareRegistry& registry = HardwareRegistry::instance();
        const HardwareRegistration* reg = registry.getRegistration(hardwareType, implementation);
        if (!reg) {
            qWarning() << "Cannot set hardware selection: implementation" << implementation 
                       << "not registered for type" << hardwareType;
            return false;
        }
    }
    
    // Update in-memory configuration
    QString key = BC::Key::hwKey(hardwareType, label);
    HardwareSelection selection;
    selection.type = hardwareType;
    selection.implementation = implementation;
    
    d_activeHardware[key] = selection;
    
    qDebug() << "Set hardware selection:" << key << "=" << implementation;
    
    // Update the profile manager
    activateProfile(hardwareType, label);
    
    return true;
}


bool RuntimeHardwareConfig::removeHardwareSelection(const QString& hardwareType, const QString& label)
{
    QWriteLocker locker(&d_configLock);
    
    if (hardwareType.isEmpty() || label.isEmpty()) {
        return false;
    }
    
    QString key = BC::Key::hwKey(hardwareType, label);
    bool removed = d_activeHardware.remove(key) > 0;
    
    if (removed) {
        qDebug() << "Removed hardware selection for:" << key;
        
        // Update the profile manager
        deactivateProfile(hardwareType, label);
    }
    
    return removed;
}

void RuntimeHardwareConfig::clearConfiguration()
{
    QWriteLocker locker(&d_configLock);
    
    qDebug() << "Clearing all hardware configuration...";
    d_activeHardware.clear();
}

void RuntimeHardwareConfig::registerHardwareForTesting(const QString& hardwareType, const QString& implementation, int mapIndex)
{
    QWriteLocker locker(&d_configLock);
    
    // Create stable test label based on map position (zero-padded for sorting)
    QString testLabel = QString("test%1").arg(mapIndex, 2, 10, QChar('0'));
    QString key = BC::Key::hwKey(hardwareType, testLabel);
    
    HardwareSelection selection;
    selection.type = hardwareType;
    selection.implementation = implementation;
    
    d_activeHardware[key] = selection;
    
    qDebug() << "TEMPORARY: Registered test hardware:" << key << "=" << implementation;
    
    // Create corresponding profile in HardwareProfileManager for consistency
    HardwareProfileManager& profileManager = HardwareProfileManager::instance();
    profileManager.createHardwareProfile(hardwareType, implementation, testLabel);
    profileManager.activateHardwareProfile(hardwareType, testLabel);
}



// ============================================================================
// INTEGRATION WITH HARDWAREPROFILEMANAGER
// ============================================================================

void RuntimeHardwareConfig::syncWithProfiles()
{
    qDebug() << "Syncing RuntimeHardwareConfig with active hardware profiles...";
    
    QWriteLocker locker(&d_configLock);
    d_activeHardware.clear();
    
    // Get HardwareProfileManager singleton instance and load active profiles
    HardwareProfileManager& profileManager = HardwareProfileManager::instance();
    QStringList hardwareTypes = profileManager.keys(); // Get all hardware types with profiles
    
    int loadedCount = 0;
    for (const QString& type : hardwareTypes) {
        QStringList activeLabels = profileManager.getActiveProfiles(type);
        
        for (const QString& label : activeLabels) {
            QString implementation = profileManager.getImplementation(type, label);
            if (!implementation.isEmpty()) {
                QString key = BC::Key::hwKey(type, label);
                
                HardwareSelection selection;
                selection.type = type;
                selection.implementation = implementation;
                
                d_activeHardware[key] = selection;
                loadedCount++;
                
                qDebug() << "Loaded active profile:" << key << "=" << implementation;
            }
        }
    }
    
    qDebug() << "Loaded" << loadedCount << "active hardware profiles into runtime config";
}

void RuntimeHardwareConfig::activateProfile(const QString& hardwareType, const QString& label)
{
    // Activate profile in HardwareProfileManager
    // Note: This doesn't hold the config lock since it's called from methods that already hold it
    
    HardwareProfileManager& profileManager = HardwareProfileManager::instance();
    profileManager.activateHardwareProfile(hardwareType, label);
}

void RuntimeHardwareConfig::deactivateProfile(const QString& hardwareType, const QString& label)
{
    // Deactivate profile in HardwareProfileManager
    // Note: This doesn't hold the config lock since it's called from methods that already hold it
    
    HardwareProfileManager& profileManager = HardwareProfileManager::instance();
    profileManager.deactivateHardwareProfile(hardwareType, label);
}

// ============================================================================
// PRIVATE HELPERS
// ============================================================================

HardwareValidationResult RuntimeHardwareConfig::validateHardwareSelectionInternal(const QString& hardwareType, const QString& label, const HardwareSelection& selection) const
{
    // Note: This method assumes the caller already holds the appropriate lock
    
    HardwareValidationResult result;
    
    if (selection.implementation.isEmpty()) {
        result.isValid = false;
        result.errors << QString("No implementation selected for hardware '%1.%2'").arg(hardwareType, label);
        return result;
    }
    
    // Check if implementation is registered (simple validation only)
    HardwareRegistry& registry = HardwareRegistry::instance();
    if (!registry.isRegistered(hardwareType, selection.implementation)) {
        result.isValid = false;
        result.errors << QString("Selected implementation '%1' for hardware '%2.%3' is not registered")
                         .arg(selection.implementation, hardwareType, label);
        return result;
    }
    
    // NOTE: Actual instantiation and runtime validation is handled by HardwareManager
    // This validation only checks configuration completeness and registration
    
    // Validation passed
    result.isValid = true;
    result.selectedImplementation = selection.implementation;
    
    // Check if this is required hardware
    if (isHardwareRequired(hardwareType)) {
        result.warnings << QString("Required hardware '%1.%2' is configured and available").arg(hardwareType, label);
    }
    
    return result;
}

QStringList RuntimeHardwareConfig::getMissingRequiredHardwareInternal() const
{
    // Note: This method assumes the caller already holds the appropriate lock
    
    QStringList missing;
    HardwareRegistry& registry = HardwareRegistry::instance();
    QStringList allTypes = registry.getHardwareTypes();
    
    for (const QString& type : allTypes) {
        if (isHardwareRequired(type)) {
            // Check if we have any active hardware of this type
            bool hasValidHardware = false;
            for (auto it = d_activeHardware.cbegin(); it != d_activeHardware.cend(); ++it) {
                if (it->type == type) {
                    auto [hwType, label] = BC::Key::parseKey(it.key());
                    HardwareValidationResult result = validateHardwareSelectionInternal(hwType, label, it.value());
                    if (result.isValid) {
                        hasValidHardware = true;
                        break;
                    }
                }
            }
            
            if (!hasValidHardware) {
                missing.append(type);
            }
        }
    }
    
    return missing;
}

bool RuntimeHardwareConfig::applyConfiguration(const std::map<QString, QString>& config)
{
    QWriteLocker locker(&d_configLock);
    
    // Clear current active hardware
    d_activeHardware.clear();
    
    // Apply new configuration
    for (auto it = config.begin(); it != config.end(); ++it) {
        const QString& key = it->first;
        const QString& implementation = it->second;
        
        // Parse hardware type and label from key
        auto [hardwareType, label] = BC::Key::parseKey(key);
        
        if (hardwareType.isEmpty() || label.isEmpty() || implementation.isEmpty()) {
            qWarning() << "RuntimeHardwareConfig::applyConfiguration: Invalid configuration entry:"
                      << "key=" << key << "implementation=" << implementation;
            continue;
        }
        
        // Set hardware selection using internal method
        HardwareSelection selection;
        selection.type = hardwareType;
        selection.implementation = implementation;
        
        d_activeHardware[key] = selection;
    }
    
    return true;
}
