#include "runtimehardwareconfig.h"
#include "hardwareregistry.h"

#include <QReadLocker>
#include <QWriteLocker>
#include <QDebug>

// Static member definitions
RuntimeHardwareConfig* RuntimeHardwareConfig::s_instance = nullptr;

RuntimeHardwareConfig::RuntimeHardwareConfig()
    : SettingsStorage(BC::Key::RuntimeHw::runtimeHw)
{
    qDebug() << "Initializing RuntimeHardwareConfig...";
    loadFromSettings();
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

QString RuntimeHardwareConfig::getHardwareSelection(const QString& hardwareType) const
{
    QReadLocker locker(&d_configLock);
    
    auto it = d_hardwareConfig.find(hardwareType);
    if (it == d_hardwareConfig.end() || !it->enabled) {
        return QString(); // Not configured or disabled
    }
    
    return it->implementation;
}

bool RuntimeHardwareConfig::isHardwareEnabled(const QString& hardwareType) const
{
    QReadLocker locker(&d_configLock);
    
    auto it = d_hardwareConfig.find(hardwareType);
    return it != d_hardwareConfig.end() && it->enabled;
}

std::map<QString, QString> RuntimeHardwareConfig::getCurrentHardware() const
{
    QReadLocker locker(&d_configLock);
    
    std::map<QString, QString> hardware;
    
    for (auto it = d_hardwareConfig.cbegin(); it != d_hardwareConfig.cend(); ++it) {
        if (it->enabled && !it->implementation.isEmpty()) {
            hardware[it.key()] = it->implementation;
        }
    }
    
    return hardware;
}

QHash<QString, HardwareValidationResult> RuntimeHardwareConfig::validateConfiguration() const
{
    QReadLocker locker(&d_configLock);
    
    QHash<QString, HardwareValidationResult> results;
    
    for (auto it = d_hardwareConfig.cbegin(); it != d_hardwareConfig.cend(); ++it) {
        results[it.key()] = validateHardwareTypeInternal(it.key(), it.value());
    }
    
    return results;
}

HardwareValidationResult RuntimeHardwareConfig::validateHardwareType(const QString& hardwareType) const
{
    QReadLocker locker(&d_configLock);
    
    auto it = d_hardwareConfig.find(hardwareType);
    if (it == d_hardwareConfig.end()) {
        HardwareValidationResult result;
        result.isValid = false;
        result.errors << QString("Hardware type '%1' is not configured").arg(hardwareType);
        return result;
    }
    
    return validateHardwareTypeInternal(hardwareType, it.value());
}

bool RuntimeHardwareConfig::isConfigurationValid() const
{
    QReadLocker locker(&d_configLock);
    
    // Check all configured hardware
    for (auto it = d_hardwareConfig.cbegin(); it != d_hardwareConfig.cend(); ++it) {
        if (it->enabled) {
            HardwareValidationResult result = validateHardwareTypeInternal(it.key(), it.value());
            if (!result.isValid) {
                return false;
            }
        }
    }
    
    // Check that all required hardware is configured
    QStringList missing = getMissingRequiredHardwareInternal();
    return missing.isEmpty();
}

QStringList RuntimeHardwareConfig::getConfiguredHardwareTypes() const
{
    QReadLocker locker(&d_configLock);
    
    return d_hardwareConfig.keys();
}

bool RuntimeHardwareConfig::isHardwareRequired(const QString& hardwareType) const
{
    // Define required hardware types at compile time
    // These are hardware types that must be configured for the system to operate
    static const QStringList requiredTypes = {
        "ftmwDigitizer",  // Required for FTMW spectroscopy
        "clock"           // Required for timing
    };
    
    return requiredTypes.contains(hardwareType);
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
                                                 const QString& implementation,
                                                 bool enabled)
{
    QWriteLocker locker(&d_configLock);
    
    if (hardwareType.isEmpty()) {
        qWarning() << "Cannot set hardware selection: hardware type is empty";
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
    HardwareConfig config;
    config.implementation = implementation;
    config.enabled = enabled;
    
    d_hardwareConfig[hardwareType] = config;
    
    qDebug() << "Set hardware selection:" << hardwareType << "=" << implementation 
             << "(enabled:" << enabled << ")";
    
    return true;
}

bool RuntimeHardwareConfig::setHardwareEnabled(const QString& hardwareType, bool enabled)
{
    QWriteLocker locker(&d_configLock);
    
    if (hardwareType.isEmpty()) {
        qWarning() << "Cannot set hardware enabled state: hardware type is empty";
        return false;
    }
    
    auto it = d_hardwareConfig.find(hardwareType);
    if (it != d_hardwareConfig.end()) {
        it->enabled = enabled;
    } else {
        // Create new entry with empty implementation
        HardwareConfig config;
        config.enabled = enabled;
        d_hardwareConfig[hardwareType] = config;
    }
    
    qDebug() << "Set hardware enabled state:" << hardwareType << "=" << enabled;
    
    return true;
}

bool RuntimeHardwareConfig::removeHardwareSelection(const QString& hardwareType)
{
    QWriteLocker locker(&d_configLock);
    
    if (hardwareType.isEmpty()) {
        return false;
    }
    
    d_hardwareConfig.remove(hardwareType);
    
    qDebug() << "Removed hardware selection for:" << hardwareType;
    
    return true;
}

void RuntimeHardwareConfig::clearConfiguration()
{
    QWriteLocker locker(&d_configLock);
    
    qDebug() << "Clearing all hardware configuration...";
    d_hardwareConfig.clear();
}


void RuntimeHardwareConfig::saveToSettings()
{
    QWriteLocker locker(&d_configLock);
    
    qDebug() << "Saving runtime hardware configuration to settings...";
    
    // Clear existing settings - remove all previous hardware configuration keys
    for (auto it = d_hardwareConfig.cbegin(); it != d_hardwareConfig.cend(); ++it) {
        QString typeKey = it.key();
        clearValue(QString("%1_%2").arg(typeKey, BC::Key::RuntimeHw::selection));
        clearValue(QString("%1_%2").arg(typeKey, BC::Key::RuntimeHw::enabled));
    }
    
    // Save current configuration
    for (auto it = d_hardwareConfig.cbegin(); it != d_hardwareConfig.cend(); ++it) {
        QString typeKey = it.key();
        const HardwareConfig& config = it.value();
        
        set(QString("%1_%2").arg(typeKey, BC::Key::RuntimeHw::selection), config.implementation);
        set(QString("%1_%2").arg(typeKey, BC::Key::RuntimeHw::enabled), config.enabled);
    }
    
    qDebug() << "Saved" << d_hardwareConfig.size() << "hardware configurations to settings";
    
    // Persist changes to storage
    save();
}

void RuntimeHardwareConfig::loadFromSettings()
{
    qDebug() << "Loading runtime hardware configuration from settings...";
    
    // Get all setting keys without holding the config lock
    QStringList allKeys = keys();
    
    QWriteLocker locker(&d_configLock);
    d_hardwareConfig.clear();
    
    int loadedCount = 0;
    
    // Parse existing settings directly rather than querying registry
    for (const QString& key : allKeys) {
        if (key.endsWith(QString("_%1").arg(BC::Key::RuntimeHw::selection))) {
            QString type = key.left(key.length() - BC::Key::RuntimeHw::selection.length() - 1);
            QString enabledKey = QString("%1_%2").arg(type, BC::Key::RuntimeHw::enabled);
            
            HardwareConfig config;
            config.implementation = get<QString>(key, QString());
            config.enabled = get<bool>(enabledKey, true);
            
            d_hardwareConfig[type] = config;
            loadedCount++;
            
            qDebug() << "Loaded hardware config:" << type << "=" << config.implementation 
                     << "(enabled:" << config.enabled << ")";
        }
    }
    
    if (loadedCount == 0) {
        qDebug() << "No saved hardware configuration found, will use defaults when needed";
    } else {
        qDebug() << "Loaded" << loadedCount << "hardware configurations from settings";
    }
}

// ============================================================================
// PRIVATE HELPERS
// ============================================================================

HardwareValidationResult RuntimeHardwareConfig::validateHardwareTypeInternal(const QString& hardwareType, const HardwareConfig& config) const
{
    // Note: This method assumes the caller already holds the appropriate lock
    
    HardwareValidationResult result;
    
    // Check if hardware is enabled
    if (!config.enabled) {
        result.isValid = true; // Disabled hardware is considered "valid" (just not used)
        result.selectedImplementation = QString();
        result.warnings << QString("Hardware type '%1' is disabled").arg(hardwareType);
        return result;
    }
    
    if (config.implementation.isEmpty()) {
        result.isValid = false;
        result.errors << QString("No implementation selected for hardware type '%1'").arg(hardwareType);
        return result;
    }
    
    // Check if implementation is registered (simple validation only)
    HardwareRegistry& registry = HardwareRegistry::instance();
    if (!registry.isRegistered(hardwareType, config.implementation)) {
        result.isValid = false;
        result.errors << QString("Selected implementation '%1' for hardware type '%2' is not registered")
                         .arg(config.implementation, hardwareType);
        return result;
    }
    
    // NOTE: Actual instantiation and runtime validation is handled by HardwareManager
    // This validation only checks configuration completeness and registration
    
    // Validation passed
    result.isValid = true;
    result.selectedImplementation = config.implementation;
    
    // Check if this is required hardware
    if (isHardwareRequired(hardwareType)) {
        result.warnings << QString("Required hardware '%1' is configured and available").arg(hardwareType);
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
            auto it = d_hardwareConfig.find(type);
            if (it == d_hardwareConfig.end() || !it->enabled) {
                missing.append(type);
            } else {
                HardwareValidationResult result = validateHardwareTypeInternal(type, it.value());
                if (!result.isValid) {
                    missing.append(type);
                }
            }
        }
    }
    
    return missing;
}