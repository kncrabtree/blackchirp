#include "hardwareregistry.h"
#include "hardwareobject.h"
#include <hardware/library/vendorlibrary.h>

#include <QMutexLocker>
#include <QSet>
#include <QDebug>
#include <data/loghandler.h>

// Include base hardware classes for type-safe hardware type determination
#include <hardware/core/ftmwdigitizer/ftmwdigitizer.h>
#include <hardware/optional/chirpsource/awg.h>
#include <hardware/core/liflaser/liflaser.h>
#include <hardware/core/lifdigitizer/lifdigitizer.h>

HardwareRegistry* HardwareRegistry::s_instance = nullptr;

HardwareRegistry::HardwareRegistry(QObject *parent)
    : QObject(parent)
{
}

HardwareRegistry& HardwareRegistry::instance()
{
    if (!s_instance) {
        s_instance = new HardwareRegistry();
    }
    return *s_instance;
}

bool HardwareRegistry::registerHardware(const QString& key, const QString& subKey,
                                       const QString& description,
                                       std::function<HardwareObject*(const QString&)> factory,
                                       const QStringList& inheritanceChain)
{
    QMutexLocker locker(&d_registryMutex);

    QString registryKey = makeRegistryKey(key, subKey);

    // Check if already registered
    if (d_registrations.contains(registryKey)) {
        qWarning() << "Hardware already registered:" << key << subKey;
        return false;
    }

    // Validate required parameters
    if (key.isEmpty() || subKey.isEmpty() || description.isEmpty() || !factory) {
        qWarning() << "Invalid hardware registration parameters for" << key << subKey;
        return false;
    }

    // Create registration
    HardwareRegistration reg(key, subKey, description, factory, inheritanceChain);

    // Store registration
    d_registrations.insert(registryKey, reg);

    bcDebug(u"Registered hardware: %1 %2 (%3)"_s.arg(key, subKey, description));

    locker.unlock();  // Release mutex before emitting signal to avoid re-entrancy deadlocks.
    emit hardwareRegistered(key, subKey);

    return true;
}

HardwareObject* HardwareRegistry::createHardware(const QString& key, const QString& subKey, const QString& label)
{
    // Copy the factory out under the lock, then release before calling it.
    // The factory constructs a HardwareObject whose constructor calls back into
    // the registry (e.g. getSupportedProtocols), so we must not hold the mutex.
    std::function<HardwareObject*(const QString&)> factory;
    {
        QMutexLocker locker(&d_registryMutex);

        QString registryKey = makeRegistryKey(key, subKey);
        auto it = d_registrations.find(registryKey);

        if (it == d_registrations.end()) {
            qWarning() << "Hardware not registered:" << key << subKey;
            return nullptr;
        }

        factory = it.value().factory;
    }

    // Mutex is released here — safe to call back into registry from the constructor.
    HardwareObject* hardware = nullptr;
    if (factory) {
        try {
            hardware = factory(label);
            if (hardware) {
                bcDebug(u"Created hardware instance: %1 %2, label: %3"_s.arg(key, subKey, label));
            } else {
                qWarning() << "Factory returned null for" << key << subKey;
            }
        } catch (...) {
            qWarning() << "Exception in factory function for" << key << subKey;
            hardware = nullptr;
        }
    }

    return hardware;
}

QStringList HardwareRegistry::getImplementations(const QString& key)
{
    QMutexLocker locker(&d_registryMutex);
    
    QStringList implementations;
    
    for (auto it = d_registrations.cbegin(); it != d_registrations.cend(); ++it) {
        const HardwareRegistration& reg = it.value();
        if (reg.key == key) {
            implementations.append(reg.subKey);
        }
    }
    
    return implementations;
}

QStringList HardwareRegistry::getHardwareTypes()
{
    QMutexLocker locker(&d_registryMutex);
    
    QStringList types;
    
    for (auto it = d_registrations.cbegin(); it != d_registrations.cend(); ++it) {
        const HardwareRegistration& reg = it.value();
        if (!types.contains(reg.key)) {
            types.append(reg.key);
        }
    }
    
    return types;
}

const HardwareRegistration* HardwareRegistry::getRegistration(const QString& key, const QString& subKey)
{
    QMutexLocker locker(&d_registryMutex);
    
    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);
    
    if (it == d_registrations.end()) {
        return nullptr;
    }
    
    return &it.value();
}

bool HardwareRegistry::isRegistered(const QString& key, const QString& subKey)
{
    QMutexLocker locker(&d_registryMutex);
    
    QString registryKey = makeRegistryKey(key, subKey);
    return d_registrations.contains(registryKey);
}

bool HardwareRegistry::isMultiInstanceType(const QString& hardwareType)
{
    // Type-safe helper function to extract hardware type from class
    auto hardwareTypeOf = [](auto* typePtr) -> QString {
        using T = std::remove_pointer_t<decltype(typePtr)>;
        return QString(T::staticMetaObject.className());
    };
    
    // Single-instance hardware types (only one instance allowed)
    static const QStringList singleInstanceTypes = {
        hardwareTypeOf(static_cast<FtmwDigitizer*>(nullptr)),
        hardwareTypeOf(static_cast<AWG*>(nullptr)),
        hardwareTypeOf(static_cast<LifLaser*>(nullptr)),
        hardwareTypeOf(static_cast<LifDigitizer*>(nullptr))
    };
    
    // If hardware type is in the single-instance list, return false
    // All other types are multi-instance by default
    return !singleInstanceTypes.contains(hardwareType);
}

QString HardwareRegistry::makeRegistryKey(const QString& key, const QString& subKey) const
{
    return QString("%1::%2").arg(key, subKey);
}

bool HardwareRegistry::addSupportedProtocols(const QString& key, const QString& subKey,
                                            const QVector<CommunicationProtocol::CommType>& protocols)
{
    QMutexLocker locker(&d_registryMutex);

    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);

    if (it == d_registrations.end()) {
        qWarning() << "Cannot add supported protocols - hardware not registered:" << key << subKey;
        return false;
    }

    d_registrations[registryKey].supportedProtocols = protocols;
    return true;
}

QVector<CommunicationProtocol::CommType> HardwareRegistry::getSupportedProtocols(
    const QString& key, const QString& subKey) const
{
    QMutexLocker locker(&d_registryMutex);

    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);

    if (it == d_registrations.end())
        return {};

    return it.value().supportedProtocols;
}

bool HardwareRegistry::addSettingDefs(const QString& key, const QString& subKey,
                                      const QVector<HwSettingDef>& settings)
{
    QMutexLocker locker(&d_registryMutex);

    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);

    if (it == d_registrations.end()) {
        qWarning() << "Cannot add setting defs - hardware not registered:" << key << subKey;
        return false;
    }

    it.value().settingDefs.append(settings);
    return true;
}

QVector<HwSettingDef> HardwareRegistry::getSettingDefs(const QString& key, const QString& subKey) const
{
    QMutexLocker locker(&d_registryMutex);

    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);

    if (it == d_registrations.end())
        return {};

    QVector<HwSettingDef> result = it.value().settingDefs;

    QSet<QString> presentKeys;
    for (const auto& def : result)
        presentKeys.insert(def.key);

    // Append base class settings, skipping any key already defined by the implementation
    for (const QString& baseClass : it.value().inheritanceChain) {
        auto baseIt = d_baseSettingDefs.find(baseClass);
        if (baseIt != d_baseSettingDefs.end()) {
            for (const auto& def : *baseIt) {
                if (!presentKeys.contains(def.key)) {
                    result.append(def);
                    presentKeys.insert(def.key);
                }
            }
        }
    }

    return result;
}

bool HardwareRegistry::addArraySettingDef(const QString& key, const QString& subKey,
                                          const QString& arrayKey, const QString& label,
                                          const QString& description, HwSettingPriority priority)
{
    QMutexLocker locker(&d_registryMutex);

    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);

    if (it == d_registrations.end()) {
        qWarning() << "Cannot add array setting def - hardware not registered:" << key << subKey;
        return false;
    }

    it.value().arraySettingDefs[arrayKey] = HwArraySettingDef{arrayKey, label, description, {}, priority};
    return true;
}

bool HardwareRegistry::addArraySettingEntry(const QString& key, const QString& subKey,
                                            const QString& arrayKey,
                                            const SettingsStorage::SettingsMap& entry)
{
    QMutexLocker locker(&d_registryMutex);

    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);

    if (it == d_registrations.end()) {
        qWarning() << "Cannot add array setting entry - hardware not registered:" << key << subKey;
        return false;
    }

    auto arrayIt = it.value().arraySettingDefs.find(arrayKey);
    if (arrayIt == it.value().arraySettingDefs.end()) {
        qWarning() << "Cannot add array entry - array key not registered:" << arrayKey << "for" << key << subKey;
        return false;
    }

    arrayIt.value().entries.push_back(entry);
    return true;
}

QMap<QString, HwArraySettingDef> HardwareRegistry::getArraySettingDefs(
    const QString& key, const QString& subKey) const
{
    QMutexLocker locker(&d_registryMutex);

    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);

    if (it == d_registrations.end())
        return {};

    // Start with base class array defs (outermost ancestor first), then let
    // nearer bases and the implementation override by inserting later.
    QMap<QString, HwArraySettingDef> result;
    const QStringList& chain = it.value().inheritanceChain;
    for (int i = chain.size() - 1; i >= 0; --i) {
        auto baseIt = d_baseArrayDefs.find(chain[i]);
        if (baseIt != d_baseArrayDefs.end()) {
            for (auto aIt = baseIt->cbegin(); aIt != baseIt->cend(); ++aIt)
                result.insert(aIt.key(), aIt.value());
        }
    }
    // Implementation-specific defs override base class defs for the same key
    for (auto aIt = it.value().arraySettingDefs.cbegin();
         aIt != it.value().arraySettingDefs.cend(); ++aIt)
        result.insert(aIt.key(), aIt.value());

    return result;
}

bool HardwareRegistry::addBaseSettingDefs(const QString& className,
                                          const QVector<HwSettingDef>& settings)
{
    QMutexLocker locker(&d_registryMutex);
    d_baseSettingDefs[className].append(settings);
    return true;
}

bool HardwareRegistry::addBaseArraySettingDef(const QString& className, const QString& arrayKey,
                                              const QString& label, const QString& description,
                                              HwSettingPriority priority)
{
    QMutexLocker locker(&d_registryMutex);
    d_baseArrayDefs[className][arrayKey] = HwArraySettingDef{arrayKey, label, description, {}, priority};
    return true;
}

bool HardwareRegistry::addBaseArraySettingEntry(const QString& className, const QString& arrayKey,
                                                const SettingsStorage::SettingsMap& entry)
{
    QMutexLocker locker(&d_registryMutex);

    auto classIt = d_baseArrayDefs.find(className);
    if (classIt == d_baseArrayDefs.end()) {
        qWarning() << "Cannot add base array entry - class not registered:" << className;
        return false;
    }

    auto arrayIt = classIt->find(arrayKey);
    if (arrayIt == classIt->end()) {
        qWarning() << "Cannot add base array entry - array key not registered:" << arrayKey << "for" << className;
        return false;
    }

    arrayIt->entries.push_back(entry);
    return true;
}

bool HardwareRegistry::addCustomCommDefs(const QString& key, const QString& subKey,
                                         const QVector<CustomCommDef>& defs)
{
    QMutexLocker locker(&d_registryMutex);

    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);

    if (it == d_registrations.end()) {
        qWarning() << "Cannot add custom comm defs - hardware not registered:" << key << subKey;
        return false;
    }

    it.value().customCommDefs.append(defs);
    return true;
}

QVector<CustomCommDef> HardwareRegistry::getCustomCommDefs(const QString& key, const QString& subKey) const
{
    QMutexLocker locker(&d_registryMutex);

    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);

    if (it == d_registrations.end())
        return {};

    QVector<CustomCommDef> result = it.value().customCommDefs;

    QSet<QString> presentKeys;
    for (const auto& def : result)
        presentKeys.insert(def.key);

    for (const QString& baseClass : it.value().inheritanceChain) {
        auto baseIt = d_baseCustomCommDefs.find(baseClass);
        if (baseIt != d_baseCustomCommDefs.end()) {
            for (const auto& def : *baseIt) {
                if (!presentKeys.contains(def.key)) {
                    result.append(def);
                    presentKeys.insert(def.key);
                }
            }
        }
    }

    return result;
}

bool HardwareRegistry::addBaseCustomCommDefs(const QString& className,
                                             const QVector<CustomCommDef>& defs)
{
    QMutexLocker locker(&d_registryMutex);
    d_baseCustomCommDefs[className].append(defs);
    return true;
}

bool HardwareRegistry::addLibraryDependency(const QString& key, const QString& subKey, const QString& libraryName,
                                           std::function<VendorLibrary*()> libraryGetter)
{
    QMutexLocker locker(&d_registryMutex);
    
    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);
    
    if (it == d_registrations.end()) {
        qWarning() << "Cannot add library dependency - hardware not registered:" << key << subKey;
        return false;
    }
    
    if (!it.value().libraryDependencies.contains(libraryName)) {
        it.value().libraryDependencies.append(libraryName);
        bcDebug(u"Added library dependency: %1 to %2 %3"_s.arg(libraryName, key, subKey));
    }
    
    // Store the library getter function
    if (!d_libraryGetters.contains(libraryName)) {
        d_libraryGetters.insert(libraryName, libraryGetter);
        bcDebug(u"Registered library getter for: %1"_s.arg(libraryName));
    }
    
    return true;
}

QStringList HardwareRegistry::getLibraryDependencies(const QString& implementationName) const
{
    QMutexLocker locker(&d_registryMutex);
    
    // Find the registration with this implementation name
    for (const auto& reg : d_registrations) {
        if (reg.subKey == implementationName) {
            return reg.libraryDependencies;
        }
    }
    
    return {};  // No dependencies found
}

QStringList HardwareRegistry::getHardwareDependingOnLibrary(const QString& libraryName) const
{
    QMutexLocker locker(&d_registryMutex);
    
    QStringList dependentHardware;
    for (const auto& reg : d_registrations) {
        if (reg.libraryDependencies.contains(libraryName)) {
            dependentHardware.append(reg.subKey);
        }
    }
    
    return dependentHardware;
}

bool HardwareRegistry::hardwareUsesLibrary(const QString& implementationName, const QString& libraryName) const
{
    return getLibraryDependencies(implementationName).contains(libraryName);
}

QStringList HardwareRegistry::getLibrariesWithChanges() const
{
    QMutexLocker locker(&d_registryMutex);
    
    QStringList changedLibraries;
    
    // Check all registered library getters
    for (auto it = d_libraryGetters.cbegin(); it != d_libraryGetters.cend(); ++it) {
        const QString& libraryName = it.key();
        const auto& libraryGetter = it.value();
        
        try {
            VendorLibrary* lib = libraryGetter();
            if (lib && lib->hasUnstagedChanges()) {
                changedLibraries.append(libraryName);
            }
        } catch (...) {
            qWarning() << "Exception when checking library changes for:" << libraryName;
        }
    }
    
    return changedLibraries;
}