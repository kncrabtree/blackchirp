#include "hardwareregistry.h"
#include "hardwareobject.h"

#include <QMutexLocker>
#include <QDebug>

// Include base hardware classes for type-safe hardware type determination
#include <hardware/core/ftmwdigitizer/ftmwscope.h>
#include <hardware/optional/chirpsource/awg.h>
#include <hardware/core/liflaser/liflaser.h>
#include <hardware/core/lifdigitizer/lifscope.h>

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
                                       std::function<HardwareObject*(const QString&)> factory)
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
    HardwareRegistration reg(key, subKey, description, factory);
    
    // Store registration
    d_registrations.insert(registryKey, reg);
    
    qDebug() << "Registered hardware:" << key << subKey << "(" << description << ")";
    
    // Signal registration
    emit hardwareRegistered(key, subKey);
    
    return true;
}

HardwareObject* HardwareRegistry::createHardware(const QString& key, const QString& subKey, const QString& label)
{
    QMutexLocker locker(&d_registryMutex);
    
    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);
    
    if (it == d_registrations.end()) {
        qWarning() << "Hardware not registered:" << key << subKey;
        return nullptr;
    }
    
    const HardwareRegistration& reg = it.value();
    
    // Create hardware instance
    HardwareObject* hardware = nullptr;
    if (reg.factory) {
        try {
            hardware = reg.factory(label);
            if (hardware) {
                qDebug() << "Created hardware instance:" << key << subKey << "with label:" << label;
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
        hardwareTypeOf(static_cast<FtmwScope*>(nullptr)),
        hardwareTypeOf(static_cast<AWG*>(nullptr)),
        hardwareTypeOf(static_cast<LifLaser*>(nullptr)),
        hardwareTypeOf(static_cast<LifScope*>(nullptr))
    };
    
    // If hardware type is in the single-instance list, return false
    // All other types are multi-instance by default
    return !singleInstanceTypes.contains(hardwareType);
}

QString HardwareRegistry::makeRegistryKey(const QString& key, const QString& subKey) const
{
    return QString("%1::%2").arg(key, subKey);
}