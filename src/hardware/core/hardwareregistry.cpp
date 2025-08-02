#include "hardwareregistry.h"
#include "hardwareobject.h"

#include <QMutexLocker>
#include <QDebug>

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
                                       const QString& prettyName, const QString& description,
                                       std::function<HardwareObject*()> factory)
{
    QMutexLocker locker(&d_registryMutex);
    
    QString registryKey = makeRegistryKey(key, subKey);
    
    // Check if already registered
    if (d_registrations.contains(registryKey)) {
        qWarning() << "Hardware already registered:" << key << subKey;
        return false;
    }
    
    // Validate required parameters
    if (key.isEmpty() || subKey.isEmpty() || prettyName.isEmpty() || !factory) {
        qWarning() << "Invalid hardware registration parameters for" << key << subKey;
        return false;
    }
    
    // Create registration
    HardwareRegistration reg(key, subKey, prettyName, description, factory);
    
    // Store registration
    d_registrations.insert(registryKey, reg);
    
    qDebug() << "Registered hardware:" << key << subKey << "(" << prettyName << ")";
    
    // Signal registration
    emit hardwareRegistered(key, subKey);
    
    return true;
}

HardwareObject* HardwareRegistry::createHardware(const QString& key, const QString& subKey)
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
            hardware = reg.factory();
            if (hardware) {
                qDebug() << "Created hardware instance:" << key << subKey;
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

QString HardwareRegistry::makeRegistryKey(const QString& key, const QString& subKey) const
{
    return QString("%1::%2").arg(key, subKey);
}