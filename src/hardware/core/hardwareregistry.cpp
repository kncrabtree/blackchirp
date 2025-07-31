#include "hardwareregistry.h"
#include "hardwareobject.h"
#include <hardware/library/vendorlibrary.h>

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
                                       const QStringList& dependencies,
                                       std::function<HardwareObject*()> factory,
                                       std::function<bool()> availabilityCheck,
                                       bool required)
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
    HardwareRegistration reg(key, subKey, prettyName, description, dependencies,
                           factory, availabilityCheck, required);
    
    // Store registration
    d_registrations.insert(registryKey, reg);
    
    qDebug() << "Registered hardware:" << key << subKey << "(" << prettyName << ")";
    
    // Signal registration
    emit hardwareRegistered(key, subKey);
    
    return true;
}

bool HardwareRegistry::isHardwareAvailable(const QString& key, const QString& subKey)
{
    QMutexLocker locker(&d_registryMutex);
    
    QString registryKey = makeRegistryKey(key, subKey);
    auto it = d_registrations.find(registryKey);
    
    if (it == d_registrations.end()) {
        return false;
    }
    
    HardwareRegistration& reg = it.value();
    
    // Check if availability has been cached
    if (reg.availability == HardwareAvailability::Unknown) {
        // Run availability check
        bool available = false;
        if (reg.availabilityCheck) {
            try {
                available = reg.availabilityCheck();
            } catch (...) {
                qWarning() << "Exception in availability check for" << key << subKey;
                available = false;
            }
        }
        
        // Cache result
        reg.availability = available ? HardwareAvailability::Available : HardwareAvailability::Unavailable;
        
        qDebug() << "Checked availability for" << key << subKey << ":" << available;
    }
    
    return reg.availability == HardwareAvailability::Available;
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
    
    // Check availability first
    locker.unlock();
    if (!isHardwareAvailable(key, subKey)) {
        qWarning() << "Hardware not available:" << key << subKey;
        return nullptr;
    }
    locker.relock();
    
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

QStringList HardwareRegistry::getAvailableImplementations(const QString& key)
{
    QMutexLocker locker(&d_registryMutex);
    
    QStringList implementations;
    
    for (auto it = d_registrations.cbegin(); it != d_registrations.cend(); ++it) {
        const HardwareRegistration& reg = it.value();
        if (reg.key == key) {
            // Check availability (this will unlock/relock the mutex)
            locker.unlock();
            bool available = isHardwareAvailable(key, reg.subKey);
            locker.relock();
            
            if (available) {
                implementations.append(reg.subKey);
            }
        }
    }
    
    return implementations;
}

QStringList HardwareRegistry::getRegisteredHardwareTypes()
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

void HardwareRegistry::refreshAvailability()
{
    QMutexLocker locker(&d_registryMutex);
    
    qDebug() << "Refreshing hardware availability...";
    
    for (auto it = d_registrations.begin(); it != d_registrations.end(); ++it) {
        HardwareRegistration& reg = it.value();
        HardwareAvailability oldAvailability = reg.availability;
        
        // Reset cached availability
        reg.availability = HardwareAvailability::Unknown;
        
        // Check new availability (this will unlock/relock the mutex)
        locker.unlock();
        bool available = isHardwareAvailable(reg.key, reg.subKey);
        locker.relock();
        
        // Emit signal if availability changed
        if ((oldAvailability == HardwareAvailability::Available) != available) {
            emit hardwareAvailabilityChanged(reg.key, reg.subKey, available);
        }
    }
    
    qDebug() << "Hardware availability refresh complete";
}

bool HardwareRegistry::allRequiredHardwareAvailable()
{
    QMutexLocker locker(&d_registryMutex);
    
    for (auto it = d_registrations.cbegin(); it != d_registrations.cend(); ++it) {
        const HardwareRegistration& reg = it.value();
        if (reg.isRequired) {
            // Check availability (this will unlock/relock the mutex)
            locker.unlock();
            bool available = isHardwareAvailable(reg.key, reg.subKey);
            locker.relock();
            
            if (!available) {
                return false;
            }
        }
    }
    
    return true;
}

QStringList HardwareRegistry::getUnavailableRequiredHardware()
{
    QMutexLocker locker(&d_registryMutex);
    
    QStringList unavailable;
    
    for (auto it = d_registrations.cbegin(); it != d_registrations.cend(); ++it) {
        const HardwareRegistration& reg = it.value();
        if (reg.isRequired) {
            // Check availability (this will unlock/relock the mutex)
            locker.unlock();
            bool available = isHardwareAvailable(reg.key, reg.subKey);
            locker.relock();
            
            if (!available) {
                unavailable.append(QString("%1 (%2)").arg(reg.prettyName, reg.subKey));
            }
        }
    }
    
    return unavailable;
}

QString HardwareRegistry::getDefaultImplementation(const QString& key)
{
    QMutexLocker locker(&d_registryMutex);
    
    QString defaultImpl;
    QString fallbackImpl;
    
    for (auto it = d_registrations.cbegin(); it != d_registrations.cend(); ++it) {
        const HardwareRegistration& reg = it.value();
        if (reg.key == key) {
            // Check availability (this will unlock/relock the mutex)
            locker.unlock();
            bool available = isHardwareAvailable(key, reg.subKey);
            locker.relock();
            
            if (available) {
                if (reg.isRequired && defaultImpl.isEmpty()) {
                    // Prefer required hardware implementations
                    defaultImpl = reg.subKey;
                } else if (fallbackImpl.isEmpty()) {
                    // Fallback to any available implementation
                    fallbackImpl = reg.subKey;
                }
            }
        }
    }
    
    return defaultImpl.isEmpty() ? fallbackImpl : defaultImpl;
}

QString HardwareRegistry::makeRegistryKey(const QString& key, const QString& subKey) const
{
    return QString("%1::%2").arg(key, subKey);
}