#include "hardwareregistration.h"
#include "hardwareregistry.h"

#include <QDebug>

void initializeHardwareRegistrations()
{
    qDebug() << "Initializing hardware registrations...";
    
    // Hardware registrations are performed automatically through static
    // HardwareAutoRegistration instances in each hardware implementation file.
    // This function serves as a centralized point for any additional
    // registration logic if needed in the future.
    
    // Force evaluation of any lazy registration by accessing the registry
    HardwareRegistry& registry = HardwareRegistry::instance();
    
    // Log registered hardware types
    QStringList types = registry.getRegisteredHardwareTypes();
    qDebug() << "Registered hardware types:" << types;
    
    for (const QString& type : types) {
        QStringList implementations = registry.getAvailableImplementations(type);
        qDebug() << QString("Available implementations for %1:").arg(type) << implementations;
    }
    
    qDebug() << "Hardware registration initialization complete";
}