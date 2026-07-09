#include "hardwareregistration.h"
#include "hardwareregistry.h"

#include <QHash>
#include <QMutex>
#include <QMutexLocker>

using namespace Qt::Literals::StringLiterals;

void initializeHardwareRegistrations()
{
    // Hardware registrations are performed automatically through static
    // HardwareAutoRegistration instances in each hardware implementation file.
    // This function serves as a centralized point for any additional
    // registration logic if needed in the future.

    // Force evaluation of any lazy registration by accessing the registry
    HardwareRegistry::instance();
}

namespace {

QHash<QString, QStringList> &arraySchemaRegistry()
{
    static QHash<QString, QStringList> reg;
    return reg;
}

QMutex &arraySchemaMutex()
{
    static QMutex m;
    return m;
}

QString arraySchemaRegistryKey(const QString &key, const QString &subKey, const QString &arrayKey)
{
    return key + u"::"_s + subKey + u"::"_s + arrayKey;
}

} // namespace

bool registerHardwareArraySchema(const QString &key, const QString &subKey,
                                  const QString &arrayKey, const QStringList &subKeys)
{
    QMutexLocker locker(&arraySchemaMutex());
    arraySchemaRegistry()[arraySchemaRegistryKey(key, subKey, arrayKey)] = subKeys;
    return true;
}

QStringList hardwareArraySchema(const QString &key, const QString &subKey, const QString &arrayKey)
{
    QMutexLocker locker(&arraySchemaMutex());
    return arraySchemaRegistry().value(arraySchemaRegistryKey(key, subKey, arrayKey));
}
