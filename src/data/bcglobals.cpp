#include "bcglobals.h"

#include <QStringList>

// New label-based API
QString BC::Key::hwKey(const QString& type, const QString& label)
{
    return QString("%1%2%3").arg(type, hwIndexSep, label);
}

std::pair<QString, QString> BC::Key::parseKey(const QString& key)
{
    QStringList l = key.split(hwIndexSep);
    if(l.size() < 2)
        return {key, QString()};
    else {
        return {l.at(0), l.at(1)};
    }
}

// Legacy index-based API (deprecated)
QString BC::Key::hwKey(const QString& k, const int index)
{
    return QString("%1%2%3").arg(k, hwIndexSep, QString::number(index));
}

std::pair<QString, int> BC::Key::parseIndexKey(const QString& key)
{
   QStringList l = key.split(hwIndexSep);
   if(l.size() < 2)
       return {key, -1};
   else {
       bool ok = false;
       auto idx = l.at(1).toInt(&ok);
       return {l.at(0), ok ? idx : -1};
   }
}

QString BC::Key::widgetKey(const QString& widgetKey, const QString& hwKey)
{
    return QString("%1%2%3").arg(widgetKey, hwIndexSep, hwKey);
}

// Migration utilities
QString BC::Key::migrateIndexKey(const QString& oldKey, const QString& type, int index)
{
    // Convert index-based key to label-based key
    auto [keyType, keyIndex] = parseIndexKey(oldKey);
    if (keyType == type && keyIndex == index) {
        // Generate default label based on index
        QString defaultLabel = QString("device%1").arg(index);
        return hwKey(type, defaultLabel);
    }
    return oldKey; // Not a matching index key
}

bool BC::Key::isIndexKey(const QString& key)
{
    auto [type, index] = parseIndexKey(key);
    return index >= 0; // Valid index indicates old format
}

QString BC::Key::generateDefaultLabel(const QString& /* type */, const QStringList& existingLabels)
{
    // Default label priority order
    static const QStringList defaultLabels = {
        "default", "main", "primary", "secondary", "backup"
    };
    
    // Try standard defaults first
    for (const QString& label : defaultLabels) {
        if (!existingLabels.contains(label)) {
            return label;
        }
    }
    
    // Fall back to numbered defaults
    int counter = 1;
    QString candidate;
    do {
        candidate = QString("device%1").arg(counter++);
    } while (existingLabels.contains(candidate));
    
    return candidate;
}
