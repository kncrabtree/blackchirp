#include "curvefactory.h"
#include "blackchirpplotcurve.h"
#include <data/experiment/overlaybase.h>
#include <data/bcglobals.h>
#include <QDebug>

// SettingsStorageWrapper Implementation

SettingsStorageWrapper::SettingsStorageWrapper(const QString& key, SettingsStorage::Type type)
    : SettingsStorage(QStringList{BC::Key::bcCurve, key}, type)
{
}

void SettingsStorageWrapper::set(const QString& key, const QVariant& value)
{
    SettingsStorage::set(key, value);
}

QVariant SettingsStorageWrapper::get(const QString& key, const QVariant& defaultValue) const
{
    return SettingsStorage::get(key, defaultValue);
}

// OverlayMetadataStorage Implementation

OverlayMetadataStorage::OverlayMetadataStorage(std::shared_ptr<OverlayBase> overlay)
    : d_overlay(overlay)
{
    // No need to sync from overlay - we'll access d_curveMetadata directly
}

void OverlayMetadataStorage::set(const QString& key, const QVariant& value)
{
    if (!d_overlay) {
        return;
    }
    
    // Direct access to overlay's curve metadata via friend class access
    d_overlay->d_curveMetadata[key] = value;
    d_overlay->setModified(true);
}

QVariant OverlayMetadataStorage::get(const QString& key, const QVariant& defaultValue) const
{
    if (!d_overlay) {
        return defaultValue;
    }
    
    // Direct access to overlay's curve metadata via friend class access
    auto it = d_overlay->d_curveMetadata.find(key);
    return (it != d_overlay->d_curveMetadata.end()) ? it->second : defaultValue;
}
