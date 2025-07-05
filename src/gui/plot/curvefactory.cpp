#include "curvefactory.h"
#include "blackchirpplotcurve.h"
#include <data/experiment/overlaybase.h>
#include <data/bcglobals.h>

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

OverlayMetadataStorage::OverlayMetadataStorage(OverlayBase* overlay)
    : d_overlay(overlay)
{
    syncFromOverlay();
}

void OverlayMetadataStorage::set(const QString& key, const QVariant& value)
{
    d_cache[key] = value;
    syncToOverlay();
    d_overlay->setModified(true);
}

QVariant OverlayMetadataStorage::get(const QString& key, const QVariant& defaultValue) const
{
    auto it = d_cache.find(key);
    return (it != d_cache.end()) ? it->second : defaultValue;
}

void OverlayMetadataStorage::syncFromOverlay()
{
    // Load curve settings from overlay metadata
    std::map<QString, QVariant> meta;
    d_overlay->_storeMetadata(meta);
    
    // Extract curve-specific keys (those with "curve_" prefix)
    d_cache.clear();
    for (const auto& [key, value] : meta) {
        if (key.startsWith("curve_")) {
            d_cache[key.mid(6)] = value; // Remove "curve_" prefix
        }
    }
}

void OverlayMetadataStorage::syncToOverlay()
{
    // Get current metadata from overlay
    std::map<QString, QVariant> meta;
    d_overlay->_storeMetadata(meta);
    
    // Remove old curve settings
    auto it = meta.begin();
    while (it != meta.end()) {
        if (it->first.startsWith("curve_")) {
            it = meta.erase(it);
        } else {
            ++it;
        }
    }
    
    // Add current curve settings with "curve_" prefix
    for (const auto& [key, value] : d_cache) {
        meta["curve_" + key] = value;
    }
    
    // Store updated metadata back to overlay
    d_overlay->_retrieveMetadata(meta);
}