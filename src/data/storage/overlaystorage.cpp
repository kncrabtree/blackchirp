#include "overlaystorage.h"
#include <data/experiment/overlaytypes.h>
#include <data/storage/blackchirpcsv.h>
#include <QDir>
#include <QDebug>
#include <QRegularExpression>
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>


OverlayStorage::OverlayStorage(int number, QString path) :
    QObject(), DataStorageBase(number, path)
{
    if (number < 1)
        return;
        
    // Look for overlays.csv file, if found, load overlays
    std::map<QString,QVariant> m;
    readMetadata(BC::Key::Overlay::overlayMdFile, m, BC::Key::Overlay::overlayDir);
    
    for(auto it = m.cbegin(); it != m.cend(); ++it)
    {
        // Skip version metadata entries
        if (it->first.startsWith("BC") && it->first.endsWith("Version"))
            continue;
            
        auto fileBase = it->first;
        auto type = it->second.value<OverlayBase::OverlayType>();
        
        if (!loadOverlay(fileBase, type))
        {
            qDebug() << "Failed to load overlay:" << fileBase;
        }
    }
}

OverlayStorage::~OverlayStorage()
{
    // Cleanup is automatic with shared_ptr
}


bool OverlayStorage::loadOverlay(QString fileBase, OverlayBase::OverlayType t)
{
    if (d_number < 1)
        return false;
        
    // Check if overlay already loaded
    auto it = d_overlays.find(fileBase);
    if (it != d_overlays.end())
        return true; // Already loaded
    
    // Create overlay object
    auto overlay = createOverlayObject(t);
    if (!overlay)
        return false;
    
    // Load overlay settings
    using namespace BC::Key::Overlay;
    std::map<QString,QVariant> overlaySettings;
    QString settingsFile = overlaySettingsFile.arg(fileBase);
    readMetadata(settingsFile, overlaySettings, overlayDir);
    
    if (overlaySettings.empty())
    {
        qDebug() << "No settings found for overlay:" << fileBase;
        return false;
    }
    
    // Set destination file path
    overlay->setDestFile(getOverlayDataPath(fileBase));
    
    // Load metadata into overlay
    overlay->retrieveMetadata(overlaySettings);
    
    // Load XY data from destination file
    overlay->readFromDest();
    
    // Mark as unmodified since we just loaded from disk
    overlay->setModified(false);
    
    // Add to internal storage
    d_overlays[fileBase] = overlay;
    
    return true;
}

void OverlayStorage::save()
{
    if (d_number < 1)
        return;
    
    // Create overlays directory if it doesn't exist
    QDir expDir = BlackchirpCSV::exptDir(d_number, d_path);
    if (!expDir.exists(BC::Key::Overlay::overlayDir))
    {
        if (!expDir.mkpath(BC::Key::Overlay::overlayDir))
        {
            qDebug() << "Failed to create overlays directory for experiment" << d_number;
            return;
        }
    }
    
    using namespace BC::Key::Overlay;
    std::map<QString,QVariant> m;
    
    // Add version information
    addVersionMetadata(m);
    
    // Save each overlay
    for (auto it = d_overlays.begin(); it != d_overlays.end(); ++it)
    {
        const QString& label = it->first;
        auto overlay = it->second;
        
        // Add overlay type to main index
        m.emplace(label, static_cast<int>(overlay->type()));
        
        // Save overlay-specific settings (metadata only)
        std::map<QString,QVariant> overlaySettings;
        overlay->storeMetadata(overlaySettings);
        addVersionMetadata(overlaySettings);
        QString settingsFile = overlaySettingsFile.arg(label);
        writeMetadata(settingsFile, overlaySettings, overlayDir);
        
        // Note: xyData is written asynchronously when overlay is added
    }
    
    // Save main overlays index
    writeMetadata(overlayMdFile, m, overlayDir);
}

QVector<std::shared_ptr<OverlayBase>> OverlayStorage::getAllOverlays() const
{
    QVector<std::shared_ptr<OverlayBase>> result;
    result.reserve(d_overlays.size());
    
    for (const auto& pair : d_overlays)
    {
        result.append(pair.second);
    }
    
    return result;
}

std::shared_ptr<OverlayBase> OverlayStorage::createOverlayObject(OverlayBase::OverlayType type)
{
    switch (type)
    {
    case OverlayBase::BCExperiment:
        return std::make_shared<BCExpOverlay>();
    case OverlayBase::Catalog:
        return std::make_shared<CatalogOverlay>();
    case OverlayBase::GenericXY:
        // TODO: Implement when GenericXY overlay type is available
        qDebug() << "GenericXY overlay type not yet implemented";
        return nullptr;
    default:
        qDebug() << "Unknown overlay type:" << static_cast<int>(type);
        return nullptr;
    }
}

QString OverlayStorage::sanitizeLabel(const QString& label) const
{
    QString sanitized = label;
    
    // Remove or replace characters that are problematic for filenames
    sanitized.replace(QRegularExpression("[/\\\\:*?\"<>|]"), "_");
    
    // Trim whitespace
    sanitized = sanitized.trimmed();
    
    // Ensure it's not empty
    if (sanitized.isEmpty())
        sanitized = "overlay";
    
    return sanitized;
}

void OverlayStorage::addVersionMetadata(std::map<QString, QVariant>& metadata) const
{
    using namespace BC::Key::Overlay;
    metadata.emplace(bcMajorVersion, STRINGIFY(BC_MAJOR_VERSION));
    metadata.emplace(bcMinorVersion, STRINGIFY(BC_MINOR_VERSION));
    metadata.emplace(bcPatchVersion, STRINGIFY(BC_PATCH_VERSION));
    metadata.emplace(bcReleaseVersion, STRINGIFY(BC_RELEASE_VERSION));
    metadata.emplace(bcBuildVersion, STRINGIFY(BC_BUILD_VERSION));
}

bool OverlayStorage::validateOverlayLabel(const QString& label) const
{
    if (label.isEmpty())
        return false;
        
    // Check length constraints
    if (label.length() > 255)
        return false;
        
    // Check for reserved names or problematic patterns
    if (label.startsWith(".") || label == "..")
        return false;
        
    return true;
}

bool OverlayStorage::addOverlay(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay)
        return false;
        
    QString label = overlay->getLabel();
    if (!validateOverlayLabel(label))
        return false;
        
    QString sanitizedLabel = sanitizeLabel(label);
    
    // Check if overlay with this label already exists
    if (d_overlays.find(sanitizedLabel) != d_overlays.end())
        return false;
        
    // Update overlay label to sanitized version if needed
    if (sanitizedLabel != label)
        overlay->setLabel(sanitizedLabel);
    

    // Add to storage
    d_overlays[sanitizedLabel] = overlay;


    //from here on, only do the disk writing if the number is >0 and not in preview mode

    if(d_number > 0 && !overlay->isPreview())
    {
        // Set destination file path for the overlay data
        overlay->setDestFile(getOverlayDataPath(sanitizedLabel));

        // Save metadata and create directory structure
        save();

        // Start background write of xyData
        auto future = QtConcurrent::run([this, overlay, sanitizedLabel]() {
            try {
                overlay->writeToDest();
                // Signal success on main thread
                QMetaObject::invokeMethod(this, [this, sanitizedLabel]() {
                    onWriteCompleted(sanitizedLabel, true);
                }, Qt::QueuedConnection);
            } catch (const std::exception& e) {
                // Signal failure on main thread
                QMetaObject::invokeMethod(this, [this, sanitizedLabel, e]() {
                    onWriteCompleted(sanitizedLabel, false, e.what());
                }, Qt::QueuedConnection);
            } catch (...) {
                // Signal failure on main thread
                QMetaObject::invokeMethod(this, [this, sanitizedLabel]() {
                    onWriteCompleted(sanitizedLabel, false, "Unknown error during data write");
                }, Qt::QueuedConnection);
            }
        });

        // Track the pending write
        d_pendingWrites[sanitizedLabel] = future;
        emit pendingWritesChanged(d_pendingWrites.size());
    }
    
    // Emit signal that overlay was added to storage
    emit overlayAdded(overlay);
    
    return true;
}

bool OverlayStorage::removeOverlay(const QString& label)
{
    auto it = d_overlays.find(label);
    if (it != d_overlays.end())
    {
        // Wait for any pending write to complete before removing
        auto writeIt = d_pendingWrites.find(label);
        if (writeIt != d_pendingWrites.end()) {
            writeIt->second.waitForFinished();
            d_pendingWrites.erase(writeIt);
            emit pendingWritesChanged(d_pendingWrites.size());
        }
        
        // Delete associated files from disk (data file and metadata file)
        QString dataFilePath = getOverlayDataPath(label);
        if (QFile::exists(dataFilePath)) {
            if (!QFile::remove(dataFilePath)) {
                qDebug() << "Warning: Failed to delete overlay data file:" << dataFilePath;
            }
        }
        
        QString settingsFilePath = getOverlaySettingsPath(label);
        if (QFile::exists(settingsFilePath)) {
            if (!QFile::remove(settingsFilePath)) {
                qDebug() << "Warning: Failed to delete overlay settings file:" << settingsFilePath;
            }
        }
        
        // Store reference to overlay before removing from storage
        auto overlay = it->second;
        
        d_overlays.erase(it);
        
        // Update the overlays.csv file to remove the overlay reference
        save();
        
        // Emit signal that overlay was removed
        emit overlayRemoved(overlay);
        
        return true;
    }
    return false;
}

bool OverlayStorage::renameOverlay(const QString& currentLabel, const QString& newLabel)
{
    // Validate new label
    if (!validateOverlayLabel(newLabel)) {
        qDebug() << "Invalid new label for rename operation:" << newLabel;
        return false;
    }
    
    // Sanitize both labels
    QString currentSanitized = sanitizeLabel(currentLabel);
    QString newSanitized = sanitizeLabel(newLabel);
    
    // Check if the overlay exists
    auto it = d_overlays.find(currentSanitized);
    if (it == d_overlays.end()) {
        qDebug() << "Overlay not found for rename:" << currentLabel;
        return false;
    }
    
    // Check if new label already exists (different from current)
    if (currentSanitized != newSanitized && d_overlays.find(newSanitized) != d_overlays.end()) {
        qDebug() << "Overlay with new label already exists:" << newLabel;
        return false;
    }
    
    // If sanitized labels are the same, just update the overlay label and we're done
    if (currentSanitized == newSanitized) {
        it->second->setLabel(newLabel);
        saveOverlayMetadata(it->second);
        return true;
    }
    
    // Wait for any pending writes to complete before renaming
    auto writeIt = d_pendingWrites.find(currentSanitized);
    if (writeIt != d_pendingWrites.end()) {
        writeIt->second.waitForFinished();
        d_pendingWrites.erase(writeIt);
        emit pendingWritesChanged(d_pendingWrites.size());
    }
    
    // Get file paths
    QString oldDataPath = getOverlayDataPath(currentSanitized);
    QString oldSettingsPath = getOverlaySettingsPath(currentSanitized);
    QString newDataPath = getOverlayDataPath(newSanitized);
    QString newSettingsPath = getOverlaySettingsPath(newSanitized);
    
    // Attempt to rename files atomically
    bool dataRenamed = false;
    
    // Rename data file if it exists
    if (QFile::exists(oldDataPath)) {
        if (QFile::rename(oldDataPath, newDataPath)) {
            dataRenamed = true;
        } else {
            qDebug() << "Failed to rename overlay data file from" << oldDataPath << "to" << newDataPath;
            return false;
        }
    }
    
    // Rename settings file if it exists
    if (QFile::exists(oldSettingsPath)) {
        if (!QFile::rename(oldSettingsPath, newSettingsPath)) {
            qDebug() << "Failed to rename overlay settings file from" << oldSettingsPath << "to" << newSettingsPath;
            
            // Rollback data file rename if settings file rename failed
            if (dataRenamed && QFile::exists(newDataPath)) {
                QFile::rename(newDataPath, oldDataPath);
            }
            return false;
        }
    }
    
    // Update overlay label and move in storage map
    auto overlay = it->second;
    overlay->setLabel(newLabel);
    
    // Update the overlay's destFile to point to the new data file path
    overlay->setDestFile(newDataPath);
    
    // Remove from old key and add to new key
    d_overlays.erase(it);
    d_overlays.emplace(newSanitized, overlay);
    
    // Update the overlays.csv index file
    save();
    
    return true;
}

bool OverlayStorage::hasPendingWrites() const
{
    return !d_pendingWrites.empty();
}

void OverlayStorage::waitForPendingWrites()
{
    for (auto& [label, future] : d_pendingWrites) {
        future.waitForFinished();
    }
    // Clear all completed writes
    d_pendingWrites.clear();
    emit pendingWritesChanged(0);
}

int OverlayStorage::pendingWriteCount() const
{
    return d_pendingWrites.size();
}

QString OverlayStorage::getOverlayDataPath(const QString& sanitizedLabel) const
{
    using namespace BC::Key::Overlay;
    return BlackchirpCSV::exptDir(d_number, d_path).absoluteFilePath(
        overlayDir + "/" + overlayDataFile.arg(sanitizedLabel));
}

QString OverlayStorage::getOverlaySettingsPath(const QString& sanitizedLabel) const
{
    using namespace BC::Key::Overlay;
    return BlackchirpCSV::exptDir(d_number, d_path).absoluteFilePath(
        overlayDir + "/" + overlaySettingsFile.arg(sanitizedLabel));
}

void OverlayStorage::saveOverlayMetadata(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay || d_number < 1 || overlay->isPreview()) {
        return;
    }
    
    QString label = overlay->getLabel();
    QString sanitizedLabel = sanitizeLabel(label);
    
    // Check if this overlay exists in our storage
    auto it = d_overlays.find(sanitizedLabel);
    if (it == d_overlays.end()) {
        qDebug() << "Warning: Attempting to save metadata for overlay not in storage:" << label;
        return;
    }
    
    // Create overlays directory if it doesn't exist
    QDir expDir = BlackchirpCSV::exptDir(d_number, d_path);
    if (!expDir.exists(BC::Key::Overlay::overlayDir)) {
        if (!expDir.mkpath(BC::Key::Overlay::overlayDir)) {
            qDebug() << "Failed to create overlays directory for experiment" << d_number;
            return;
        }
    }
    
    // Save overlay-specific settings (metadata only)
    using namespace BC::Key::Overlay;
    std::map<QString,QVariant> overlaySettings;
    overlay->storeMetadata(overlaySettings);
    addVersionMetadata(overlaySettings);
    QString settingsFile = overlaySettingsFile.arg(sanitizedLabel);
    writeMetadata(settingsFile, overlaySettings, overlayDir);
    
    // Mark overlay as not modified since metadata is now saved
    overlay->setModified(false);
}

void OverlayStorage::onWriteCompleted(const QString& label, bool success, const QString& error)
{
    // Remove from pending writes
    auto it = d_pendingWrites.find(label);
    if (it != d_pendingWrites.end()) {
        d_pendingWrites.erase(it);
        emit pendingWritesChanged(d_pendingWrites.size());
    }
    
    // Get the overlay
    auto overlayIt = d_overlays.find(label);
    if (overlayIt == d_overlays.end()) {
        return; // Overlay was removed before write completed
    }
    
    auto overlay = overlayIt->second;
    
    if (success) {
        // Mark overlay as not modified since data write is complete
        overlay->setModified(false);
        emit overlayWriteCompleted(overlay);
    } else {
        // Remove failed overlay from storage
        d_overlays.erase(overlayIt);
        emit overlayWriteFailed(overlay, error);
    }
}
