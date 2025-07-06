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
    case OverlayBase::SPCAT:
        // TODO: Implement when SPCAT overlay type is available
        qDebug() << "SPCAT overlay type not yet implemented";
        return nullptr;
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
    if (!overlay || d_number < 1)
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
    
    // Set destination file path for the overlay data
    overlay->setDestFile(getOverlayDataPath(sanitizedLabel));
    
    // Add to storage
    d_overlays[sanitizedLabel] = overlay;
    
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
        
        d_overlays.erase(it);
        return true;
    }
    return false;
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
    const_cast<OverlayStorage*>(this)->d_pendingWrites.clear();
    const_cast<OverlayStorage*>(this)->emit pendingWritesChanged(0);
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
