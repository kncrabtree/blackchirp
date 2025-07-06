#include "overlaystorage.h"
#include <data/experiment/overlaytypes.h>
#include <data/storage/blackchirpcsv.h>
#include <QDir>
#include <QDebug>
#include <QRegularExpression>


OverlayStorage::OverlayStorage(int number, QString path) :
    DataStorageBase(number, path)
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

bool OverlayStorage::createOverlay(OverlayBase::OverlayType t, QString sourceFile, QString label)
{
    if (d_number < 1)
        return false;
        
    // Validate and sanitize label
    if (!validateOverlayLabel(label))
        return false;
        
    QString sanitizedLabel = sanitizeLabel(label);
    
    // Check if overlay with this label already exists
    auto it = d_overlays.find(sanitizedLabel);
    if (it != d_overlays.end())
        return false;
    
    // Create overlay object
    auto overlay = createOverlayObject(t);
    if (!overlay)
        return false;
    
    // Set up overlay properties
    overlay->setLabel(sanitizedLabel);
    overlay->setSourceFile(sourceFile);
    
    // Set destination file path
    using namespace BC::Key::Overlay;
    QString overlayDataPath = BlackchirpCSV::exptDir(d_number, d_path).absoluteFilePath(
        overlayDir + "/" + overlayDataFile.arg(sanitizedLabel));
    overlay->setDestFile(overlayDataPath);
    
    // Add to internal storage
    d_overlays[sanitizedLabel] = overlay;
    
    return true;
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
    readMetadata(overlaySettingsFile.arg(fileBase), overlaySettings, overlayDir);
    
    if (overlaySettings.empty())
    {
        qDebug() << "No settings found for overlay:" << fileBase;
        return false;
    }
    
    // Set destination file path
    QString overlayDataPath = BlackchirpCSV::exptDir(d_number, d_path).absoluteFilePath(
        overlayDir + "/" + overlayDataFile.arg(fileBase));
    overlay->setDestFile(overlayDataPath);
    
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
        
        // Save overlay-specific settings
        std::map<QString,QVariant> overlaySettings;
        overlay->storeMetadata(overlaySettings);
        addVersionMetadata(overlaySettings);
        writeMetadata(overlaySettingsFile.arg(label), overlaySettings, overlayDir);
        
        // Save overlay data
        overlay->save();
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
    QString overlayDataPath = BlackchirpCSV::exptDir(d_number, d_path).absoluteFilePath(
        BC::Key::Overlay::overlayDir + "/" + QString("overlay-data-%1.csv").arg(sanitizedLabel));
    overlay->setDestFile(overlayDataPath);
    
    // Add to storage
    d_overlays[sanitizedLabel] = overlay;
    
    return true;
}

bool OverlayStorage::removeOverlay(const QString& label)
{
    auto it = d_overlays.find(label);
    if (it != d_overlays.end())
    {
        // TODO: Delete associated files from disk (data file and metadata file)
        d_overlays.erase(it);
        return true;
    }
    return false;
}