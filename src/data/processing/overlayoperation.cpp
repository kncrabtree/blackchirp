#include "overlayoperation.h"
#include <data/experiment/overlaytypes.h>
#include <data/storage/overlaystorage.h>
#include <data/processing/parsers/fileparserregistry.h>
#include <data/processing/parsers/catalogparser.h>

#include <QDebug>
#include <QFileInfo>

// OverlayOperation base class implementation

OverlayOperation::OverlayOperation(Type type, Priority priority, QObject* parent)
    : QObject(parent),
      d_type(type),
      d_priority(priority)
{
}

OverlayOperation::~OverlayOperation() = default;

void OverlayOperation::setProgress(int percentage, const QString& message)
{
    updateProgress(percentage, message);
}

void OverlayOperation::checkCancellation()
{
    if (d_cancelled.load()) {
        throw OperationCancelledException();
    }
}

void OverlayOperation::updateProgress(int percentage, const QString& message)
{
    percentage = qBound(0, percentage, 100);
    d_progress.store(percentage);
    
    if (!message.isEmpty()) {
        d_progressMessage = message;
    }
    
    emit progressChanged(percentage, d_progressMessage);
}

// CreateOverlayOperation implementation

CreateOverlayOperation::CreateOverlayOperation(OverlayBase::OverlayType type,
                                             const std::map<QString, QVariant>& settings,
                                             QObject* parent)
    : OverlayOperation(Type::Deferred, Priority::Normal, parent),
      d_overlayType(type),
      d_settings(settings)
{
}

std::shared_ptr<OverlayBase> CreateOverlayOperation::execute()
{
    try {
        updateProgress(0, "Creating overlay...");
        checkCancellation();
        
        std::shared_ptr<OverlayBase> overlay;
        
        // Create overlay based on type
        switch (d_overlayType) {
        case OverlayBase::BCExperiment:
            overlay = std::make_shared<BCExpOverlay>();
            updateProgress(50, "Creating BC Experiment overlay...");
            break;
            
        case OverlayBase::Catalog:
            overlay = std::make_shared<CatalogOverlay>();
            updateProgress(50, "Creating catalog overlay...");
            break;
            
        case OverlayBase::GenericXY:
            overlay = std::make_shared<GenericXYOverlay>();
            updateProgress(50, "Creating GenericXY overlay...");
            break;
            
        default:
            throw std::runtime_error("Unknown overlay type");
        }
        
        checkCancellation();
        
        // Apply settings to overlay using public interface
        updateProgress(75, "Applying settings...");
        
        // Apply basic overlay settings using public setters
        for (const auto& [key, value] : d_settings) {
            if (key == BC::Key::Overlay::oLabel) {
                overlay->setLabel(value.toString());
            } else if (key == BC::Key::Overlay::oSourceFile) {
                overlay->setSourceFile(value.toString());
            } else if (key == BC::Key::Overlay::oPlotId) {
                overlay->setPlotId(value.toString());
            } else if (key == BC::Key::Overlay::oYScale) {
                overlay->setYScale(value.toDouble());
            } else if (key == BC::Key::Overlay::oYOffset) {
                overlay->setYOffset(value.toDouble());
            } else if (key == BC::Key::Overlay::oXOffset) {
                overlay->setXOffset(value.toDouble());
            }
            // Additional settings can be applied through type-specific methods
        }
        
        checkCancellation();
        
        // For catalog overlays, trigger convolution if enabled
        if (d_overlayType == OverlayBase::Catalog) {
            auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(overlay);
            if (catalogOverlay && catalogOverlay->convolutionEnabled()) {
                updateProgress(90, "Performing convolution...");
                // The convolution will be handled by the overlay's internal logic
                // when xyData() is called for the first time
            }
        }
        
        updateProgress(100, "Overlay created successfully");
        return overlay;
        
    } catch (const OperationCancelledException&) {
        updateProgress(0, "Operation cancelled");
        throw;
    } catch (const std::exception& e) {
        updateProgress(0, QString("Error: %1").arg(e.what()));
        throw;
    }
}

void CreateOverlayOperation::cancel()
{
    d_cancelled.store(true);
    emit cancellationRequested();
}

QString CreateOverlayOperation::getDescription() const
{
    QString typeName;
    switch (d_overlayType) {
    case OverlayBase::BCExperiment:
        typeName = "BC Experiment";
        break;
    case OverlayBase::Catalog:
        typeName = "Catalog";
        break;
    case OverlayBase::GenericXY:
        typeName = "Generic XY";
        break;
    }
    return QString("Create %1 overlay").arg(typeName);
}

// ConvolutionOperation implementation

ConvolutionOperation::ConvolutionOperation(std::shared_ptr<OverlayBase> overlay,
                                           bool enabled,
                                           CatalogOverlay::LineshapeType lineshape,
                                           double linewidthKHz,
                                           double freqMinMHz,
                                           double freqMaxMHz,
                                           int numConvolutionPoints,
                                           QObject* parent)
    : OverlayOperation(Type::Deferred, Priority::High, parent),
      d_overlay(overlay),
      d_convolutionEnabled(enabled),
      d_lineshape(lineshape),
      d_linewidthKHz(linewidthKHz),
      d_freqMinMHz(freqMinMHz),
      d_freqMaxMHz(freqMaxMHz),
      d_numConvolutionPoints(numConvolutionPoints)
{
}

std::shared_ptr<OverlayBase> ConvolutionOperation::execute()
{
    try {
        updateProgress(0, "Starting convolution...");
        checkCancellation();
        
        if (!d_overlay) {
            throw std::runtime_error("Null overlay provided for convolution");
        }
        
        auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(d_overlay);
        if (!catalogOverlay) {
            throw std::runtime_error("Convolution only supported for catalog overlays");
        }
        
        // Validate convolution parameters
        if (d_convolutionEnabled) {
            if (d_linewidthKHz < 0.0) {
                updateProgress(0, "Error: Linewidth must be positive");
                return nullptr; // Signal failure by returning null
            }
            if (d_freqMinMHz >= d_freqMaxMHz) {
                updateProgress(0, "Error: Frequency minimum must be less than maximum");
                return nullptr;
            }
            if (d_numConvolutionPoints <= 0) {
                updateProgress(0, "Error: Number of convolution points must be positive");
                return nullptr;
            }
        }
        
        updateProgress(25, "Configuring convolution settings...");
        
        // Apply convolution settings
        catalogOverlay->setConvolutionSettings(d_convolutionEnabled, d_lineshape,
                                               d_linewidthKHz, d_freqMinMHz, d_freqMaxMHz,
                                               d_numConvolutionPoints);
        
        checkCancellation();
        
        if (d_convolutionEnabled) {
            updateProgress(5, "Performing convolution...");
            
            // Set cache to pending state before triggering convolution
            catalogOverlay->setCachePending();
            
            // Create progress callback that maps chunk progress to our 5%-99% range
            auto progressCallback = [this](int chunkPercent, const QString& message) -> bool {
                // Check for cancellation
                if (d_cancelled.load()) {
                    return false; // Signal cancellation
                }
                
                // Map chunk progress to 5%-99% range
                int totalPercent = 5 + (chunkPercent * 94 / 100);
                updateProgress(totalPercent, message);
                return true; // Continue processing
            };
            
            // Generate convolved spectrum using chunked processing
            auto convolvedData = catalogOverlay->generateConvolvedSpectrum(progressCallback);
            
            // Check if operation was cancelled (empty result indicates cancellation)
            if (convolvedData.isEmpty() && d_cancelled.load()) {
                catalogOverlay->invalidateConvolutionCache();
                updateProgress(0, "Convolution cancelled");
                return catalogOverlay;
            }
            
            updateProgress(99, "Finalizing convolution...");
            checkCancellation();
            
            // Mark cache as valid with the convolved data
            catalogOverlay->setCacheValid(convolvedData);
        }
        
        updateProgress(100, "Convolution completed");
        return catalogOverlay;
        
    } catch (const OperationCancelledException&) {
        updateProgress(0, "Convolution cancelled");
        throw;
    } catch (const std::exception& e) {
        updateProgress(0, QString("Convolution error: %1").arg(e.what()));
        throw;
    }
}

void ConvolutionOperation::cancel()
{
    d_cancelled.store(true);
    emit cancellationRequested();
}

QString ConvolutionOperation::getDescription() const
{
    if (d_convolutionEnabled) {
        QString lineshapeStr = (d_lineshape == CatalogOverlay::LineshapeType::Lorentzian) ? "Lorentzian" : "Gaussian";
        return QString("Apply %1 convolution (%.3f kHz FWHM)").arg(lineshapeStr).arg(d_linewidthKHz);
    } else {
        return "Disable convolution";
    }
}

// SaveOverlayOperation implementation

SaveOverlayOperation::SaveOverlayOperation(std::shared_ptr<OverlayBase> overlay, QObject* parent)
    : OverlayOperation(Type::Atomic, Priority::High, parent),
      d_overlay(overlay)
{
}

std::shared_ptr<OverlayBase> SaveOverlayOperation::execute()
{
    try {
        updateProgress(0, "Saving overlay...");
        
        if (!d_overlay) {
            throw std::runtime_error("Null overlay provided for save operation");
        }
        
        updateProgress(50, "Writing overlay data...");
        
        // Save the overlay
        d_overlay->save();
        
        updateProgress(100, "Overlay saved successfully");
        return d_overlay;
        
    } catch (const std::exception& e) {
        updateProgress(0, QString("Save error: %1").arg(e.what()));
        throw;
    }
}

void SaveOverlayOperation::cancel()
{
    // Save operations cannot be cancelled - they must complete atomically
    qWarning() << "Attempted to cancel atomic save operation";
}

QString SaveOverlayOperation::getDescription() const
{
    return QString("Save overlay: %1").arg(d_overlay ? d_overlay->getLabel() : "Unknown");
}

// ParseCatalogOperation implementation

ParseCatalogOperation::ParseCatalogOperation(std::shared_ptr<OverlayBase> overlay,
                                            const QString& filePath,
                                            QObject* parent)
    : OverlayOperation(Type::Deferred, Priority::Normal, parent),
      d_overlay(overlay),
      d_filePath(filePath)
{
}

std::shared_ptr<OverlayBase> ParseCatalogOperation::execute()
{
    try {
        updateProgress(0, "Parsing catalog file...");
        checkCancellation();
        
        if (!d_overlay) {
            throw std::runtime_error("Null overlay provided for catalog parsing");
        }
        
        auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(d_overlay);
        if (!catalogOverlay) {
            throw std::runtime_error("Catalog parsing only supported for catalog overlays");
        }
        
        QFileInfo fileInfo(d_filePath);
        if (!fileInfo.exists()) {
            throw std::runtime_error(QString("Catalog file does not exist: %1").arg(d_filePath).toStdString());
        }
        
        updateProgress(25, QString("Reading catalog: %1").arg(fileInfo.fileName()));
        
        // Set source file path
        catalogOverlay->setSourceFile(d_filePath);
        
        checkCancellation();
        updateProgress(50, "Finding appropriate parser...");
        
        // Use type-safe parser lookup to ensure we get a CatalogParser
        auto registry = FileParserRegistry::instance();
        auto parser = registry->findParserOfType<CatalogParser>(d_filePath);
        
        if (!parser) {
            throw std::runtime_error(QString("No catalog parser found for file: %1").arg(d_filePath).toStdString());
        }
        
        checkCancellation();
        updateProgress(75, "Parsing catalog data...");
        
        // Parse the catalog file
        CatalogData catalogData = parser->parse(d_filePath);
        
        if (catalogData.isEmpty()) {
            throw std::runtime_error(QString("No data found in catalog file: %1").arg(d_filePath).toStdString());
        }
        
        checkCancellation();
        updateProgress(90, "Loading catalog data into overlay...");
        
        // Set the catalog data in the overlay
        catalogOverlay->setCatalogData(catalogData);
        
        updateProgress(100, "Catalog parsing completed successfully");
        
        return catalogOverlay;
        
    } catch (const OperationCancelledException&) {
        updateProgress(0, "Catalog parsing cancelled");
        throw;
    } catch (const std::exception& e) {
        updateProgress(0, QString("Parse error: %1").arg(e.what()));
        throw;
    }
}

void ParseCatalogOperation::cancel()
{
    d_cancelled.store(true);
    emit cancellationRequested();
}

QString ParseCatalogOperation::getDescription() const
{
    QFileInfo fileInfo(d_filePath);
    return QString("Parse catalog file: %1").arg(fileInfo.fileName());
}
