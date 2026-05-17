#ifndef OVERLAYOPERATION_H
#define OVERLAYOPERATION_H

#include <QObject>
#include <QString>
#include <memory>
#include <atomic>

#include <data/experiment/overlaybase.h>
#include <data/experiment/overlaytypes.h>
#include <data/experiment/catalogdata.h>
#include <data/processing/parsers/genericxyparser.h>

/**
 * @brief Abstract base class for all overlay operations
 * 
 * This class defines the interface for operations that can be executed
 * in background threads by the OverlayProcessManager. Operations are
 * classified by type and priority to optimize execution order and
 * user experience.
 */
class OverlayOperation : public QObject
{
    Q_OBJECT

public:
    // Operation type classification
    enum class Type {
        Immediate,  // Fast operations (< 100ms) - labels, offsets, simple settings
        Deferred,   // Expensive operations (> 100ms) - convolution, file loading
        Atomic      // Critical operations that must complete atomically - final save
    };
    
    // Operation priority within its type
    enum class Priority {
        Low = 0,
        Normal = 1,
        High = 2,
        Critical = 3
    };

    explicit OverlayOperation(Type type, Priority priority = Priority::Normal, QObject* parent = nullptr);
    virtual ~OverlayOperation();

    // Core operation interface - must be implemented by subclasses
    virtual std::shared_ptr<OverlayBase> execute() = 0;
    virtual void cancel() = 0;
    virtual bool canCancel() const = 0;

    // Whether execute() yields an OverlayBase as its result. Operations
    // that instead carry typed payload data retrievable off the
    // operation object (e.g. file parsing) return false; for those a
    // null execute() result that did not throw is a success, not a
    // failure.
    virtual bool producesOverlay() const { return true; }
    
    // Operation metadata
    virtual QString getDescription() const = 0;
    virtual QString getOperationName() const = 0;
    
    // Type and priority accessors
    Type getType() const { return d_type; }
    Priority getPriority() const { return d_priority; }
    
    // Progress tracking
    void setProgress(int percentage, const QString& message = QString());
    int getProgress() const { return d_progress.load(); }
    QString getProgressMessage() const { return d_progressMessage; }
    
    // Cancellation support
    bool isCancelled() const { return d_cancelled.load(); }
    
    // Timeout support (0 = no timeout)
    void setTimeout(int seconds) { d_timeoutSeconds = seconds; }
    int getTimeout() const { return d_timeoutSeconds; }

signals:
    void progressChanged(int percentage, const QString& message);
    void cancellationRequested();

protected:
    // Helper methods for subclasses
    void checkCancellation(); // Throws OperationCancelledException if cancelled
    void updateProgress(int percentage, const QString& message = QString());
    
    // Cancellation flag - subclasses should check this periodically
    std::atomic<bool> d_cancelled{false};

private:
    Type d_type;
    Priority d_priority;
    std::atomic<int> d_progress{0};
    QString d_progressMessage;
    int d_timeoutSeconds{0};
};

/**
 * @brief Exception thrown when operation is cancelled
 */
class OperationCancelledException : public std::runtime_error
{
public:
    OperationCancelledException() : std::runtime_error("Operation was cancelled") {}
};

// Concrete operation implementations

/**
 * @brief Operation for creating a new overlay from scratch
 */
class CreateOverlayOperation : public OverlayOperation
{
    Q_OBJECT

public:
    CreateOverlayOperation(OverlayBase::OverlayType type,
                          const std::map<QString, QVariant, std::less<>>& settings,
                          QObject* parent = nullptr);

    std::shared_ptr<OverlayBase> execute() override;
    void cancel() override;
    bool canCancel() const override { return true; }
    QString getDescription() const override;
    QString getOperationName() const override { return "CreateOverlay"; }

private:
    OverlayBase::OverlayType d_overlayType;
    std::map<QString, QVariant, std::less<>> d_settings;
};

/**
 * @brief Operation for applying convolution to catalog overlays
 */
class ConvolutionOperation : public OverlayOperation
{
    Q_OBJECT

public:
    ConvolutionOperation(std::shared_ptr<OverlayBase> overlay,
                        bool enabled,
                        CatalogOverlay::LineshapeType lineshape,
                        double linewidthKHz,
                        double freqMinMHz,
                        double freqMaxMHz,
                        int numConvolutionPoints,
                        QObject* parent = nullptr);

    std::shared_ptr<OverlayBase> execute() override;
    void cancel() override;
    bool canCancel() const override { return true; }
    QString getDescription() const override;
    QString getOperationName() const override { return "Convolution"; }

private:
    std::shared_ptr<OverlayBase> d_overlay;
    bool d_convolutionEnabled;
    CatalogOverlay::LineshapeType d_lineshape;
    double d_linewidthKHz;
    double d_freqMinMHz;
    double d_freqMaxMHz;
    int d_numConvolutionPoints;
};

/**
 * @brief Operation for saving overlay data to disk
 */
class SaveOverlayOperation : public OverlayOperation
{
    Q_OBJECT

public:
    SaveOverlayOperation(std::shared_ptr<OverlayBase> overlay,
                        QObject* parent = nullptr);

    std::shared_ptr<OverlayBase> execute() override;
    void cancel() override;
    bool canCancel() const override { return false; } // Save operations should complete atomically
    QString getDescription() const override;
    QString getOperationName() const override { return "SaveOverlay"; }

private:
    std::shared_ptr<OverlayBase> d_overlay;
};

/**
 * @brief Operation for parsing a catalog file off the UI thread
 *
 * Carries the parsed CatalogData on the operation object; no overlay
 * is involved. A failed parse throws (surfaced as operationFailed);
 * success is signalled by a non-throwing run with the data available
 * via parsedData().
 */
class ParseCatalogOperation : public OverlayOperation
{
    Q_OBJECT

public:
    explicit ParseCatalogOperation(const QString& filePath,
                                   QObject* parent = nullptr);

    std::shared_ptr<OverlayBase> execute() override;
    void cancel() override;
    bool canCancel() const override { return true; }
    bool producesOverlay() const override { return false; }
    QString getDescription() const override;
    QString getOperationName() const override { return "ParseCatalog"; }

    // Valid after a successful run.
    const CatalogData& parsedData() const { return d_parsedData; }

private:
    QString d_filePath;
    CatalogData d_parsedData;
};

/**
 * @brief Operation for parsing a generic XY file off the UI thread
 *
 * When constructed without explicit settings the parser auto-detects
 * them; the resolved settings are available via resolvedSettings() so
 * the dialog can reflect them back into its controls. The parsed data
 * is available via parsedData(). A failed parse throws.
 */
class ParseGenericXYOperation : public OverlayOperation
{
    Q_OBJECT

public:
    // Auto-detect parse settings from the file.
    explicit ParseGenericXYOperation(const QString& filePath,
                                     QObject* parent = nullptr);
    // Parse with explicit, user-edited settings.
    ParseGenericXYOperation(const QString& filePath,
                            const GenericXYParser::ParseSettings& settings,
                            QObject* parent = nullptr);

    std::shared_ptr<OverlayBase> execute() override;
    void cancel() override;
    bool canCancel() const override { return true; }
    bool producesOverlay() const override { return false; }
    QString getDescription() const override;
    QString getOperationName() const override { return "ParseGenericXY"; }

    // Valid after a successful run.
    const GenericXYData& parsedData() const { return d_parsedData; }
    const GenericXYParser::ParseSettings& resolvedSettings() const { return d_settings; }
    bool didAutoDetect() const { return d_autoDetect; }

private:
    QString d_filePath;
    GenericXYParser::ParseSettings d_settings;
    bool d_autoDetect;
    GenericXYData d_parsedData;
};

#endif // OVERLAYOPERATION_H
