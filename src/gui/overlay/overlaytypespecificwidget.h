#ifndef OVERLAYTYPESPECIFICWIDGET_H
#define OVERLAYTYPESPECIFICWIDGET_H

#include <QWidget>
#include <QString>
#include <memory>

#include <data/experiment/overlaybase.h>
#include <data/processing/overlayprocessmanager.h>
#include <data/analysis/ft.h>

// Forward declarations
class OverlayOperation;

/**
 * @brief Operation capability metadata for overlay widgets
 */
struct OperationCapability {
    enum Type {
        Creation,        // Creating overlay from scratch
        Convolution,     // Applying convolution to existing overlay
        Validation,      // Validating settings/source files
        PreviewUpdate    // Updating preview with new settings
    };
    
    Type type;
    bool isExpensive;              // Whether operation should be background-processed
    int estimatedDurationMs;       // Rough estimate for progress indication
    QString description;           // Human-readable operation description
    OverlayProcessManager::Priority priority; // Default priority for operation
    
    OperationCapability(Type t, bool expensive = false, int durationMs = 0, 
                       const QString &desc = QString(),
                       OverlayProcessManager::Priority prio = OverlayProcessManager::Priority::Normal)
        : type(t), isExpensive(expensive), estimatedDurationMs(durationMs), 
          description(desc), priority(prio) {}
};

/**
 * @brief Abstract base class for type-specific overlay widgets
 * 
 * This class defines the interface for type-specific overlay configuration
 * widgets that integrate with the UnifiedOverlayWidget's three-tier architecture.
 * Each overlay type (BCExperiment, Catalog, GenericXY) should inherit from this
 * class and implement the type-specific functionality.
 */
class OverlayTypeSpecificWidget : public QWidget
{
    Q_OBJECT

public:
    explicit OverlayTypeSpecificWidget(const Ft &currentFt, QWidget *parent = nullptr);
    virtual ~OverlayTypeSpecificWidget() = default;

    // Setup methods for different contexts
    virtual void setupForCreation() = 0;
    virtual void setupForSettings(std::shared_ptr<OverlayBase> overlay) = 0;
        
    // Overlay creation and modification interface
    virtual std::shared_ptr<OverlayBase> createOverlay() = 0;
    virtual void applyToOverlay(std::shared_ptr<OverlayBase> overlay) const = 0;
    
    // Validation
    virtual bool validateSettings(QString &errorMessage) const = 0;
    virtual bool isDataValid() const = 0;
    
    // Source file management
    virtual bool hasValidSourceFile() const = 0;
    virtual QString getSourceFilePath() const = 0;
    virtual void setSourceFilePath(const QString &path) = 0;
    virtual bool validateSourceFile(QString &errorMessage) = 0;
    
    // Reset/accept functionality
    virtual void resetToDefaults() = 0;
    virtual void onAccept() { saveSettings(); }
    
    // Settings state capture for preview sync tracking
    virtual QHash<QString, QVariant> getSettingsHash() const = 0;
    
    // Operation declaration interface
    virtual constexpr QVector<OperationCapability> getSupportedOperations() const = 0;
    virtual constexpr bool supportsBackgroundOperation(OperationCapability::Type type) const = 0;
    virtual std::shared_ptr<OverlayOperation> createOperation(OperationCapability::Type type,
                                                             std::shared_ptr<OverlayBase> overlay = nullptr) const = 0;
    
    // UI component access for three-tier architecture
    virtual QWidget* getSourceFileConfigWidget() = 0;      // File selection, metadata display
    virtual QWidget* getSourceFileSettingsWidget() = 0;    // Source-dependent controls
    virtual QWidget* getOverlaySettingsWidget() = 0;       // Source-independent controls
    
    // Validation for unsaved changes
    virtual bool hasUnsavedChanges() const { return false; } // Default implementation
    virtual bool validateAcceptance() { return true; } // Default implementation - returns true if dialog should proceed

signals:
    void settingsChanged();
    void sourceFileChanged();
    void dataValidityChanged(bool isValid);
    void progressOperationStarted(const QString &message);
    void progressOperationFinished();
    void progressValueChanged(int value);
    void labelUpdateRequested(const QString &newLabel);

protected:
    // Helper methods for derived classes
    virtual void setupUI() = 0;
    virtual void setupConnections() = 0;
    virtual void loadSettings() = 0;
    virtual void saveSettings() = 0;
    
    // Context information (set by UnifiedOverlayWidget)
    enum class Context {
        Creation,
        Settings
    };
    
    Context d_context;
    std::shared_ptr<OverlayBase> d_overlay; // Only valid in settings context
    const Ft d_currentFt; // Current spectroscopic data for intelligent defaults and analysis
    
    friend class UnifiedOverlayWidget;
    
private:
    void setContext(Context context) { d_context = context; }
    void setOverlay(std::shared_ptr<OverlayBase> overlay) { d_overlay = overlay; }
};

#endif // OVERLAYTYPESPECIFICWIDGET_H
