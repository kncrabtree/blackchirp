#ifndef OVERLAYTYPESPECIFICWIDGET_H
#define OVERLAYTYPESPECIFICWIDGET_H

#include <QWidget>
#include <QString>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QList>
#include <memory>

#include <data/experiment/overlaybase.h>
#include <data/processing/overlayprocessmanager.h>
#include <data/analysis/ft.h>

// Forward declarations
class OverlayOperation;
class SettingsTable;
class QCheckBox;

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
    bool validateSettings(); // Non-virtual - manages base class state
    QString getSettingsErrorMessage() const { return d_settingsErrorMessage; }
    virtual bool isDataValid() const = 0;
    
    // Source file management
    virtual bool hasValidSourceFile() const = 0;
    virtual QString getSourceFilePath() const = 0;
    virtual void setSourceFilePath(const QString &path) = 0;
    bool validateSourceFile(); // Non-virtual - manages base class state
    QString getSourceFileErrorMessage() const { return d_sourceFileErrorMessage; }
    
    // Accept functionality
    virtual void onAccept() { saveSettings(); }
    
    // Settings state capture for preview sync tracking
    virtual QHash<QString, QVariant> getSettingsHash() const = 0;
    
    // Operation declaration interface
    virtual constexpr QVector<OperationCapability> getSupportedOperations() const = 0;
    virtual constexpr bool supportsBackgroundOperation(OperationCapability::Type type) const = 0;
    virtual std::shared_ptr<OverlayOperation> createOperation(OperationCapability::Type type,
                                                             std::shared_ptr<OverlayBase> overlay = nullptr) const = 0;
    
    // UI setup - must be called after construction since it calls virtual methods
    void setupUI();
    
    // Context-aware UI behavior
    void configureForContext(); // Non-virtual - orchestrates context setup
    virtual void configureForCreationContext() {} // Override to customize creation UI
    virtual void configureForSettingsContext() {} // Override to customize settings UI
    
    // Three-tier state management (moved from UnifiedOverlayWidget)
    void updateSourceFileControls(); // Update source file UI state - controls base class regions
    void onSourceFileConfigToggled(bool enabled); // Handle source config changes

    // Source-file-config abstraction. The Creation/Settings state
    // machine talks to these instead of the underlying control so the
    // section row that replaced the QGroupBox can be repointed without
    // touching the state-machine logic.
    bool isSourceConfigEnabled() const;
    void setSourceConfigChecked(bool checked); // signal-blocked, no window grow
    void setSourceConfigCheckable(bool checkable);
    void setSourceConfigTitle(const QString &title);
    
    // Helper method for compact file path display with tooltips
    void updatePathDisplayAndTooltip(QLineEdit* lineEdit, const QString &fullPath);
    
    // Access to full source file path (separate from potentially abbreviated display)
    QString getStoredFullSourceFilePath() const { return d_fullSourceFilePath; }
    
    // Visual appearance configuration
    void configureGroupBoxAppearance(QGroupBox* groupBox);
    
    // Validation state getters
    bool getSourceFileValid() const { return d_sourceFileValid; }
    bool getSourceFileEnabled() const { return d_sourceFileEnabled; }
    bool getSettingsValid() const { return d_settingsValid; }
    
protected:
    // Derived class validation interface
    virtual bool validateSourceFileImpl() = 0;
    virtual bool validateSettingsImpl() = 0;
    
    // Error message management for derived classes
    void setSourceFileErrorMessage(const QString &message) { d_sourceFileErrorMessage = message; }
    void setSettingsErrorMessage(const QString &message) { d_settingsErrorMessage = message; }
    
    // Type-specific widgets now handle their own layout internally
    
    // Validation for unsaved changes
    virtual bool hasUnsavedChanges() const { return false; } // Default implementation
    virtual bool validateAcceptance() { return true; } // Default implementation - returns true if dialog should proceed
    
    // Type-specific settings visibility
    virtual bool hasTypeSpecificSettings() const { return true; } // Default implementation - show settings by default

signals:
    void settingsChanged();
    void sourceConfigToggled(bool enabled); // relayed to onSourceFileConfigToggled
    void dataValidityChanged(bool isValid);
    void progressOperationStarted(const QString &message);
    void progressOperationFinished();
    void progressValueChanged(int value);
    void labelUpdateRequested(const QString &newLabel);

protected:
    // Three-tier UI creation interface - pure virtual methods for derived classes.
    // The source-file-config tier fills a SettingsTable; the base has
    // already added the checkable "Source File Configuration" section
    // row at the top, so the subclass only appends its file-selection /
    // status rows (and any dynamic detail rows it manages itself).
    virtual void createSourceFileConfigUI(SettingsTable *table) = 0;
    virtual void createSourceFileSettingsUI(QGroupBox *parent) = 0;
    virtual void createTypeSpecificSettingsUI(QGroupBox *parent) = 0;

    // Re-assert subclass-managed dynamic row visibility after the base
    // applies context state (e.g. catalog's parsed-file detail rows,
    // which are hidden until a file is loaded). Default: no-op.
    virtual void refreshSourceFileConfigState() {}
    
    // Helper methods for derived classes
    virtual void setupConnections() = 0;
    virtual void loadSettings() = 0;
    virtual void saveSettings() = 0;
    
    // Context information (set by UnifiedOverlayWidget)
    enum class Context {
        Creation,
        Settings
    };
    
    // Context query methods
    bool isCreationContext() const { return d_context == Context::Creation; }
    bool isSettingsContext() const { return d_context == Context::Settings; }
    Context getContext() const { return d_context; }
    
    Context d_context;
    std::shared_ptr<OverlayBase> d_overlay; // Only valid in settings context
    const Ft d_currentFt; // Current spectroscopic data for intelligent defaults and analysis
    
    // Source-file-config tier: a flat SettingsTable whose first row is
    // the checkable "Source File Configuration" section. The remaining
    // two tiers stay flat QGroupBoxes (only the config box is being
    // converted).
    SettingsTable *p_sourceFileConfigTable;
    QCheckBox *p_sourceConfigBox;       // section checkbox; alive across modes
    int d_sourceConfigSection;          // section row index
    QList<int> d_sourceConfigRows;      // subclass rows bound to the section
    QGroupBox *p_sourceFileSettingsBox;
    QGroupBox *p_overlaySettingsBox;
    
    // Three-tier state management (moved from UnifiedOverlayWidget)
    bool d_sourceFileValid = false;
    bool d_sourceFileEnabled = true;
    bool d_settingsValid = false;
    QString d_sourceFileErrorMessage;
    QString d_settingsErrorMessage;
    
    // Full path storage (separate from potentially abbreviated display text)
    QString d_fullSourceFilePath;
    
    friend class UnifiedOverlayWidget;
    
private:
    void setContext(Context context) { d_context = context; }
    void setOverlay(std::shared_ptr<OverlayBase> overlay) { 
        d_overlay = overlay; 
        // In settings context, start with source file configuration disabled
        // User must explicitly enable it to change the source file
        if (isSettingsContext()) {
            d_sourceFileEnabled = false;
        }
    }
};

#endif // OVERLAYTYPESPECIFICWIDGET_H
