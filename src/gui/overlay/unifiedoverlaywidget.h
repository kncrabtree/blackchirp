#ifndef UNIFIEDOVERLAYWIDGET_H
#define UNIFIEDOVERLAYWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpacerItem>
#include <QGroupBox>
#include <QProgressBar>
#include <QLabel>
#include <QCheckBox>
#include <QPushButton>
#include <memory>

#include <data/experiment/overlaybase.h>
#include <data/storage/overlaystorage.h>
#include <data/storage/settingsstorage.h>
#include <data/analysis/ft.h>
#include "overlaytypespecificwidget.h"

// Forward declarations
class OverlayBaseOptionsWidget;
class CurveAppearanceWidget;

// Settings keys will be defined by subclasses

/**
 * @brief Context-aware unified widget for all overlay functionality
 * 
 * This widget integrates all overlay-related UI components into a single,
 * reusable component that adapts its behavior based on whether it's being
 * used for overlay creation or configuration. This eliminates code duplication
 * between OverlayConfigDialog and OverlaySettingsDialog while providing
 * superior user experience.
 */
class UnifiedOverlayWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    // Use the Context enum from OverlayTypeSpecificWidget
    using Context = OverlayTypeSpecificWidget::Context;

    explicit UnifiedOverlayWidget(const QString &settingsKey, Context context, QWidget *parent = nullptr);
    ~UnifiedOverlayWidget();

    // Setup methods - must be called after construction
    void setupForCreation(OverlayBase::OverlayType type, 
                         const QStringList &plotNames,
                         const Ft &currentFt,
                         const QVector<std::shared_ptr<OverlayBase>> &existingOverlays = {});
    
    void setupForSettings(std::shared_ptr<OverlayBase> overlay,
                         const QStringList &plotNames,
                         const Ft &currentFt,
                         std::shared_ptr<OverlayStorage> overlayStorage);

    // Overlay creation/modification interface
    std::shared_ptr<OverlayBase> createOverlay();
    void applyToOverlay() const; // Apply current settings to existing overlay (settings context only)
    
    // Validation
    bool validateSettings(QString &errorMessage) const;
    bool isDataValid() const;
    bool validateAcceptance(); // Returns true if dialog should proceed with acceptance
    
    // Reset functionality
    void resetToDefaults();
    
    // Auto-preview functionality (creation context only)
    std::shared_ptr<OverlayBase> getPreviewOverlay() const { return d_previewOverlay; }
    void clearPreviewOverlay() { d_previewOverlay.reset(); }
    void cleanupPreviewOverlay(); // Safe cleanup with signal blocking
    bool isBeingDestroyed() const; // Check if widget is being destroyed
    
    // Progress indication (settings context only)
    void showProgress(const QString &message);
    void hideProgress();
    void updateProgress(int value, const QString &message = QString());
    
    // State backup/restore for cancel functionality (settings context only)
    void backupOverlayState();
    void restoreOverlayState();
    void clearBackupState();
    QString getOriginalLabel() const; // Get original label from backup metadata
    
    // Type-specific widget access
    OverlayTypeSpecificWidget* getTypeSpecificWidget() const { return p_typeSpecificWidget; }

signals:
    void settingsChanged(); // Emitted when any setting changes
    void overlayDataChanged(std::shared_ptr<OverlayBase> overlay); // Real-time overlay updates (settings context)
    void validationStatusChanged(bool isValid, const QString &message);
    
    // Auto-preview signals (creation context only)
    void previewRequested();
    void previewCancelled();

public slots:
    void onSourceFileConfigToggled(bool enabled); // For checkable source file config box
    void onSettingsChanged();
    void onRealTimeUpdate(); // Settings context only
    void onDataValidityChanged(bool isValid); // Auto-preview handler for creation context
    void onAccept();

private slots:
    void onProgressOperationStarted(const QString &message);
    void onProgressOperationFinished();
    void onProgressValueChanged(int value);
    void onLabelUpdateRequested(const QString &newLabel);
    void onColorChangeRequested();

private:
    // UI Setup
    void setupUI();
    void setupConnections();
    void createSourceFileConfigBox();
    void createSourceFileSettingsBox();
    void createTypeSpecificSettingsBox();
    void createOverlayBaseOptionsBox();
    void createOverlayBaseOptionsWidget();
    void createCurveAppearanceBox();
    void createProgressIndicator();
    
    // Settings loading
    void loadOverlaySettings();
    
    // Settings management
    void saveSettings();
    
    // Context management
    void configureForContext();
    void updateSourceFileControls();
    void validateSourceFile();
    
    // Type-specific widget management
    void setupTypeSpecificWidget();
    void setupTypeSpecificWidgetContext();
    void setupTypeSpecificWidgetConnections();
    void reparentTypeSpecificWidgets();
    OverlayTypeSpecificWidget* createPlaceholderWidget(const QString &typeName, const Ft &currentFt);
    
    // Helper methods
    QString getContextName() const;
    bool isCreationContext() const { return d_context == Context::Creation; }
    bool isSettingsContext() const { return d_context == Context::Settings; }
    
    // Centralized validation
    void performCompleteValidation();
    
    // Auto-preview management (creation context only)
    void createAutoPreview();
    void updateAutoPreview();
    void removeAutoPreview();
    std::shared_ptr<OverlayBase> getCurrentTargetOverlay() const;
    
    // Context and state (immutable after construction)
    const Context d_context;
    OverlayBase::OverlayType d_overlayType;
    std::shared_ptr<OverlayBase> d_overlay; // Settings context only
    std::shared_ptr<OverlayStorage> p_overlayStorage; // Settings context only
    
    // Plot and spectroscopic data information
    QStringList d_plotNames;
    Ft d_currentFt; // Current spectroscopic data for intelligent defaults
    QVector<std::shared_ptr<OverlayBase>> d_existingOverlays; // Creation context only
    
    // UI Components - Three-tier architecture
    QHBoxLayout *p_mainLayout;
    
    // Source file tier
    QGroupBox *p_sourceFileConfigBox;
    QWidget *p_sourceFileConfigContent;
    
    QGroupBox *p_sourceFileSettingsBox;
    QWidget *p_sourceFileSettingsContent;
    
    // Type-specific tier  
    QGroupBox *p_typeSpecificSettingsBox;
    OverlayTypeSpecificWidget *p_typeSpecificWidget;
    
    // Base overlay options tier
    QGroupBox *p_overlayBaseOptionsBox;
    OverlayBaseOptionsWidget *p_overlayBaseOptionsWidget;
    
    // Curve appearance tier
    QGroupBox *p_curveAppearanceBox;
    CurveAppearanceWidget *p_curveAppearanceWidget;
    
    
    // Progress indication
    QWidget *p_progressWidget;
    QProgressBar *p_progressBar;
    QLabel *p_progressLabel;
    
    // State tracking
    bool d_sourceFileValid;
    bool d_sourceFileEnabled; // Settings context only
    QString d_lastValidationError;
    
    // Auto-preview state (creation context only)
    std::shared_ptr<OverlayBase> d_previewOverlay;
    
    // Backup state for cancel functionality (settings context only)
    bool d_hasBackupState;
    std::map<QString, QVariant> d_backupMetadata; // Complete overlay metadata backup
};

#endif // UNIFIEDOVERLAYWIDGET_H
