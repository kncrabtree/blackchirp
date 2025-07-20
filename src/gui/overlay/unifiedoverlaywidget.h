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

    explicit UnifiedOverlayWidget(const QString &settingsKey, 
                                 OverlayBase::OverlayType type,
                                 const QStringList &plotNames,
                                 const Ft &currentFt,
                                 std::shared_ptr<OverlayBase> overlay = nullptr,
                                 std::shared_ptr<OverlayStorage> overlayStorage = nullptr,
                                 QWidget *parent = nullptr);
    ~UnifiedOverlayWidget();


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
    // Three-tier control moved to OverlayTypeSpecificWidget base class
    void onSettingsChanged();
    void onRealTimeUpdate(); // Settings context only
    void onDataValidityChanged(bool isValid); // Auto-preview handler for creation context
    void onAccept();

private slots:
    void onLabelUpdateRequested(const QString &newLabel);
    void onColorChangeRequested();

private:
    // UI Setup
    void setupUI();
    void setupConnections();
    void createOverlayBaseOptionsBox();
    void createCurveAppearanceBox();
    void createProgressIndicator();
    
    // Settings loading
    void loadOverlaySettings();
    
    // Settings management
    void saveSettings();
    
    // Context management
    void configureForContext();
    // Three-tier validation moved to OverlayTypeSpecificWidget base class
    
    // Type-specific widget management
    void setupTypeSpecificWidget();
    void setupTypeSpecificWidgetContext();
    void setupTypeSpecificWidgetConnections();
    
    // Helper methods
    QString getContextName() const;
    bool isCreationContext() const { return d_context == Context::Creation; }
    bool isSettingsContext() const { return d_context == Context::Settings; }
    QVector<std::shared_ptr<OverlayBase>> getExistingOverlays() const;
    
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
    
    // Plot and spectroscopic data information
    QStringList d_plotNames;
    Ft d_currentFt; // Current spectroscopic data for intelligent defaults
    std::shared_ptr<OverlayBase> d_overlay; // Settings context only
    std::shared_ptr<OverlayStorage> p_overlayStorage; // Both contexts
    
    // UI Components - Three-tier architecture
    QHBoxLayout *p_mainLayout;
    
    // Type-specific tier  
    OverlayTypeSpecificWidget *p_typeSpecificWidget;
    
    // Base overlay options tier
    QGroupBox *p_overlayBaseOptionsBox;
    OverlayBaseOptionsWidget *p_overlayBaseOptionsWidget;
    
    // Curve appearance tier
    QGroupBox *p_curveAppearanceBox;
    CurveAppearanceWidget *p_curveAppearanceWidget;
    
    
    
    // State tracking - three-tier logic moved to OverlayTypeSpecificWidget
    QString d_lastValidationError;
    
    // Auto-preview state (creation context only)
    std::shared_ptr<OverlayBase> d_previewOverlay;
    
    // Backup state for cancel functionality (settings context only)
    bool d_hasBackupState;
    std::map<QString, QVariant> d_backupMetadata; // Complete overlay metadata backup
};

#endif // UNIFIEDOVERLAYWIDGET_H
