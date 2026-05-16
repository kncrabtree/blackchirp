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

    // Construction and destruction
    explicit UnifiedOverlayWidget(const QString &settingsKey, 
                                 OverlayBase::OverlayType type,
                                 const QStringList &plotNames,
                                 const Ft &currentFt,
                                 std::shared_ptr<OverlayBase> overlay = nullptr,
                                 std::shared_ptr<OverlayStorage> overlayStorage = nullptr,
                                 QWidget *parent = nullptr);
    ~UnifiedOverlayWidget();

    // Core overlay interface
    std::shared_ptr<OverlayBase> createOverlay();
    void applyToOverlay() const; // Apply current settings to existing overlay (settings context only)
    
    // Validation interface
    bool validateSettings(QString &errorMessage) const;
    bool isDataValid() const;
    bool validateAcceptance(); // Returns true if dialog should proceed with acceptance
    
    // Auto-preview interface (creation context only)
    std::shared_ptr<OverlayBase> getPreviewOverlay() const { return d_previewOverlay; }
    void clearPreviewOverlay() { d_previewOverlay.reset(); }
    void cleanupPreviewOverlay(); // Safe cleanup with signal blocking
    bool isBeingDestroyed() const; // Check if widget is being destroyed
    
    // State management interface (settings context only)
    void backupOverlayState();
    void restoreOverlayState();
    QString getOriginalLabel() const; // Get original label from backup metadata
    

signals:
    void overlayDataChanged(std::shared_ptr<OverlayBase> overlay); // Real-time overlay updates (settings context)
    void validationStatusChanged(bool isValid, const QString &message);
    
    // Auto-preview signals (creation context only)
    void previewRequested();
    void previewCancelled();

public slots:
    // Three-tier control moved to OverlayTypeSpecificWidget base class
    void onRealTimeUpdate(); // Settings context only
    void onDataValidityChanged(); // Auto-preview handler for creation context
    void onAccept();

private slots:
    void onLabelUpdateRequested(const QString &newLabel);
    void onColorChangeRequested();

private:
    // UI setup methods
    void setupUI();
    void setupConnections();
    void createOverlayBaseOptionsBox();
    void createCurveAppearanceBox();
    
    // Type-specific widget management
    void setupTypeSpecificWidget();
    void setupTypeSpecificWidgetContext();
    void setupTypeSpecificWidgetConnections();
    
    // Settings management
    void loadOverlaySettings();
    void saveSettings();
    void configureForContext();
    
    // Validation helpers
    void performCompleteValidation();
    QVector<std::shared_ptr<OverlayBase>> getExistingOverlays() const;
    
    // Auto-preview management (creation context only)
    void createAutoPreview();
    void updateAutoPreview();
    void removeAutoPreview();
    std::shared_ptr<OverlayBase> getCurrentTargetOverlay() const;
    
    // Context helpers
    bool isCreationContext() const { return d_context == Context::Creation; }
    bool isSettingsContext() const { return d_context == Context::Settings; }
    
    // Context and configuration (immutable after construction)
    const Context d_context;
    OverlayBase::OverlayType d_overlayType;
    QStringList d_plotNames;
    Ft d_currentFt; // Current spectroscopic data for intelligent defaults
    std::shared_ptr<OverlayBase> d_overlay; // Settings context only
    std::shared_ptr<OverlayStorage> p_overlayStorage; // Both contexts
    
    // UI Components - Three-tier architecture
    QHBoxLayout *p_mainLayout;
    OverlayTypeSpecificWidget *p_typeSpecificWidget; // Type-specific tier
    QGroupBox *p_typeSpecificBox; // Titled wrapper, matches the other two panels
    QGroupBox *p_overlayBaseOptionsBox; // Base overlay options tier
    OverlayBaseOptionsWidget *p_overlayBaseOptionsWidget;
    QGroupBox *p_curveAppearanceBox; // Curve appearance tier
    CurveAppearanceWidget *p_curveAppearanceWidget;
    
    // State tracking
    QString d_lastValidationError;
    
    // Auto-preview state (creation context only)
    std::shared_ptr<OverlayBase> d_previewOverlay;
    
    // Backup state for cancel functionality (settings context only)
    bool d_hasBackupState;
    std::map<QString, QVariant, std::less<>> d_backupMetadata; // Complete overlay metadata backup
};

#endif // UNIFIEDOVERLAYWIDGET_H
