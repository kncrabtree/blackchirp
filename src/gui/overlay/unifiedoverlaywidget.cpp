#include "unifiedoverlaywidget.h"

#include <QFormLayout>
#include <QSplitter>
#include <QMessageBox>
#include <QColorDialog>
#include <QDebug>

#include "overlaybaseoptionswidget.h"
#include <gui/plot/curveappearancewidget.h>
#include "overlaytypespecificwidget.h"
#include "bcexpoverlaywidget.h"
#include "catalogoverlaywidget.h"
#include <data/storage/settingsstorage.h>
#include <gui/plot/curveappearancepresetmanager.h>
#include <gui/plot/blackchirpplotcurve.h>

UnifiedOverlayWidget::UnifiedOverlayWidget(const QString &settingsKey, Context context, QWidget *parent)
    : QWidget(parent),
      SettingsStorage(settingsKey, SettingsStorage::General),
      d_context(context),
      d_overlayType(OverlayBase::BCExperiment),
      p_mainLayout(nullptr),
      p_sourceFileConfigBox(nullptr),
      p_sourceFileConfigContent(nullptr),
      p_sourceFileSettingsBox(nullptr),
      p_sourceFileSettingsContent(nullptr),
      p_typeSpecificSettingsBox(nullptr),
      p_typeSpecificStack(nullptr),
      p_typeSpecificWidget(nullptr),
      p_overlayBaseOptionsBox(nullptr),
      p_overlayBaseOptionsWidget(nullptr),
      p_curveAppearanceBox(nullptr),
      p_curveAppearanceWidget(nullptr),
      p_progressWidget(nullptr),
      p_progressBar(nullptr),
      p_progressLabel(nullptr),
      d_sourceFileValid(false),
      d_sourceFileEnabled(false),
      d_hasBackupState(false)
{
    setupUI();
}

UnifiedOverlayWidget::~UnifiedOverlayWidget()
{
    // Ensure preview overlay is properly cleaned up to avoid dangling references
    cleanupPreviewOverlay();
}

void UnifiedOverlayWidget::setupForCreation(OverlayBase::OverlayType type,
                                           const QStringList &plotNames,
                                           const Ft &currentFt,
                                           const QVector<std::shared_ptr<OverlayBase>> &existingOverlays)
{
    d_overlayType = type;
    d_plotNames = plotNames;
    d_currentFt = currentFt;
    d_existingOverlays = existingOverlays;
    d_overlay.reset();
    p_overlayStorage.reset();
    
    setupTypeSpecificWidget();
    
    // Create and initialize overlay base options widget
    createOverlayBaseOptionsWidget();
    
    configureForContext();
}

void UnifiedOverlayWidget::setupForSettings(std::shared_ptr<OverlayBase> overlay,
                                           const QStringList &plotNames,
                                           const Ft &currentFt,
                                           std::shared_ptr<OverlayStorage> overlayStorage)
{
    
    d_overlay = overlay;
    
    d_overlayType = overlay ? overlay->type() : OverlayBase::BCExperiment;
    d_plotNames = plotNames;
    d_currentFt = currentFt;
    p_overlayStorage = overlayStorage;
    d_existingOverlays.clear();
    
    setupTypeSpecificWidget();
    
    // Create and initialize overlay base options widget
    createOverlayBaseOptionsWidget();
    
    configureForContext();
    
    // Load current overlay settings (must be after configureForContext to avoid conflicts)
    if (overlay && p_overlayBaseOptionsWidget) {
        loadOverlaySettings();
    }
    
    // Create backup of original overlay state for cancel functionality (after loading settings)
    backupOverlayState();
}

std::shared_ptr<OverlayBase> UnifiedOverlayWidget::createOverlay() const
{
    if (d_context != Context::Creation) {
        qWarning() << "createOverlay() called in settings context";
        return nullptr;
    }
    
    if (!isDataValid()) {
        return nullptr;
    }
    
    // Delegate to type-specific widget for overlay creation
    std::shared_ptr<OverlayBase> overlay;
    if (p_typeSpecificWidget) {
        overlay = p_typeSpecificWidget->createOverlay();
    }
    
    if (!overlay) {
        return nullptr;
    }
    
    // Apply base overlay options
    if (p_overlayBaseOptionsWidget) {
        p_overlayBaseOptionsWidget->applyToOverlay(overlay);
    }
    
    // Apply curve appearance settings
    if (p_curveAppearanceWidget) {
        p_curveAppearanceWidget->applyToOverlay(overlay);
    }
    
    return overlay;
}

void UnifiedOverlayWidget::applyToOverlay() const
{
    // Get the appropriate target overlay based on context
    auto targetOverlay = getCurrentTargetOverlay();
    if (!targetOverlay) {
        return; // No target overlay available
    }
    
    // Apply base overlay options
    if (p_overlayBaseOptionsWidget) {
        p_overlayBaseOptionsWidget->applyToOverlay(targetOverlay);
    }
    
    // Apply curve appearance settings
    if (p_curveAppearanceWidget) {
        p_curveAppearanceWidget->applyToOverlay(targetOverlay);
    }
    
    // Apply type-specific settings
    if (p_typeSpecificWidget) {
        p_typeSpecificWidget->applyToOverlay(targetOverlay);
    }
}

bool UnifiedOverlayWidget::validateSettings(QString &errorMessage) const
{
    QStringList errors;
    
    // Validate base overlay options
    if (p_overlayBaseOptionsWidget) {
        QString baseError;
        if (!p_overlayBaseOptionsWidget->validateSettings(baseError, d_existingOverlays)) {
            errors << baseError;
        }
    }
    
    // Validate type-specific settings
    if (p_typeSpecificWidget) {
        QString typeError;
        if (!p_typeSpecificWidget->validateSettings(typeError)) {
            errors << typeError;
        }
    }
    
    if (!errors.isEmpty()) {
        errorMessage = errors.join("\n");
        return false;
    }
    
    return true;
}

bool UnifiedOverlayWidget::isDataValid() const
{
    // Check type-specific widget data validity
    if (p_typeSpecificWidget && !p_typeSpecificWidget->isDataValid()) {
        return false;
    }
    
    // Check overlay base options validation (includes label validation)
    if (p_overlayBaseOptionsWidget) {
        QString errorMessage;
        if (!p_overlayBaseOptionsWidget->validateSettings(errorMessage, d_existingOverlays)) {
            return false;
        }
    }
    
    return true;
}

void UnifiedOverlayWidget::resetToDefaults()
{
    if (p_overlayBaseOptionsWidget) {
        // Reset base options - method to be implemented
    }
    
    if (p_curveAppearanceWidget) {
        // Reset curve appearance - method to be implemented  
    }
    
    if (p_typeSpecificWidget) {
        p_typeSpecificWidget->resetToDefaults();
    }
}

void UnifiedOverlayWidget::showProgress(const QString &message)
{
    if (!isSettingsContext()) {
        return;
    }
    
    if (p_progressLabel) {
        p_progressLabel->setText(message);
    }
    
    if (p_progressWidget) {
        p_progressWidget->show();
    }
}

void UnifiedOverlayWidget::hideProgress()
{
    if (p_progressWidget) {
        p_progressWidget->hide();
    }
}

void UnifiedOverlayWidget::updateProgress(int value, const QString &message)
{
    if (!isSettingsContext()) {
        return;
    }
    
    if (p_progressBar) {
        p_progressBar->setValue(value);
    }
    
    if (!message.isEmpty() && p_progressLabel) {
        p_progressLabel->setText(message);
    }
}

void UnifiedOverlayWidget::onSourceFileConfigToggled(bool enabled)
{
    d_sourceFileEnabled = enabled;
    updateSourceFileControls();
    
    // Note: sourceFileEnabled is contextual and should not be persisted
    
    emit settingsChanged();
}

void UnifiedOverlayWidget::onSettingsChanged()
{
    qDebug() << "onSettingsChanged: called in" << (isCreationContext() ? "creation" : "settings") << "context";
    
    // Use centralized validation logic
    performCompleteValidation();
    emit settingsChanged();
}

void UnifiedOverlayWidget::onRealTimeUpdate()
{
    // Only apply settings and emit updates if current settings are valid
    QString errorMessage;
    bool isValid = validateSettings(errorMessage);
    
    if (!isValid) {
        qDebug() << "onRealTimeUpdate: skipping overlay update due to invalid settings:" << errorMessage;
        return;
    }
    
    // Apply current settings to the appropriate overlay
    applyToOverlay();
    
    // Emit update signal with the appropriate overlay based on context
    if (isSettingsContext()) {
        // Settings context: emit for the actual overlay
        if (d_overlay) {
            qDebug() << "onRealTimeUpdate: emitting overlayDataChanged for settings overlay" << d_overlay->getLabel();
            emit overlayDataChanged(d_overlay);
        }
    } else {
        // Creation context: emit for the preview overlay
        if (d_previewOverlay) {
            qDebug() << "onRealTimeUpdate: emitting overlayDataChanged for preview overlay" << d_previewOverlay->getLabel();
            emit overlayDataChanged(d_previewOverlay);
        } else {
            qDebug() << "onRealTimeUpdate: no preview overlay to update";
        }
    }
}

void UnifiedOverlayWidget::onDataValidityChanged(bool isValid)
{
    Q_UNUSED(isValid); // Don't use this parameter - always do full validation
    
    // Use centralized validation logic - this will handle both UI updates and auto-preview
    performCompleteValidation();
}

void UnifiedOverlayWidget::onProgressOperationStarted(const QString &message)
{
    showProgress(message);
}

void UnifiedOverlayWidget::onProgressOperationFinished()
{
    hideProgress();
}

void UnifiedOverlayWidget::onProgressValueChanged(int value)
{
    updateProgress(value);
}

void UnifiedOverlayWidget::setupUI()
{
    // Create main horizontal layout
    p_mainLayout = new QHBoxLayout(this);
    p_mainLayout->setContentsMargins(6, 6, 6, 6);
    p_mainLayout->setSpacing(12);
    
    // Create left side vertical layout for overlay widgets
    auto leftVLayout = new QVBoxLayout();
    leftVLayout->setSpacing(6);
    
    // Add top spacer
    leftVLayout->addItem(new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding));
    
    // Create overlay widgets
    createSourceFileConfigBox();
    createSourceFileSettingsBox();
    createTypeSpecificSettingsBox();
    createOverlayBaseOptionsBox();
    createProgressIndicator();
    
    // Add overlay widgets to left layout
    leftVLayout->addWidget(p_sourceFileConfigBox);
    leftVLayout->addWidget(p_sourceFileSettingsBox);
    leftVLayout->addWidget(p_typeSpecificSettingsBox);
    leftVLayout->addWidget(p_overlayBaseOptionsBox);
    leftVLayout->addWidget(p_progressWidget);
    
    // Add bottom spacer
    leftVLayout->addItem(new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding));
    
    // Create right side vertical layout for curve appearance
    auto rightVLayout = new QVBoxLayout();
    rightVLayout->setSpacing(6);
    
    // Add top spacer
    rightVLayout->addItem(new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding));
    
    // Create curve appearance widget
    createCurveAppearanceBox();
    rightVLayout->addWidget(p_curveAppearanceBox);
    
    // Add bottom spacer
    rightVLayout->addItem(new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding));
    
    // Add both layouts to main horizontal layout with equal stretch
    p_mainLayout->addLayout(leftVLayout, 1);
    p_mainLayout->addLayout(rightVLayout, 1);
    
    setLayout(p_mainLayout);
    
    setupConnections();
}

void UnifiedOverlayWidget::setupConnections()
{
    // Source file config box connections
    if (p_sourceFileConfigBox) {
        connect(p_sourceFileConfigBox, &QGroupBox::toggled,
                this, &UnifiedOverlayWidget::onSourceFileConfigToggled);
    }
    
    // Base options widget connections will be added when widget is created
    // Curve appearance widget connections will be added when widget is created
    // Type-specific widget connections will be added when widget is created
}

void UnifiedOverlayWidget::createSourceFileConfigBox()
{
    p_sourceFileConfigBox = new QGroupBox("Source File Configuration", this);
    p_sourceFileConfigBox->setCheckable(false); // Will be set in configureForContext()
    
    // Content widget for source file configuration
    p_sourceFileConfigContent = new QWidget();
    auto contentLayout = new QVBoxLayout(p_sourceFileConfigContent);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    
    // Placeholder for type-specific source file controls
    auto placeholderLabel = new QLabel("Source file configuration will be populated by type-specific widget");
    placeholderLabel->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    contentLayout->addWidget(placeholderLabel);
    
    auto boxLayout = new QVBoxLayout(p_sourceFileConfigBox);
    boxLayout->addWidget(p_sourceFileConfigContent);
}

void UnifiedOverlayWidget::createSourceFileSettingsBox()
{
    p_sourceFileSettingsBox = new QGroupBox("Source File Settings", this);
    p_sourceFileSettingsBox->setCheckable(false);
    
    // Content widget for source file settings
    p_sourceFileSettingsContent = new QWidget();
    auto contentLayout = new QVBoxLayout(p_sourceFileSettingsContent);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    
    // Placeholder for type-specific source-dependent controls
    auto placeholderLabel = new QLabel("Source-dependent settings will be populated by type-specific widget");
    placeholderLabel->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    contentLayout->addWidget(placeholderLabel);
    
    auto boxLayout = new QVBoxLayout(p_sourceFileSettingsBox);
    boxLayout->addWidget(p_sourceFileSettingsContent);
}

void UnifiedOverlayWidget::createTypeSpecificSettingsBox()
{
    p_typeSpecificSettingsBox = new QGroupBox("Overlay Settings", this);
    
    // Use a stacked widget to handle different overlay types
    p_typeSpecificStack = new QStackedWidget();
    
    auto boxLayout = new QVBoxLayout(p_typeSpecificSettingsBox);
    boxLayout->addWidget(p_typeSpecificStack);
}

void UnifiedOverlayWidget::createOverlayBaseOptionsBox()
{
    p_overlayBaseOptionsBox = new QGroupBox("Base Options", this);
    
    // Start hidden - will be shown when widget is properly initialized
    p_overlayBaseOptionsBox->setVisible(false);
    
    // No layout or content initially - will be set in createOverlayBaseOptionsWidget()
}

void UnifiedOverlayWidget::createOverlayBaseOptionsWidget()
{
    if (p_overlayBaseOptionsWidget) {
        p_overlayBaseOptionsWidget->deleteLater();
    }
    
    // Create the base options widget with current parameters
    auto xRange = !d_currentFt.isEmpty() ? d_currentFt.xRange() : qMakePair(0.0, 1000.0);
    p_overlayBaseOptionsWidget = new OverlayBaseOptionsWidget(d_plotNames, xRange.first, xRange.second, this);
    
    // Set up the layout (should be the first and only layout for this groupbox)
    if (p_overlayBaseOptionsBox) {
        // Create layout only if none exists
        if (!p_overlayBaseOptionsBox->layout()) {
            auto boxLayout = new QVBoxLayout(p_overlayBaseOptionsBox);
            boxLayout->addWidget(p_overlayBaseOptionsWidget);
        } else {
            // If layout exists, just add the widget
            p_overlayBaseOptionsBox->layout()->addWidget(p_overlayBaseOptionsWidget);
        }
        
        // Show the groupbox now that it has real content
        p_overlayBaseOptionsBox->setVisible(true);
        
        // Connect signals for both contexts
        connect(p_overlayBaseOptionsWidget, &OverlayBaseOptionsWidget::settingsChanged,
                this, &UnifiedOverlayWidget::onSettingsChanged);
        connect(p_overlayBaseOptionsWidget, &OverlayBaseOptionsWidget::settingsChanged,
                this, &UnifiedOverlayWidget::onRealTimeUpdate);
        
        // Connect label changes to trigger validation (label validation is critical for overlay creation)
        connect(p_overlayBaseOptionsWidget, &OverlayBaseOptionsWidget::labelChanged,
                this, &UnifiedOverlayWidget::onSettingsChanged);
    }
}

void UnifiedOverlayWidget::loadOverlaySettings()
{

    if (!d_overlay || !p_overlayBaseOptionsWidget) {
        return;
    }
    
    // Load base overlay settings using proper getter methods
    QSignalBlocker blobk(p_overlayBaseOptionsWidget);
    p_overlayBaseOptionsWidget->setLabel(d_overlay->getLabel());
    p_overlayBaseOptionsWidget->setPlotId(d_overlay->getPlotId());
    p_overlayBaseOptionsWidget->setYScale(d_overlay->getYScale());
    p_overlayBaseOptionsWidget->setYOffset(d_overlay->getYOffset());
    p_overlayBaseOptionsWidget->setXOffset(d_overlay->getXOffset());
    p_overlayBaseOptionsWidget->setMinFreqLimit(d_overlay->getMinFreqEnabled(), d_overlay->getMinFreqValue());
    p_overlayBaseOptionsWidget->setMaxFreqLimit(d_overlay->getMaxFreqEnabled(), d_overlay->getMaxFreqValue());
    
    // Load curve appearance settings
    if (p_curveAppearanceWidget)
        p_curveAppearanceWidget->initializeFromOverlay(d_overlay);
}

void UnifiedOverlayWidget::createCurveAppearanceBox()
{
    p_curveAppearanceBox = new QGroupBox("Curve Appearance", this);
    
    // Create the curve appearance widget
    p_curveAppearanceWidget = new CurveAppearanceWidget(this);
    
    auto boxLayout = new QVBoxLayout(p_curveAppearanceBox);
    boxLayout->addWidget(p_curveAppearanceWidget);
    
    // Connect signals once during creation - no context-dependent connections needed
    connect(p_curveAppearanceWidget, &CurveAppearanceWidget::curveAppearanceChanged,
            this, &UnifiedOverlayWidget::onSettingsChanged);
    connect(p_curveAppearanceWidget, &CurveAppearanceWidget::curveAppearanceChanged,
            this, &UnifiedOverlayWidget::onRealTimeUpdate);
    connect(p_curveAppearanceWidget, &CurveAppearanceWidget::colorChangeRequested,
            this, &UnifiedOverlayWidget::onColorChangeRequested);
}


void UnifiedOverlayWidget::createProgressIndicator()
{
    p_progressWidget = new QWidget();
    p_progressWidget->setVisible(false); // Hidden by default
    
    auto progressLayout = new QHBoxLayout(p_progressWidget);
    progressLayout->setContentsMargins(0, 0, 0, 0);
    
    p_progressLabel = new QLabel("Processing...");
    p_progressBar = new QProgressBar();
    p_progressBar->setRange(0, 100);
    p_progressBar->setValue(0);
    
    progressLayout->addWidget(p_progressLabel);
    progressLayout->addWidget(p_progressBar);
}

void UnifiedOverlayWidget::configureForContext()
{
    QString contextName = getContextName();
    
    if (isCreationContext()) {
        // Creation context configuration
        p_sourceFileConfigBox->setCheckable(false);
        p_sourceFileConfigBox->setEnabled(true);
        p_sourceFileConfigBox->setTitle("Source File Selection");
        
        // Hide progress indicator in creation context
        if (p_progressWidget) {
            p_progressWidget->setVisible(false);
        }
        
    } else if (isSettingsContext()) {
        // Settings context configuration
        p_sourceFileConfigBox->setCheckable(true);
        p_sourceFileConfigBox->setTitle("Source File Configuration (Optional)");
        
        // In settings context, source file config starts disabled by default
        d_sourceFileEnabled = false;
        p_sourceFileConfigBox->setChecked(d_sourceFileEnabled);
        
        // Show progress indicator in settings context
        if (p_progressWidget) {
            p_progressWidget->setVisible(false); // Hidden until needed
        }
    }
    
    updateSourceFileControls();
    
    // Update type-specific settings box title
    if (p_typeSpecificSettingsBox) {
        QString typeName = "Overlay";
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
        p_typeSpecificSettingsBox->setTitle(QString("%1 Settings").arg(typeName));
    }
    
    // Set intelligent defaults for catalog overlays in creation mode
    if (p_curveAppearanceWidget && isCreationContext() && d_overlayType == OverlayBase::Catalog) {
        // Default to Stem - Secondary for discrete catalog data
        auto presetManager = CurveAppearancePresetManager::instance();
        if (presetManager && presetManager->hasPreset("Stem - Secondary")) {
            auto preset = presetManager->getPreset("Stem - Secondary");
            p_curveAppearanceWidget->setCurrentAppearance(preset.appearance);
        }
    }
}

void UnifiedOverlayWidget::updateSourceFileControls()
{
    bool sourceEnabled = isCreationContext() || d_sourceFileEnabled;
    
    if (p_sourceFileConfigContent) {
        p_sourceFileConfigContent->setEnabled(sourceEnabled);
    }
    
    if (p_sourceFileSettingsBox) {
        // In creation mode: settings only enabled if source file is valid
        // In settings mode: settings enabled when source file config is checked (allow access even if file moved)
        bool settingsEnabled;
        if (isCreationContext()) {
            settingsEnabled = sourceEnabled && d_sourceFileValid;
        } else {
            settingsEnabled = d_sourceFileEnabled; // Allow access in settings mode when checkbox checked
        }
        p_sourceFileSettingsBox->setEnabled(settingsEnabled);
    }
}

void UnifiedOverlayWidget::validateSourceFile()
{
    if (p_typeSpecificWidget) {
        QString errorMessage;
        d_sourceFileValid = p_typeSpecificWidget->validateSourceFile(errorMessage);
        if (!d_sourceFileValid && !errorMessage.isEmpty()) {
            d_lastValidationError = errorMessage;
        }
    } else {
        d_sourceFileValid = false;
    }
    
    updateSourceFileControls();
    
    // Use centralized validation logic to update UI consistently
    performCompleteValidation();
}

void UnifiedOverlayWidget::setupTypeSpecificWidget()
{
    clearTypeSpecificWidget();
    
    // Factory pattern: Create type-specific widget based on overlay type
    switch (d_overlayType) {
    case OverlayBase::BCExperiment:
        p_typeSpecificWidget = new BCExpOverlayWidget(d_currentFt, this);
        break;
    case OverlayBase::Catalog:
        p_typeSpecificWidget = new CatalogOverlayWidget(d_currentFt, this);
        break;
    case OverlayBase::GenericXY:
        // GenericXY not yet implemented - create placeholder
        p_typeSpecificWidget = createPlaceholderWidget("Generic XY", d_currentFt);
        break;
    }
    
    if (p_typeSpecificWidget && p_typeSpecificStack) {
        p_typeSpecificStack->addWidget(p_typeSpecificWidget);
        p_typeSpecificStack->setCurrentWidget(p_typeSpecificWidget);
        
        // Setup context for the type-specific widget
        setupTypeSpecificWidgetContext();
        
        // Setup connections for the type-specific widget
        setupTypeSpecificWidgetConnections();
        
        // Reparent UI components into three-tier architecture
        reparentTypeSpecificWidgets();
    }
}

void UnifiedOverlayWidget::clearTypeSpecificWidget()
{
    if (p_typeSpecificWidget) {
        p_typeSpecificWidget->deleteLater();
        p_typeSpecificWidget = nullptr;
    }
    
    if (p_typeSpecificStack) {
        while (p_typeSpecificStack->count() > 0) {
            QWidget *widget = p_typeSpecificStack->widget(0);
            p_typeSpecificStack->removeWidget(widget);
            widget->deleteLater();
        }
    }
}

QString UnifiedOverlayWidget::getContextName() const
{
    switch (d_context) {
    case Context::Creation:
        return "Creation";
    case Context::Settings:
        return "Settings";
    }
    return "Unknown";
}

void UnifiedOverlayWidget::performCompleteValidation()
{
    // Perform complete validation including both type-specific and overlay base options
    QString errorMessage;
    bool isValid = validateSettings(errorMessage);
    
    // Update internal state
    d_lastValidationError = errorMessage;
    
    // Emit validation status change for UI updates
    emit validationStatusChanged(isValid, errorMessage);
    
    // Handle auto-preview logic for creation context
    if (isCreationContext()) {
        if (isValid && isDataValid()) {
            qDebug() << "performCompleteValidation: calling updateAutoPreview";
            updateAutoPreview();
        } else {
            qDebug() << "performCompleteValidation: removing auto-preview due to invalid data";
            removeAutoPreview();
        }
    }
}

void UnifiedOverlayWidget::setupTypeSpecificWidgetContext()
{
    if (!p_typeSpecificWidget) {
        return;
    }
    
    // Set context information via friend access
    p_typeSpecificWidget->setContext(d_context);
    p_typeSpecificWidget->setOverlay(d_overlay);
    
    // Call appropriate setup method based on context
    if (isCreationContext()) {
        p_typeSpecificWidget->setupForCreation();
    } else if (isSettingsContext() && d_overlay) {
        p_typeSpecificWidget->setupForSettings(d_overlay);
    }
}

void UnifiedOverlayWidget::setupTypeSpecificWidgetConnections()
{
    if (!p_typeSpecificWidget) {
        return;
    }
    
    // Base connections for both contexts
    connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::settingsChanged,
            this, &UnifiedOverlayWidget::onSettingsChanged);
    connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::sourceFileChanged,
            this, &UnifiedOverlayWidget::validateSourceFile);
    connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::dataValidityChanged,
            this, [this](bool isValid) {
                Q_UNUSED(isValid); // Don't use this parameter - always do full validation
                performCompleteValidation(); // Centralized validation that updates UI
                // Auto-preview logic now handled by performCompleteValidation()
            });
    connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::labelUpdateRequested,
            this, &UnifiedOverlayWidget::onLabelUpdateRequested);
    connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::yScaleUpdateRequested,
            this, &UnifiedOverlayWidget::onYScaleUpdateRequested);
    connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::frequencyRangeUpdateRequested,
            this, &UnifiedOverlayWidget::onFrequencyRangeUpdateRequested);
    
    // Real-time update and progress indication connections for both contexts
    connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::settingsChanged,
            this, &UnifiedOverlayWidget::onRealTimeUpdate);
    
    // Progress indication connections for both contexts
    connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::progressOperationStarted,
            this, &UnifiedOverlayWidget::onProgressOperationStarted);
    connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::progressOperationFinished,
            this, &UnifiedOverlayWidget::onProgressOperationFinished);
    connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::progressValueChanged,
            this, &UnifiedOverlayWidget::onProgressValueChanged);
}

void UnifiedOverlayWidget::reparentTypeSpecificWidgets()
{
    if (!p_typeSpecificWidget) {
        return;
    }
    
    // Reparent source file config widget
    QWidget *sourceFileConfigWidget = p_typeSpecificWidget->getSourceFileConfigWidget();
    if (sourceFileConfigWidget && p_sourceFileConfigContent) {
        // Clear existing content
        QLayout *oldLayout = p_sourceFileConfigContent->layout();
        if (oldLayout) {
            while (oldLayout->count() > 0) {
                QLayoutItem *item = oldLayout->takeAt(0);
                if (item->widget()) {
                    item->widget()->deleteLater();
                }
                delete item;
            }
            delete oldLayout;
        }
        
        // Add new content
        auto newLayout = new QVBoxLayout(p_sourceFileConfigContent);
        newLayout->setContentsMargins(0, 0, 0, 0);
        sourceFileConfigWidget->setParent(p_sourceFileConfigContent);
        newLayout->addWidget(sourceFileConfigWidget);
    }
    
    // Reparent source file settings widget
    QWidget *sourceFileSettingsWidget = p_typeSpecificWidget->getSourceFileSettingsWidget();
    if (sourceFileSettingsWidget && p_sourceFileSettingsContent) {
        // Clear existing content
        QLayout *oldLayout = p_sourceFileSettingsContent->layout();
        if (oldLayout) {
            while (oldLayout->count() > 0) {
                QLayoutItem *item = oldLayout->takeAt(0);
                if (item->widget()) {
                    item->widget()->deleteLater();
                }
                delete item;
            }
            delete oldLayout;
        }
        
        // Add new content
        auto newLayout = new QVBoxLayout(p_sourceFileSettingsContent);
        newLayout->setContentsMargins(0, 0, 0, 0);
        sourceFileSettingsWidget->setParent(p_sourceFileSettingsContent);
        newLayout->addWidget(sourceFileSettingsWidget);
    }
    
    // The overlay settings widget stays in the type-specific stack
    // (it's already properly positioned)
}

OverlayTypeSpecificWidget* UnifiedOverlayWidget::createPlaceholderWidget(const QString &typeName, const Ft &currentFt)
{
    // Create a simple placeholder widget for unimplemented overlay types
    class PlaceholderWidget : public OverlayTypeSpecificWidget {
    public:
        PlaceholderWidget(const QString &name, const Ft &currentFt, QWidget *parent = nullptr) 
            : OverlayTypeSpecificWidget(currentFt, parent), d_typeName(name) {
            setupUI();
        }
        
        void setupForCreation() override {}
        void setupForSettings(std::shared_ptr<OverlayBase>) override {}
        std::shared_ptr<OverlayBase> createOverlay() const override { return nullptr; }
        void applyToOverlay(std::shared_ptr<OverlayBase>) const override {}
        bool validateSettings(QString &error) const override { 
            error = QString("%1 overlay type not yet implemented").arg(d_typeName);
            return false; 
        }
        bool isDataValid() const override { return false; }
        bool hasValidSourceFile() const override { return false; }
        QString getSourceFilePath() const override { return QString(); }
        void setSourceFilePath(const QString &) override {}
        bool validateSourceFile(QString &error) override { 
            error = QString("%1 overlay type not yet implemented").arg(d_typeName);
            return false; 
        }
        void resetToDefaults() override {}
        QHash<QString, QVariant> getSettingsHash() const override { return QHash<QString, QVariant>(); }
        constexpr QVector<OperationCapability> getSupportedOperations() const override { return {}; }
        constexpr bool supportsBackgroundOperation(OperationCapability::Type) const override { return false; }
        std::shared_ptr<OverlayOperation> createOperation(OperationCapability::Type, std::shared_ptr<OverlayBase>) const override { return nullptr; }
        QWidget* getSourceFileConfigWidget() override { return p_configWidget; }
        QWidget* getSourceFileSettingsWidget() override { return p_settingsWidget; }
        QWidget* getOverlaySettingsWidget() override { return p_overlayWidget; }
        
    protected:
        void setupUI() override {
            auto layout = new QVBoxLayout(this);
            auto label = new QLabel(QString("%1 overlay type not yet implemented").arg(d_typeName));
            label->setStyleSheet("QLabel { color: gray; font-style: italic; }");
            layout->addWidget(label);
            
            p_configWidget = new QLabel("Source file configuration not available");
            p_settingsWidget = new QLabel("Source file settings not available");
            p_overlayWidget = new QLabel("Overlay settings not available");
        }
        void setupConnections() override {}
        void loadSettings() override {}
        void saveSettings() override {}
        
    private:
        QString d_typeName;
        QWidget *p_configWidget;
        QWidget *p_settingsWidget;
        QWidget *p_overlayWidget;
    };
    
    return new PlaceholderWidget(typeName, currentFt, this);
}


void UnifiedOverlayWidget::createAutoPreview()
{
    if (!isCreationContext() || !isDataValid()) {
        return;
    }
    
    if (d_previewOverlay) {
        // Re-enable existing disabled preview and update it
        d_previewOverlay->setEnabled(true);
        applyToOverlay(); // Update with current settings
        emit previewRequested();
    } else {
        // Create new preview overlay
        d_previewOverlay = createOverlay();
        if (d_previewOverlay) {
            d_previewOverlay->setPreview(true);
            emit previewRequested();
        }
    }
}

void UnifiedOverlayWidget::updateAutoPreview()
{
    if (!isCreationContext()) {
        return;
    }
    
    if (!d_previewOverlay) {
        // No existing preview - create one
        createAutoPreview();
    }
    // Note: If preview overlay exists, onRealTimeUpdate() will handle the updates
    // This avoids duplicate applyToOverlay() calls that can cause race conditions
}

void UnifiedOverlayWidget::removeAutoPreview()
{
    if (d_previewOverlay) {
        // SAFETY: Don't destroy the overlay - just disable it to hide from plot
        d_previewOverlay->setEnabled(false);
        emit previewCancelled();
        // Keep the overlay object alive but disabled
    }
}

void UnifiedOverlayWidget::cleanupPreviewOverlay()
{
    if (d_previewOverlay) {
        // Block signals during cleanup to prevent race conditions during destruction
        QSignalBlocker blocker(this);
        
        // Disable the overlay safely
        d_previewOverlay->setEnabled(false);
        
        // Clear the reference
        d_previewOverlay.reset();
        
        // Re-enable signals and emit cleanup signal if not being destroyed
        blocker.unblock();
        if (!isBeingDestroyed()) {
            emit previewCancelled();
        }
    }
}

bool UnifiedOverlayWidget::isBeingDestroyed() const
{
    // Check if this widget is in the process of being destroyed
    // This prevents signal emission during destruction
    return signalsBlocked() || !parent() || parent()->signalsBlocked();
}

std::shared_ptr<OverlayBase> UnifiedOverlayWidget::getCurrentTargetOverlay() const
{
    if (isSettingsContext()) {
        // In settings context, always target the actual overlay
        return d_overlay;
    } else {
        // In creation context, target preview overlay if it exists, otherwise nullptr
        return d_previewOverlay;
    }
}

void UnifiedOverlayWidget::backupOverlayState()
{
    if (!isSettingsContext() || !d_overlay) {
        return;
    }
    
    // Clear any existing backup
    d_backupMetadata.clear();
    
    // Store complete overlay metadata including all settings and curve appearance
    d_overlay->storeMetadata(d_backupMetadata);
    d_hasBackupState = true;
}

void UnifiedOverlayWidget::restoreOverlayState()
{
    if (!isSettingsContext() || !d_overlay || !d_hasBackupState) {
        return;
    }
    
    // First, reload original data from destination file to discard any source file changes
    d_overlay->readFromDest();
    
    // Then restore all overlay metadata from backup
    d_overlay->retrieveMetadata(d_backupMetadata);
    
    // Mark overlay as modified to ensure changes are reflected
    d_overlay->setModified();
}

void UnifiedOverlayWidget::clearBackupState()
{
    d_backupMetadata.clear();
    d_hasBackupState = false;
}

void UnifiedOverlayWidget::onLabelUpdateRequested(const QString &newLabel)
{
    // Update the overlay base options widget with the new label (only in creation context)
    if (isCreationContext() && p_overlayBaseOptionsWidget && !newLabel.isEmpty()) {
        p_overlayBaseOptionsWidget->setLabel(newLabel);
    }
}

void UnifiedOverlayWidget::onYScaleUpdateRequested(double newYScale)
{
    // Update the overlay base options widget with the new y scale (only in creation context)
    if (isCreationContext() && p_overlayBaseOptionsWidget) {
        p_overlayBaseOptionsWidget->setYScale(newYScale);
    }
}

void UnifiedOverlayWidget::onFrequencyRangeUpdateRequested(double minFreq, double maxFreq, bool enableLimiting)
{
    // Update frequency range settings in base overlay options (only in creation context)
    if (isCreationContext() && p_overlayBaseOptionsWidget) {
        p_overlayBaseOptionsWidget->setMinFreqLimit(enableLimiting, minFreq);
        p_overlayBaseOptionsWidget->setMaxFreqLimit(enableLimiting, maxFreq);
    }
}

void UnifiedOverlayWidget::onColorChangeRequested()
{
    if (!p_curveAppearanceWidget) {
        return;
    }
    
    // Get current color - behavior depends on context
    QColor currentColor;
    
    if (isSettingsContext() && d_overlay) {
        // Settings context: get color from overlay metadata
        currentColor = d_overlay->getCurveMetadata(BC::Key::bcCurveColor).value<QColor>();
    } else {
        // Creation context: get color from widget's current appearance
        auto appearance = p_curveAppearanceWidget->getCurrentAppearance();
        currentColor = appearance.color;
    }
    
    // Fall back to default color if not valid
    if (!currentColor.isValid()) {
        currentColor = palette().color(QPalette::Text);
    }
    
    // Open color dialog
    QColor newColor = QColorDialog::getColor(currentColor, this, "Choose Curve Color");
    
    // Update widget color display if valid color chosen
    if (newColor.isValid()) {
        p_curveAppearanceWidget->updateColorDisplay(newColor);
    }
}
