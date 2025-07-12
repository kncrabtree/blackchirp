#include "unifiedoverlaywidget.h"

#include <QFormLayout>
#include <QSplitter>
#include <QMessageBox>
#include <QDebug>

#include "overlaybaseoptionswidget.h"
#include <gui/plot/curveappearancewidget.h>
#include "overlaytypespecificwidget.h"
#include "bcexpoverlaywidget.h"
#include "catalogoverlaywidget.h"
#include <data/storage/settingsstorage.h>

UnifiedOverlayWidget::UnifiedOverlayWidget(const QString &settingsKey, QWidget *parent)
    : QWidget(parent),
      SettingsStorage(settingsKey, SettingsStorage::General),
      d_context(Context::Creation),
      d_overlayType(OverlayBase::BCExperiment),
      d_xRangeMin(0.0),
      d_xRangeMax(1000.0),
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
      p_previewButton(nullptr),
      p_progressWidget(nullptr),
      p_progressBar(nullptr),
      p_progressLabel(nullptr),
      d_sourceFileValid(false),
      d_sourceFileEnabled(false),
      d_inPreviewMode(false),
      d_previewSyncValid(false)
{
    setupUI();
}

UnifiedOverlayWidget::~UnifiedOverlayWidget() = default;

void UnifiedOverlayWidget::setupForCreation(OverlayBase::OverlayType type,
                                           const QStringList &plotNames,
                                           double xRangeMin, double xRangeMax,
                                           const QVector<std::shared_ptr<OverlayBase>> &existingOverlays)
{
    d_context = Context::Creation;
    d_overlayType = type;
    d_plotNames = plotNames;
    d_xRangeMin = xRangeMin;
    d_xRangeMax = xRangeMax;
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
                                           double xRangeMin, double xRangeMax,
                                           std::shared_ptr<OverlayStorage> overlayStorage)
{
    d_context = Context::Settings;
    d_overlay = overlay;
    d_overlayType = overlay ? overlay->type() : OverlayBase::BCExperiment;
    d_plotNames = plotNames;
    d_xRangeMin = xRangeMin;
    d_xRangeMax = xRangeMax;
    p_overlayStorage = overlayStorage;
    d_existingOverlays.clear();
    
    setupTypeSpecificWidget();
    
    // Create and initialize overlay base options widget
    createOverlayBaseOptionsWidget();
    
    configureForContext();
    
    // Load current overlay settings
    if (overlay && p_overlayBaseOptionsWidget) {
        // Load existing overlay data - will be implemented
        loadOverlaySettings();
    }
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
    if (d_context != Context::Settings || !d_overlay) {
        qWarning() << "applyToOverlay() called in invalid context";
        return;
    }
    
    // Apply base overlay options
    if (p_overlayBaseOptionsWidget) {
        p_overlayBaseOptionsWidget->applyToOverlay(d_overlay);
    }
    
    // Apply curve appearance settings
    if (p_curveAppearanceWidget) {
        p_curveAppearanceWidget->applyToOverlay(d_overlay);
    }
    
    // Apply type-specific settings
    if (p_typeSpecificWidget) {
        p_typeSpecificWidget->applyToOverlay(d_overlay);
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
    if (p_typeSpecificWidget && !p_typeSpecificWidget->isDataValid()) {
        return false;
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
    // Validate current state
    QString errorMessage;
    bool isValid = validateSettings(errorMessage);
    
    d_lastValidationError = errorMessage;
    emit validationStatusChanged(isValid, errorMessage);
    emit settingsChanged();
    
    // Invalidate preview sync when settings change
    if (d_inPreviewMode) {
        invalidatePreviewSync();
    }
    
    // Update preview button state based on data validity (only in creation context)
    if (isCreationContext() && p_previewButton) {
        p_previewButton->setEnabled(isDataValid());
    }
}

void UnifiedOverlayWidget::onRealTimeUpdate()
{
    if (!isSettingsContext()) {
        return;
    }
    
    // Apply current settings to overlay and emit update signal
    applyToOverlay();
    emit overlayDataChanged(d_overlay);
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
    createPreviewButton();
    createProgressIndicator();
    
    // Add overlay widgets to left layout
    leftVLayout->addWidget(p_sourceFileConfigBox);
    leftVLayout->addWidget(p_sourceFileSettingsBox);
    leftVLayout->addWidget(p_typeSpecificSettingsBox);
    leftVLayout->addWidget(p_overlayBaseOptionsBox);
    leftVLayout->addWidget(p_previewButton);
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
    
    // Placeholder layout - will be replaced when widget is created
    auto boxLayout = new QVBoxLayout(p_overlayBaseOptionsBox);
    
    auto placeholderLabel = new QLabel("Base overlay options will be initialized when context is set");
    placeholderLabel->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    boxLayout->addWidget(placeholderLabel);
}

void UnifiedOverlayWidget::createOverlayBaseOptionsWidget()
{
    if (p_overlayBaseOptionsWidget) {
        p_overlayBaseOptionsWidget->deleteLater();
    }
    
    // Create the base options widget with current parameters
    p_overlayBaseOptionsWidget = new OverlayBaseOptionsWidget(d_plotNames, d_xRangeMin, d_xRangeMax, this);
    
    // Replace the placeholder content
    if (p_overlayBaseOptionsBox) {
        // Clear existing layout
        QLayout* oldLayout = p_overlayBaseOptionsBox->layout();
        if (oldLayout) {
            while (oldLayout->count() > 0) {
                QLayoutItem* item = oldLayout->takeAt(0);
                if (item->widget()) {
                    item->widget()->deleteLater();
                }
                delete item;
            }
            delete oldLayout;
        }
        
        // Create new layout with the actual widget
        auto boxLayout = new QVBoxLayout(p_overlayBaseOptionsBox);
        boxLayout->addWidget(p_overlayBaseOptionsWidget);
        
        // Connect signals
        connect(p_overlayBaseOptionsWidget, &OverlayBaseOptionsWidget::settingsChanged,
                this, &UnifiedOverlayWidget::onSettingsChanged);
        
        if (isSettingsContext()) {
            connect(p_overlayBaseOptionsWidget, &OverlayBaseOptionsWidget::settingsChanged,
                    this, &UnifiedOverlayWidget::onRealTimeUpdate);
        }
    }
}

void UnifiedOverlayWidget::loadOverlaySettings()
{
    if (!d_overlay || !p_overlayBaseOptionsWidget) {
        return;
    }
    
    // Load base overlay settings using proper getter methods
    p_overlayBaseOptionsWidget->setLabel(d_overlay->getLabel());
    p_overlayBaseOptionsWidget->setPlotId(d_overlay->getPlotId());
    p_overlayBaseOptionsWidget->setYScale(d_overlay->getYScale());
    p_overlayBaseOptionsWidget->setYOffset(d_overlay->getYOffset());
    p_overlayBaseOptionsWidget->setXOffset(d_overlay->getXOffset());
    p_overlayBaseOptionsWidget->setMinFreqLimit(d_overlay->getMinFreqEnabled(), d_overlay->getMinFreqValue());
    p_overlayBaseOptionsWidget->setMaxFreqLimit(d_overlay->getMaxFreqEnabled(), d_overlay->getMaxFreqValue());
    
    // Load curve appearance settings
    if (p_curveAppearanceWidget) {
        p_curveAppearanceWidget->initializeFromOverlay(d_overlay);
    }
}

void UnifiedOverlayWidget::createCurveAppearanceBox()
{
    p_curveAppearanceBox = new QGroupBox("Curve Appearance", this);
    
    // Create the curve appearance widget
    p_curveAppearanceWidget = new CurveAppearanceWidget(this);
    
    auto boxLayout = new QVBoxLayout(p_curveAppearanceBox);
    boxLayout->addWidget(p_curveAppearanceWidget);
    
    // Connect curve appearance signals (will be connected in configureForContext)
    // when we know the context
}

void UnifiedOverlayWidget::createPreviewButton()
{
    p_previewButton = new QPushButton("Preview", this);
    p_previewButton->setEnabled(false); // Will be enabled when data is valid
    p_previewButton->setToolTip("Preview overlay with current settings");
    
    // Style the button to make it prominent
    p_previewButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #4CAF50;"
        "    color: white;"
        "    border: none;"
        "    padding: 8px 16px;"
        "    font-weight: bold;"
        "    border-radius: 4px;"
        "}"
        "QPushButton:hover {"
        "    background-color: #45a049;"
        "}"
        "QPushButton:disabled {"
        "    background-color: #cccccc;"
        "    color: #666666;"
        "}"
    );
    
    connect(p_previewButton, &QPushButton::clicked, this, &UnifiedOverlayWidget::onPreviewToggled);
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
    
    // Setup context-aware connections for curve appearance widget
    if (p_curveAppearanceWidget) {
        // Disconnect any existing connections first
        disconnect(p_curveAppearanceWidget, &CurveAppearanceWidget::curveAppearanceChanged,
                   this, &UnifiedOverlayWidget::onSettingsChanged);
        disconnect(p_curveAppearanceWidget, &CurveAppearanceWidget::curveAppearanceChanged,
                   this, &UnifiedOverlayWidget::onRealTimeUpdate);
        
        // Connect base signal for both contexts
        connect(p_curveAppearanceWidget, &CurveAppearanceWidget::curveAppearanceChanged,
                this, &UnifiedOverlayWidget::onSettingsChanged);
        
        // Connect real-time update for settings context only
        if (isSettingsContext()) {
            connect(p_curveAppearanceWidget, &CurveAppearanceWidget::curveAppearanceChanged,
                    this, &UnifiedOverlayWidget::onRealTimeUpdate);
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
        // Source file settings are only enabled if source file is valid and accessible
        bool settingsEnabled = sourceEnabled && d_sourceFileValid;
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
    
    // Emit validation status change
    emit validationStatusChanged(d_sourceFileValid, d_lastValidationError);
}

void UnifiedOverlayWidget::setupTypeSpecificWidget()
{
    clearTypeSpecificWidget();
    
    // Factory pattern: Create type-specific widget based on overlay type
    switch (d_overlayType) {
    case OverlayBase::BCExperiment:
        p_typeSpecificWidget = new BCExpOverlayWidget(this);
        break;
    case OverlayBase::Catalog:
        p_typeSpecificWidget = new CatalogOverlayWidget(this);
        break;
    case OverlayBase::GenericXY:
        // GenericXY not yet implemented - create placeholder
        p_typeSpecificWidget = createPlaceholderWidget("Generic XY");
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
                QString errorMessage;
                emit validationStatusChanged(isValid, errorMessage);
            });
    
    // Settings-only connections for real-time updates and progress indication
    if (isSettingsContext()) {
        connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::settingsChanged,
                this, &UnifiedOverlayWidget::onRealTimeUpdate);
        
        // Progress indication connections
        connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::progressOperationStarted,
                this, &UnifiedOverlayWidget::onProgressOperationStarted);
        connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::progressOperationFinished,
                this, &UnifiedOverlayWidget::onProgressOperationFinished);
        connect(p_typeSpecificWidget, &OverlayTypeSpecificWidget::progressValueChanged,
                this, &UnifiedOverlayWidget::onProgressValueChanged);
    }
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

OverlayTypeSpecificWidget* UnifiedOverlayWidget::createPlaceholderWidget(const QString &typeName)
{
    // Create a simple placeholder widget for unimplemented overlay types
    class PlaceholderWidget : public OverlayTypeSpecificWidget {
    public:
        PlaceholderWidget(const QString &name, QWidget *parent = nullptr) 
            : OverlayTypeSpecificWidget(parent), d_typeName(name) {
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
    
    return new PlaceholderWidget(typeName, this);
}

void UnifiedOverlayWidget::enablePreviewMode()
{
    if (d_inPreviewMode || !isDataValid()) {
        return;
    }
    
    // Switch to settings context temporarily
    Context originalContext = d_context;
    d_context = Context::Settings;
    
    // Create preview overlay
    d_previewOverlay = createOverlay();
    if (!d_previewOverlay) {
        d_context = originalContext;
        return;
    }
    
    // Mark as preview mode
    d_previewOverlay->setPreview(true);
    d_inPreviewMode = true;
    
    // Capture sync state
    d_previewSyncState = captureCurrentSettings();
    d_previewSyncValid = true;
    
    // Update UI for preview mode
    updatePreviewModeUI();
    
    // Setup type-specific widget for settings context with preview overlay
    if (p_typeSpecificWidget) {
        p_typeSpecificWidget->setupForSettings(d_previewOverlay);
    }
    
    // Connect real-time updates if not already connected
    configureForContext();
    
    emit previewRequested();
}

void UnifiedOverlayWidget::disablePreviewMode()
{
    if (!d_inPreviewMode) {
        return;
    }
    
    d_inPreviewMode = false;
    d_previewOverlay.reset();
    
    // Clear sync state
    d_previewSyncState.clear();
    d_previewSyncValid = false;
    
    // Switch back to creation context
    d_context = Context::Creation;
    
    // Update UI for creation mode
    updatePreviewModeUI();
    
    // Setup type-specific widget for creation context
    if (p_typeSpecificWidget) {
        p_typeSpecificWidget->setupForCreation();
    }
    
    // Reconfigure connections for creation context
    configureForContext();
    
    emit previewCancelled();
}

void UnifiedOverlayWidget::onPreviewToggled()
{
    if (d_inPreviewMode) {
        disablePreviewMode();
    } else {
        enablePreviewMode();
    }
}

void UnifiedOverlayWidget::updatePreviewModeUI()
{
    if (d_inPreviewMode) {
        p_previewButton->setText("Stop Preview");
        p_previewButton->setStyleSheet(
            "QPushButton {"
            "    background-color: #f44336;"  // Red for stop
            "    color: white;"
            "    border: none;"
            "    padding: 8px 16px;"
            "    font-weight: bold;"
            "    border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #da190b;"
            "}"
        );
        p_previewButton->setToolTip("Stop preview and return to configuration mode");
    } else {
        p_previewButton->setText("Preview");
        p_previewButton->setStyleSheet(
            "QPushButton {"
            "    background-color: #4CAF50;"  // Green for preview
            "    color: white;"
            "    border: none;"
            "    padding: 8px 16px;"
            "    font-weight: bold;"
            "    border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #45a049;"
            "}"
            "QPushButton:disabled {"
            "    background-color: #cccccc;"
            "    color: #666666;"
            "}"
        );
        p_previewButton->setToolTip("Preview overlay with current settings");
        
        // Re-enable based on data validity
        p_previewButton->setEnabled(isDataValid());
    }
}

void UnifiedOverlayWidget::updatePreviewSyncState()
{
    if (!d_inPreviewMode) {
        d_previewSyncValid = false;
        return;
    }
    
    // Compare current settings with captured sync state
    QHash<QString, QVariant> currentSettings = captureCurrentSettings();
    d_previewSyncValid = compareSettings(d_previewSyncState, currentSettings);
}

QHash<QString, QVariant> UnifiedOverlayWidget::captureCurrentSettings() const
{
    QHash<QString, QVariant> settings;
    
    // Capture base overlay options
    if (p_overlayBaseOptionsWidget) {
        // Add overlay options to settings hash
        // Note: This would need specific implementation based on what options exist
        settings["label"] = QString(); // Placeholder - would get actual label
        settings["plotIndex"] = 0; // Placeholder - would get actual plot index
        settings["xOffset"] = 0.0; // Placeholder - would get actual offset
        settings["yOffset"] = 0.0; // Placeholder - would get actual offset
        settings["yScale"] = 1.0; // Placeholder - would get actual scale
        settings["xMin"] = d_xRangeMin; // Use current range
        settings["xMax"] = d_xRangeMax; // Use current range
    }
    
    // Capture curve appearance settings
    if (p_curveAppearanceWidget) {
        // Add curve appearance to settings hash
        // Note: This would need specific implementation based on curve appearance options
        settings["curveStyle"] = static_cast<int>(Qt::SolidLine); // Placeholder
        settings["curveWidth"] = 1; // Placeholder
        settings["curveColor"] = QColor(Qt::blue).name(); // Placeholder
    }
    
    // Capture type-specific settings
    if (p_typeSpecificWidget) {
        // Get type-specific settings
        QHash<QString, QVariant> typeSettings = p_typeSpecificWidget->getSettingsHash();
        for (auto it = typeSettings.begin(); it != typeSettings.end(); ++it) {
            settings[QString("type_%1").arg(it.key())] = it.value();
        }
    }
    
    return settings;
}

bool UnifiedOverlayWidget::compareSettings(const QHash<QString, QVariant> &state1, const QHash<QString, QVariant> &state2) const
{
    if (state1.size() != state2.size()) {
        return false;
    }
    
    for (auto it = state1.begin(); it != state1.end(); ++it) {
        if (!state2.contains(it.key()) || state2[it.key()] != it.value()) {
            return false;
        }
    }
    
    return true;
}

void UnifiedOverlayWidget::invalidatePreviewSync()
{
    if (d_previewSyncValid) {
        d_previewSyncValid = false;
        
        // Update preview button appearance to indicate stale state
        if (p_previewButton && d_inPreviewMode) {
            p_previewButton->setStyleSheet(
                "QPushButton {"
                "    background-color: #ff9800;"  // Orange for stale
                "    color: white;"
                "    border: none;"
                "    padding: 8px 16px;"
                "    font-weight: bold;"
                "    border-radius: 4px;"
                "}"
                "QPushButton:hover {"
                "    background-color: #f57c00;"
                "}"
            );
            p_previewButton->setToolTip("Preview is out of sync with current settings - click to refresh");
        }
    }
}

void UnifiedOverlayWidget::validatePreviewSync()
{
    updatePreviewSyncState();
    
    // Update UI based on sync state
    if (d_inPreviewMode) {
        updatePreviewModeUI();
    }
}