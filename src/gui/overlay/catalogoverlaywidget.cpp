#include "catalogoverlaywidget.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QStandardPaths>
#include <QDebug>

#include <gui/widget/ftmwviewwidget.h>
#include <gui/plot/curveappearancepresetmanager.h>
#include <data/processing/overlayoperation.h>
#include <data/processing/parsers/catalogparser.h>
#include <gui/style/themecolors.h>

CatalogOverlayWidget::CatalogOverlayWidget(const Ft &currentFt, QWidget *parent)
    : OverlayTypeSpecificWidget(currentFt, parent), SettingsStorage(BC::Key::CatalogWidget::key),
      d_fileValid(false),
      d_convolutionInProgress(false)
{
    // Base class handles setupUI() and setupConnections()
}

CatalogOverlayWidget::~CatalogOverlayWidget()
{
    // Cancel any pending convolution operations
    cancelPendingConvolution();
    
    // Explicitly disconnect from OverlayProcessManager to avoid signals to destroyed objects
    auto& manager = OverlayProcessManager::instance();
    disconnect(&manager, nullptr, this, nullptr);
}

void CatalogOverlayWidget::setupForCreation()
{
    d_context = Context::Creation;
    d_overlay.reset();  
    
    // Load default settings for creation context
    loadSettings();

    //Convolution should be disabled for new overlay
    p_convolutionGroupBox->setChecked(false);
    
    // Initialize convolution state - no convolution performed yet
    d_lastConvolutionState.convolutionPerformed = false;
    d_lastConvolutionState.enabled = false;
    d_lastConvolutionState.lineshapeType = p_lineshapeComboBox->currentIndex();
    d_lastConvolutionState.linewidthKHz = p_linewidthSpinBox->value();
    d_lastConvolutionState.minFreqMHz = p_convMinFreqSpinBox->value();
    d_lastConvolutionState.maxFreqMHz = p_convMaxFreqSpinBox->value();
    d_lastConvolutionState.numPoints = p_numPointsSpinBox->value();
    
    updateConvolutionControls();
    updateConvolutionButtonState();
    updateFileInfo();
}

void CatalogOverlayWidget::setupForSettings(std::shared_ptr<OverlayBase> overlay)
{
    d_context = Context::Settings;
    d_overlay = overlay;
    
    if (overlay) {
        auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(overlay);
        if (catalogOverlay) {
            // Load catalog data and settings from existing overlay
            d_catalogData = catalogOverlay->catalogData();
            d_filePath = catalogOverlay->getSourceFile();
            d_fileValid = !d_catalogData.isEmpty();
            
            // Set source file path and store full path for validation
            updatePathDisplayAndTooltip(p_filePathLineEdit, d_filePath);
            
            // Load convolution settings
            p_convolutionGroupBox->setChecked(catalogOverlay->convolutionEnabled());
            p_lineshapeComboBox->setCurrentIndex(static_cast<int>(catalogOverlay->lineshapeType()));
            p_linewidthSpinBox->setValue(catalogOverlay->linewidth());
            p_convMinFreqSpinBox->setValue(catalogOverlay->convolutionMinFreq());
            p_convMaxFreqSpinBox->setValue(catalogOverlay->convolutionMaxFreq());
            p_numPointsSpinBox->setValue(catalogOverlay->numConvolutionPoints());
            updateSpacingDisplay();
            
            // Load filtering range settings from overlay metadata
            p_filterMinFreqSpinBox->setValue(catalogOverlay->filterMinFreq());
            p_filterMaxFreqSpinBox->setValue(catalogOverlay->filterMaxFreq());
            
            // Initialize convolution state based on existing overlay settings
            d_lastConvolutionState.convolutionPerformed = catalogOverlay->convolutionEnabled();
            d_lastConvolutionState.enabled = catalogOverlay->convolutionEnabled();
            d_lastConvolutionState.lineshapeType = static_cast<int>(catalogOverlay->lineshapeType());
            d_lastConvolutionState.linewidthKHz = catalogOverlay->linewidth();
            d_lastConvolutionState.minFreqMHz = catalogOverlay->convolutionMinFreq();
            d_lastConvolutionState.maxFreqMHz = catalogOverlay->convolutionMaxFreq();
            d_lastConvolutionState.numPoints = catalogOverlay->numConvolutionPoints();
        }
    }
    
    updateFileInfo();
    updateConvolutionControls();
    updateConvolutionButtonState();
}

std::shared_ptr<OverlayBase> CatalogOverlayWidget::createOverlay()
{
    if (d_context != Context::Creation) {
        return nullptr;
    }
    
    if (!isDataValid()) {
        return nullptr;
    }
    
    // If we already have a processed overlay (with convolution results), return it
    if (d_overlay) {
        auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(d_overlay);
        if (catalogOverlay) {
            // Ensure current settings are applied to the processed overlay
            applyToOverlay(d_overlay);
            return d_overlay;
        }
    }
    
    // Create new overlay if we don't have one
    auto overlay = std::make_shared<CatalogOverlay>();
    
    // Delegate to applyToOverlay to avoid code duplication
    applyToOverlay(overlay);
    
    // Store the created overlay for convolution operations in creation context
    d_overlay = overlay;
    
    return overlay;
}

void CatalogOverlayWidget::applyToOverlay(std::shared_ptr<OverlayBase> overlay) const
{
    if (!overlay) {
        return;
    }
    
    auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(overlay);
    if (!catalogOverlay) {
        return;
    }
    
    // Apply current settings to the overlay (use pre-filtered data)
    catalogOverlay->setCatalogData(d_filteredData);
    catalogOverlay->setSourceFile(d_filePath);
    
    // Apply convolution settings
    catalogOverlay->setConvolutionEnabled(p_convolutionGroupBox->isChecked());
    catalogOverlay->setLineshapeType(static_cast<CatalogOverlay::LineshapeType>(p_lineshapeComboBox->currentIndex()));
    catalogOverlay->setLinewidth(p_linewidthSpinBox->value());
    catalogOverlay->setConvolutionFreqRange(p_convMinFreqSpinBox->value(), p_convMaxFreqSpinBox->value());
    catalogOverlay->setNumConvolutionPoints(p_numPointsSpinBox->value());
    
    // Apply filtering range settings
    catalogOverlay->setFilterRange(p_filterMinFreqSpinBox->value(), p_filterMaxFreqSpinBox->value());
    
    // COMMENTED OUT: This was overwriting user appearance settings every time applyToOverlay() was called
    // The initial appearance should be set only once during overlay creation, not continuously overwritten
    // Apply curve appearance preset based on convolution mode
    // QString presetName;
    // if (p_convolutionEnabledCheckBox->isChecked()) {
    //     presetName = "Curve - Secondary";  // Smooth curve for convolved data
    // } else {
    //     presetName = "Stem - Secondary";   // Stem plot for discrete transitions
    // }
    // 
    // auto presetManager = CurveAppearancePresetManager::instance();
    // if (presetManager && presetManager->hasPreset(presetName)) {
    //     auto preset = presetManager->getPreset(presetName);
    //     catalogOverlay->setCurveAppearanceMetadata(preset.appearance);
    // }
}

bool CatalogOverlayWidget::validateSettingsImpl()
{
    if (!d_fileValid) {
        setSettingsErrorMessage("Please select a valid catalog file.");
        return false;
    }
    
    if (d_catalogData.isEmpty()) {
        setSettingsErrorMessage("Selected catalog file contains no transitions.");
        return false;
    }
    
    QString errorMessage;
    if (!validateConvolutionSettings(errorMessage)) {
        setSettingsErrorMessage(errorMessage);
        return false;
    }
    
    return true;
}

bool CatalogOverlayWidget::isDataValid() const
{
    return d_fileValid && !d_catalogData.isEmpty();
}

bool CatalogOverlayWidget::hasValidSourceFile() const
{
    return d_fileValid;
}

QString CatalogOverlayWidget::getSourceFilePath() const
{
    return d_filePath;
}

void CatalogOverlayWidget::setSourceFilePath(const QString &path)
{
    if (path.isEmpty()) {
        return;
    }
    
    updatePathDisplayAndTooltip(p_filePathLineEdit, path);
    onFilePathChanged();
}

bool CatalogOverlayWidget::validateSourceFileImpl()
{
    QString path = p_filePathLineEdit->text().trimmed();
    
    if (path.isEmpty()) {
        setSourceFileErrorMessage("Please select a catalog file.");
        d_fileValid = false;
        return false;
    }
    
    if (!QFile::exists(path)) {
        setSourceFileErrorMessage(QString("Catalog file does not exist: %1").arg(path));
        d_fileValid = false;
        return false;
    }
    
    // Try to parse the file
    auto registry = FileParserRegistry::instance();
    auto parser = registry->findParserOfType<CatalogParser>(path);
    
    if (!parser) {
        setSourceFileErrorMessage(QString("No suitable catalog parser found for file: %1").arg(path));
        d_fileValid = false;
        return false;
    }
    
    try {
        CatalogData testData = parser->parse(path);
        if (testData.isEmpty()) {
            setSourceFileErrorMessage(QString("Catalog file contains no valid transitions: %1").arg(path));
            d_fileValid = false;
            return false;
        }
        d_fileValid = true;
        return true;
    } catch (const std::exception &e) {
        setSourceFileErrorMessage(QString("Error parsing catalog file: %1").arg(e.what()));
        d_fileValid = false;
        return false;
    } catch (...) {
        setSourceFileErrorMessage(QString("Unknown error parsing catalog file: %1").arg(path));
        d_fileValid = false;
        return false;
    }
}


QHash<QString, QVariant> CatalogOverlayWidget::getSettingsHash() const
{
    QHash<QString, QVariant> settings;
    
    // File selection settings
    settings[BC::Key::CatalogWidget::filePath] = d_filePath;
    settings[BC::Key::CatalogWidget::fileValid] = d_fileValid;
    
    // Convolution settings (including frequency range for convolution)
    settings[BC::Key::CatalogWidget::convolutionEnabled] = p_convolutionGroupBox->isChecked();
    settings[BC::Key::CatalogWidget::lineshapeType] = p_lineshapeComboBox->currentIndex();
    settings[BC::Key::CatalogWidget::linewidthKHz] = p_linewidthSpinBox->value();
    settings[BC::Key::CatalogWidget::numConvolutionPoints] = p_numPointsSpinBox->value();
    settings[BC::Key::CatalogWidget::convMinFreqMHz] = p_convMinFreqSpinBox->value();
    settings[BC::Key::CatalogWidget::convMaxFreqMHz] = p_convMaxFreqSpinBox->value();
    
    // Filtering settings
    settings[BC::Key::CatalogWidget::saveRangeOnly] = p_saveRangeOnlyCheckBox->isChecked();
    settings[BC::Key::CatalogWidget::filterMinFreqMHz] = p_filterMinFreqSpinBox->value();
    settings[BC::Key::CatalogWidget::filterMaxFreqMHz] = p_filterMaxFreqSpinBox->value();
    
    // Catalog data fingerprint (for detecting data changes)
    if (d_catalogData.size() > 0) {
        settings[BC::Key::CatalogWidget::catalogSize] = d_catalogData.size();
        settings[BC::Key::CatalogWidget::catalogSourceProgram] = d_catalogData.sourceProgram();
        settings[BC::Key::CatalogWidget::catalogMoleculeName] = d_catalogData.moleculeName();
    }
    
    return settings;
}


std::shared_ptr<OverlayOperation> CatalogOverlayWidget::createOperation(OperationCapability::Type type,
                                                                       std::shared_ptr<OverlayBase> overlay) const
{
    switch (type) {
    case OperationCapability::Convolution:
    case OperationCapability::PreviewUpdate:
        {
            if (!overlay) {
                return nullptr;
            }
            
            // Create convolution operation
            return std::make_shared<ConvolutionOperation>(
                overlay,
                p_convolutionGroupBox->isChecked(),
                static_cast<CatalogOverlay::LineshapeType>(p_lineshapeComboBox->currentIndex()),
                p_linewidthSpinBox->value(),
                p_convMinFreqSpinBox->value(),
                p_convMaxFreqSpinBox->value(),
                p_numPointsSpinBox->value(),
                nullptr  // No Qt parent - let OverlayProcessManager manage lifecycle
            );
        }
    case OperationCapability::Creation:
    case OperationCapability::Validation:
        // These use synchronous processing
        return nullptr;
    }
    
    return nullptr;
}


bool CatalogOverlayWidget::hasUnsavedChanges() const
{
    return hasUnsavedConvolutionChanges();
}

bool CatalogOverlayWidget::validateAcceptance()
{
    // Check for unsaved convolution changes
    if (hasUnsavedConvolutionChanges()) {
        QMessageBox::StandardButton reply = QMessageBox::question(this, 
            "Unsaved Convolution Settings",
            "You have unsaved convolution settings. Would you like to apply them?",
            QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
            
        if (reply == QMessageBox::Cancel) {
            return false; // User cancelled dialog acceptance
        } else if (reply == QMessageBox::Yes) {
            // Apply changes by triggering convolution
            onConvolveButtonClicked();
            // Show message about pending operation
            QMessageBox::information(this, "Convolution Started", 
                "Convolution has been started. Please wait for it to complete before closing the dialog.");
            return false; // Don't proceed with acceptance yet, wait for convolution
        }
        // If No was selected, continue with acceptance without applying changes
    }
    
    return true; // Proceed with dialog acceptance
}

void CatalogOverlayWidget::onBrowseButtonClicked()
{
    QString lastPath = get(BC::Key::CatalogWidget::lastFilePath, 
                          QStandardPaths::writableLocation(QStandardPaths::HomeLocation));
    
    QStringList filters;
    filters << "All Catalog Files (*.cat *.xo *.out)"
            << "SPCAT Files (*.cat)"
            << "XIAM Files (*.xo *.out)"
            << "All Files (*)";
    
    QString filePath = QFileDialog::getOpenFileName(this, 
                                                   "Select Catalog File", 
                                                   lastPath, 
                                                   filters.join(";;"));
    
    if (!filePath.isEmpty()) {
        updatePathDisplayAndTooltip(p_filePathLineEdit, filePath);
        onFilePathChanged();
    }
}

void CatalogOverlayWidget::onFilePathChanged()
{
    d_filePath = getStoredFullSourceFilePath(); // Use stored full path instead of potentially abbreviated display text
    
    if (d_filePath.isEmpty()) {
        d_fileValid = false;
        d_catalogData = CatalogData();
        updateFileInfo();
        emit dataValidityChanged(isDataValid());
        return;
    }
    
    emit progressOperationStarted("Loading catalog file...");
    
    loadCatalogFile(d_filePath);
    updateFileInfo();
    
    emit progressOperationFinished();
    emit dataValidityChanged(isDataValid());
    emit settingsChanged();
}

void CatalogOverlayWidget::onConvolutionEnabledToggled(bool enabled)
{
    // Apply the convolution enabled setting to the overlay if it exists
    if (d_overlay) {
        auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(d_overlay);
        if (catalogOverlay) {
            catalogOverlay->setConvolutionEnabled(enabled);
        }
    }
    
    if (enabled) {
        // When enabling convolution, check if overlay already has matching convolved data
        if (overlayHasMatchingConvolutionData()) {
            // Update the stored state to reflect that convolution is already done
            d_lastConvolutionState = getCurrentConvolutionState();
            d_lastConvolutionState.convolutionPerformed = true;
        }
        // If no matching data, the button state update will enable the convolve button
    } else {
        // When disabling convolution, we're switching back to catalog data
        // No need to update d_lastConvolutionState here
    }
    
    updateConvolutionControls();
    updateConvolutionButtonState();
    emit settingsChanged();
}

void CatalogOverlayWidget::onLineshapeTypeChanged(int index)
{
    Q_UNUSED(index);
    emit settingsChanged();
}

void CatalogOverlayWidget::onConvolutionSettingsChanged()
{
    // Update convolution button state based on current vs last settings
    updateConvolutionButtonState();

    
    emit settingsChanged();
}


void CatalogOverlayWidget::onSaveRangeOnlyToggled(bool enabled)
{
    Q_UNUSED(enabled);
    // Trigger centralized filtering when checkbox state changes
    onFilteringParametersChanged();
}


void CatalogOverlayWidget::setupConnections()
{
    connect(p_browseButton, &QToolButton::clicked, this, &CatalogOverlayWidget::onBrowseButtonClicked);
    connect(p_filePathLineEdit, &QLineEdit::textChanged, this, &CatalogOverlayWidget::onFilePathChanged);
    connect(p_convolutionGroupBox, &QGroupBox::toggled, this, &CatalogOverlayWidget::onConvolutionEnabledToggled);
    connect(p_lineshapeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CatalogOverlayWidget::onLineshapeTypeChanged);
    connect(p_linewidthSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayWidget::onConvolutionSettingsChanged);
    connect(p_convMinFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayWidget::onConvolutionSettingsChanged);
    connect(p_convMaxFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayWidget::onConvolutionSettingsChanged);
    connect(p_convMinFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayWidget::updateSpacingDisplay);
    connect(p_convMaxFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayWidget::updateSpacingDisplay);
    connect(p_numPointsSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &CatalogOverlayWidget::onConvolutionSettingsChanged);
    connect(p_numPointsSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &CatalogOverlayWidget::updateSpacingDisplay);
    connect(p_saveRangeOnlyCheckBox, &QCheckBox::toggled, this, &CatalogOverlayWidget::onSaveRangeOnlyToggled);
    connect(p_filterMinFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayWidget::onFilteringParametersChanged);
    connect(p_filterMaxFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayWidget::onFilteringParametersChanged);
    connect(p_convolveButton, &QPushButton::clicked, this, &CatalogOverlayWidget::onConvolveButtonClicked);
    
    // Connect to OverlayProcessManager signals for background convolution
    auto& manager = OverlayProcessManager::instance();
    connect(&manager, &OverlayProcessManager::operationStarted,
            this, &CatalogOverlayWidget::onConvolutionOperationStarted);
    connect(&manager, &OverlayProcessManager::operationProgress,
            this, &CatalogOverlayWidget::onConvolutionOperationProgress);
    connect(&manager, &OverlayProcessManager::operationCompleted,
            this, &CatalogOverlayWidget::onConvolutionOperationCompleted);
    connect(&manager, &OverlayProcessManager::operationFailed,
            this, &CatalogOverlayWidget::onConvolutionOperationFailed);
    connect(&manager, &OverlayProcessManager::operationCancelled,
            this, &CatalogOverlayWidget::onConvolutionOperationCancelled);
}

void CatalogOverlayWidget::loadSettings()
{
    // Clear file selection for creation context
    if (isCreationContext()) {
        p_filePathLineEdit->clear();
        d_filePath.clear();
        d_fileValid = false;
        d_catalogData = CatalogData();
    }
    
    // Load convolution settings from stored preferences
    p_convolutionGroupBox->setChecked(get(BC::Key::CatalogWidget::convolutionEnabled, DEFAULT_CONVOLUTION_ENABLED));
    p_lineshapeComboBox->setCurrentIndex(get(BC::Key::CatalogWidget::lineshapeType, DEFAULT_LINESHAPE_TYPE));
    p_linewidthSpinBox->setValue(get(BC::Key::CatalogWidget::linewidthKHz, DEFAULT_LINEWIDTH));
    updateSpacingDisplay();
    p_saveRangeOnlyCheckBox->setChecked(get(BC::Key::CatalogWidget::saveRangeOnly, DEFAULT_SAVE_RANGE_ONLY));
    
    // Load frequency range settings (will be set from Ft data if available)
    if (d_currentFt.isEmpty()) {
        p_convMinFreqSpinBox->setValue(get(BC::Key::CatalogWidget::convMinFreqMHz, DEFAULT_MIN_FREQ));
        p_convMaxFreqSpinBox->setValue(get(BC::Key::CatalogWidget::convMaxFreqMHz, DEFAULT_MAX_FREQ));
        p_numPointsSpinBox->setValue(get(BC::Key::CatalogWidget::numConvolutionPoints, DEFAULT_NUM_POINTS));
    }
    
    updateFileInfo();
    updateConvolutionControls();
}

void CatalogOverlayWidget::saveSettings()
{
    // Save current file path
    set(BC::Key::CatalogWidget::lastFilePath, d_filePath);
    
    // Save convolution settings
    set(BC::Key::CatalogWidget::convolutionEnabled, p_convolutionGroupBox->isChecked());
    set(BC::Key::CatalogWidget::lineshapeType, p_lineshapeComboBox->currentIndex());
    set(BC::Key::CatalogWidget::linewidthKHz, p_linewidthSpinBox->value());
    set(BC::Key::CatalogWidget::convMinFreqMHz, p_convMinFreqSpinBox->value());
    set(BC::Key::CatalogWidget::convMaxFreqMHz, p_convMaxFreqSpinBox->value());
    set(BC::Key::CatalogWidget::numConvolutionPoints, p_numPointsSpinBox->value());
    set(BC::Key::CatalogWidget::saveRangeOnly, p_saveRangeOnlyCheckBox->isChecked());
    
    // Save filtering range settings
    set(BC::Key::CatalogWidget::filterMinFreqMHz, p_filterMinFreqSpinBox->value());
    set(BC::Key::CatalogWidget::filterMaxFreqMHz, p_filterMaxFreqSpinBox->value());
}

void CatalogOverlayWidget::loadCatalogFile(const QString &filePath)
{
    d_fileValid = false;
    d_catalogData = CatalogData();
    
    if (!QFile::exists(filePath)) {
        return;
    }
    
    auto registry = FileParserRegistry::instance();
    auto parser = registry->findParserOfType<CatalogParser>(filePath);
    
    if (!parser) {
        return;
    }
    
    try {
        d_catalogData = parser->parse(filePath);
        d_fileValid = !d_catalogData.isEmpty();
    } catch (const std::exception &e) {
        d_fileValid = false;
        d_catalogData = CatalogData();
    } catch (...) {
        d_fileValid = false;
        d_catalogData = CatalogData();
    }
    
    // Apply filtering after successful parsing (or clear filtered data on failure)
    onFilteringParametersChanged();
}

void CatalogOverlayWidget::updateFileInfo()
{
    // Find the details frame using object name
    auto detailsFrame = p_sourceFileConfigBox->findChild<QFrame*>("fileDetailsFrame");
    
    if (!d_fileValid || d_catalogData.isEmpty()) {
        p_formatLabel->setText("-");
        p_moleculeLabel->setText("-");
        p_transitionCountLabel->setText("-");
        p_frequencyRangeLabel->setText("-");
        p_overlaySettingsBox->setEnabled(false);
        
        // Hide details frame when no valid file
        if (detailsFrame) {
            detailsFrame->setVisible(false);
        }
        return;
    }
    
    // Show format and molecule info
    p_formatLabel->setText(d_catalogData.sourceProgram());
    p_moleculeLabel->setText(d_catalogData.moleculeName());
    p_transitionCountLabel->setText(QString::number(d_catalogData.size()));
    
    // Auto-set overlay label from molecule name (only in creation context)
    if (d_context == Context::Creation) {
        QString moleculeName = d_catalogData.moleculeName();
        if (!moleculeName.isEmpty() && moleculeName != "-") {
            emit labelUpdateRequested(moleculeName);
        }
    }
    
    // Calculate and display frequency range
    if (d_catalogData.size() > 0) {
        double minFreq = std::numeric_limits<double>::max();
        double maxFreq = std::numeric_limits<double>::lowest();
        
        for (int i = 0; i < d_catalogData.size(); ++i) {
            double freq = d_catalogData.at(i).frequency;
            minFreq = qMin(minFreq, freq);
            maxFreq = qMax(maxFreq, freq);
        }
        
        p_frequencyRangeLabel->setText(formatFrequencyRange(minFreq, maxFreq));
        p_overlaySettingsBox->setEnabled(true);
        
        // Show details frame when we have valid file data
        if (detailsFrame) {
            detailsFrame->setVisible(true);
        }
    }
    else {
        p_overlaySettingsBox->setEnabled(false);
        if (detailsFrame) {
            detailsFrame->setVisible(false);
        }
    }
}

void CatalogOverlayWidget::updateConvolutionControls()
{
    bool convolutionEnabled = p_convolutionGroupBox->isChecked();
    bool hasOverlay = (d_overlay != nullptr);
    
    // Enable/disable the entire group box based on whether we have an overlay
    p_overlaySettingsBox->setEnabled(hasOverlay);
    
    // Individual control states based on checkbox (preserved when group is re-enabled)
    p_lineshapeComboBox->setEnabled(convolutionEnabled);
    p_linewidthSpinBox->setEnabled(convolutionEnabled);
    p_convMinFreqSpinBox->setEnabled(convolutionEnabled);
    p_convMaxFreqSpinBox->setEnabled(convolutionEnabled);
    p_numPointsSpinBox->setEnabled(convolutionEnabled);
    p_spacingDisplayLabel->setEnabled(convolutionEnabled);
}

void CatalogOverlayWidget::updateSpacingDisplay()
{
    double minFreq = p_convMinFreqSpinBox->value();
    double maxFreq = p_convMaxFreqSpinBox->value();
    int numPoints = p_numPointsSpinBox->value();
    
    if (numPoints > 1 && maxFreq > minFreq) {
        double spacing = (maxFreq - minFreq) / (numPoints - 1);
        p_spacingDisplayLabel->setText(QString("%1 MHz").arg(spacing, 0, 'f', 6));
        
        // Add performance warning for very large point counts
        if (numPoints > 500000) {
            p_spacingDisplayLabel->setStyleSheet(QString("QLabel { color: %1; font-weight: bold; }").arg(ThemeColors::getCSSColor(ThemeColors::StatusWarning, this)));
            p_spacingDisplayLabel->setToolTip("Warning: Large number of points may cause slow performance");
        } else if (numPoints > 2000000) {
            p_spacingDisplayLabel->setStyleSheet(QString("QLabel { color: %1; font-weight: bold; }").arg(ThemeColors::getCSSColor(ThemeColors::StatusError, this)));
            p_spacingDisplayLabel->setToolTip("Warning: Very large number of points will cause slow performance");
        } else {
            p_spacingDisplayLabel->setStyleSheet(QString("QLabel { color: %1; }").arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
            p_spacingDisplayLabel->setToolTip("");
        }
    } else {
        p_spacingDisplayLabel->setText("0.000 MHz");
        p_spacingDisplayLabel->setStyleSheet(QString("QLabel { color: %1; }").arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
        p_spacingDisplayLabel->setToolTip("");
    }
}



bool CatalogOverlayWidget::validateConvolutionSettings(QString &errorMessage) const
{
    if (!p_convolutionGroupBox->isChecked()) {
        return true; // No validation needed if convolution disabled
    }
    
    double minFreq = p_convMinFreqSpinBox->value();
    double maxFreq = p_convMaxFreqSpinBox->value();
    
    if (minFreq >= maxFreq) {
        errorMessage = "Minimum frequency must be less than maximum frequency.";
        return false;
    }
    
    if (p_linewidthSpinBox->value() <= 0) {
        errorMessage = "Linewidth must be positive.";
        return false;
    }
    
    if (p_numPointsSpinBox->value() <= 0) {
        errorMessage = "Point spacing must be positive.";
        return false;
    }
    
    return true;
}

QString CatalogOverlayWidget::formatFrequencyRange(double min, double max) const
{
    return QString("%1 - %2 MHz").arg(min, 0, 'f', 1).arg(max, 0, 'f', 1);
}

void CatalogOverlayWidget::configureSpinBox(QDoubleSpinBox *spinBox, const QString &minKey, const QString &maxKey, 
                                           const QString &decimalsKey, const QString &stepKey, 
                                           double defaultMin, double defaultMax, int defaultDecimals, double defaultStep)
{
    spinBox->setMinimum(get(minKey, defaultMin));
    spinBox->setMaximum(get(maxKey, defaultMax));
    spinBox->setDecimals(get(decimalsKey, defaultDecimals));
    spinBox->setSingleStep(get(stepKey, defaultStep));
}

void CatalogOverlayWidget::triggerBackgroundConvolution()
{
    if (!d_overlay || d_convolutionInProgress) {
        return;
    }
    
    // Cancel any pending convolution
    cancelPendingConvolution();
    
    // Create convolution operation
    auto convolutionOp = std::make_shared<ConvolutionOperation>(
        d_overlay,
        p_convolutionGroupBox->isChecked(),
        static_cast<CatalogOverlay::LineshapeType>(p_lineshapeComboBox->currentIndex()),
        p_linewidthSpinBox->value(),
        p_convMinFreqSpinBox->value(),
        p_convMaxFreqSpinBox->value(),
        p_numPointsSpinBox->value(),
        nullptr  // No Qt parent - let OverlayProcessManager manage lifecycle
    );
    
    // Set cache state to pending before starting operation
    auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(d_overlay);
    if (catalogOverlay) {
        catalogOverlay->setCachePending();
    }
    
    // Queue the operation with high priority for real-time updates
    auto& manager = OverlayProcessManager::instance();
    d_currentConvolutionId = manager.queueOperation(convolutionOp, OverlayProcessManager::Priority::High);
    d_convolutionInProgress = true;
    
    // Emit progress operation started signal
    emit progressOperationStarted("Updating convolution...");
}

void CatalogOverlayWidget::cancelPendingConvolution()
{
    if (!d_currentConvolutionId.isEmpty()) {
        auto& manager = OverlayProcessManager::instance();
        manager.cancelOperation(d_currentConvolutionId);
        d_currentConvolutionId.clear();
        d_convolutionInProgress = false;
    }
}

// Background operation signal handlers
void CatalogOverlayWidget::onConvolutionOperationStarted(const QString &operationId)
{
    if (operationId != d_currentConvolutionId) {
        return;
    }
    
    emit progressOperationStarted("Performing convolution...");
}

void CatalogOverlayWidget::onConvolutionOperationProgress(const QString &operationId, int percentage, const QString &message)
{
    Q_UNUSED(message);
    
    if (operationId != d_currentConvolutionId) {
        return;
    }
    
    emit progressValueChanged(percentage);
}

void CatalogOverlayWidget::onConvolutionOperationCompleted(const QString &operationId, std::shared_ptr<OverlayBase> result)
{
    if (operationId != d_currentConvolutionId) {
        return;
    }
    
    d_currentConvolutionId.clear();
    d_convolutionInProgress = false;
    
    // Update the overlay with the result
    if (result) {
        d_overlay = result;
    }
    
    emit progressOperationFinished();
    emit settingsChanged(); // Trigger UI updates
}

void CatalogOverlayWidget::onConvolutionOperationFailed(const QString &operationId, const QString &error)
{
    if (operationId != d_currentConvolutionId) {
        return;
    }
    
    d_currentConvolutionId.clear();
    d_convolutionInProgress = false;
    
    // Reset cache state on failure
    auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(d_overlay);
    if (catalogOverlay) {
        catalogOverlay->invalidateConvolutionCache();
    }
    
    qWarning() << "Convolution operation failed:" << error;
    emit progressOperationFinished();
}

void CatalogOverlayWidget::onConvolutionOperationCancelled(const QString &operationId)
{
    if (operationId != d_currentConvolutionId) {
        return;
    }
    
    d_currentConvolutionId.clear();
    d_convolutionInProgress = false;
    
    // Reset cache state on cancellation
    auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(d_overlay);
    if (catalogOverlay) {
        catalogOverlay->invalidateConvolutionCache();
    }
    
    emit progressOperationFinished();
}

void CatalogOverlayWidget::onFilteringParametersChanged()
{
    // Centralized filtering slot called whenever any filtering parameter changes:
    // 1. After catalog is parsed successfully (via onFilePathChanged)
    // 2. When filtering checkbox is toggled (via onSaveRangeOnlyToggled)
    // 3. When frequency ranges change (from spinboxes or base overlay widget)
    
    if (!d_fileValid || d_catalogData.isEmpty()) {
        // No valid data to filter - clear filtered data
        d_filteredData = CatalogData();
        emit dataValidityChanged(false);
        return;
    }
    
    // Start with raw catalog data
    d_filteredData = d_catalogData;
    
    // Apply frequency range filtering if enabled
    if (p_saveRangeOnlyCheckBox->isChecked()) {
        // Filter transitions to only include those within the frequency range
        double minFreq = p_filterMinFreqSpinBox->value();
        double maxFreq = p_filterMaxFreqSpinBox->value();
        
        QVector<TransitionData> filteredTransitions;
        for (int i = 0; i < d_catalogData.size(); ++i) {
            const auto &transition = d_catalogData.at(i);
            if (transition.frequency >= minFreq && transition.frequency <= maxFreq) {
                filteredTransitions.append(transition);
            }
        }
        
        // triggers deep copy; original transitions are still in d_catalogData
        // in case user later disables filtering or changes ranges
        d_filteredData.setTransitions(filteredTransitions);
    }
    
    // Emit signals to update UI and trigger real-time preview updates
    bool hasValidData = !d_filteredData.isEmpty();
    emit dataValidityChanged(hasValidData);
    emit settingsChanged(); // Trigger real-time preview updates in settings context
}

void CatalogOverlayWidget::createSourceFileConfigUI(QGroupBox *parent)
{
    // Use compact horizontal layout instead of form layout
    auto mainLayout = new QVBoxLayout(parent);
    mainLayout->setSpacing(6);
    mainLayout->setContentsMargins(6, 6, 6, 6);
    
    // Primary file selection row - single row for file path + browse + status
    auto fileRow = new QHBoxLayout();
    fileRow->setSpacing(6);
    
    // File path input (give adequate space for visibility)
    p_filePathLineEdit = new QLineEdit(parent);
    p_filePathLineEdit->setPlaceholderText("Select catalog file (.cat, .xo, .out)...");
    p_filePathLineEdit->setMinimumWidth(250); // Ensure adequate width
    fileRow->addWidget(p_filePathLineEdit, 1); // Give it stretch priority
    
    // Browse button (compact with icon)
    p_browseButton = new QToolButton(parent);
    p_browseButton->setText("📁"); // Use folder icon instead of text to save space
    p_browseButton->setToolTip("Browse for catalog file");
    p_browseButton->setMaximumSize(30, 30);
    fileRow->addWidget(p_browseButton);
    
    mainLayout->addLayout(fileRow);
    
    // File information display (compact, expandable on click)
    auto detailsFrame = new QFrame(parent);
    detailsFrame->setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
    detailsFrame->setLineWidth(1);
    auto detailsLayout = new QGridLayout(detailsFrame);
    detailsLayout->setSpacing(4);
    detailsLayout->setContentsMargins(6, 4, 6, 4);
    
    // Create compact labels for file information
    p_formatLabel = new QLabel("-", detailsFrame);
    p_moleculeLabel = new QLabel("-", detailsFrame);
    p_transitionCountLabel = new QLabel("-", detailsFrame);
    p_frequencyRangeLabel = new QLabel("-", detailsFrame);
    
    // Configure labels for compact display
    auto configureCompactLabel = [this](QLabel* label) {
        label->setWordWrap(false);
        label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        label->setMinimumHeight(18);
        label->setMaximumHeight(20);
        label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        label->setStyleSheet(QString("QLabel { font-size: 11px; color: %1; }").arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
    };
    
    configureCompactLabel(p_formatLabel);
    configureCompactLabel(p_moleculeLabel);
    configureCompactLabel(p_transitionCountLabel);
    configureCompactLabel(p_frequencyRangeLabel);
    
    // Arrange file info in 2x2 grid for compact display
    auto formatLabel = new QLabel("Format:", detailsFrame);
    formatLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    detailsLayout->addWidget(formatLabel, 0, 0);
    detailsLayout->addWidget(p_formatLabel, 0, 1);
    auto moleculeLabel = new QLabel("Molecule:", detailsFrame);
    moleculeLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    detailsLayout->addWidget(moleculeLabel, 0, 2);
    detailsLayout->addWidget(p_moleculeLabel, 0, 3);
    auto transitionsLabel = new QLabel("Transitions:", detailsFrame);
    transitionsLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    detailsLayout->addWidget(transitionsLabel, 1, 0);
    detailsLayout->addWidget(p_transitionCountLabel, 1, 1);
    auto rangeLabel = new QLabel("Range:", detailsFrame);
    rangeLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    detailsLayout->addWidget(rangeLabel, 1, 2);
    detailsLayout->addWidget(p_frequencyRangeLabel, 1, 3);
    
    // Configure grid column stretches for even distribution
    detailsLayout->setColumnStretch(1, 1);
    detailsLayout->setColumnStretch(3, 1);
    
    mainLayout->addWidget(detailsFrame);
    
    // Initially hide details frame - will be shown when file is loaded
    detailsFrame->setVisible(false);
    detailsFrame->setObjectName("fileDetailsFrame"); // For easy access in updateFileInfo()
}

void CatalogOverlayWidget::createSourceFileSettingsUI(QGroupBox *parent)
{
    QFormLayout *sourceLayout = new QFormLayout(parent);
    
    // Save range only option (source-dependent)
    p_saveRangeOnlyCheckBox = new QCheckBox("Save only transitions within frequency range (recommended)");
    p_saveRangeOnlyCheckBox->setChecked(get(BC::Key::CatalogWidget::saveRangeOnly, DEFAULT_SAVE_RANGE_ONLY));
    p_saveRangeOnlyCheckBox->setToolTip("When enabled, only saves catalog transitions within the frequency range, reducing file size and improving performance.");
    sourceLayout->addRow(p_saveRangeOnlyCheckBox);
    
    // Filtering frequency range spinboxes
    QHBoxLayout *filterRangeLayout = new QHBoxLayout();
    p_filterMinFreqSpinBox = new QDoubleSpinBox(parent);
    configureSpinBox(p_filterMinFreqSpinBox,
                     BC::Key::CatalogWidget::freqMin, BC::Key::CatalogWidget::freqMax,
                     BC::Key::CatalogWidget::freqDecimals, BC::Key::CatalogWidget::freqStep,
                     0.0, 10000000.0, 3, 1.0);
    p_filterMinFreqSpinBox->setSuffix(" MHz");
    
    p_filterMaxFreqSpinBox = new QDoubleSpinBox(parent);
    configureSpinBox(p_filterMaxFreqSpinBox,
                     BC::Key::CatalogWidget::freqMin, BC::Key::CatalogWidget::freqMax,
                     BC::Key::CatalogWidget::freqDecimals, BC::Key::CatalogWidget::freqStep,
                     0.0, 10000000.0, 3, 1.0);
    p_filterMaxFreqSpinBox->setSuffix(" MHz");
    
    // Initialize with intelligent defaults from Ft data
    if (!d_currentFt.isEmpty()) {
        auto ftRange = d_currentFt.xRange();
        p_filterMinFreqSpinBox->setValue(ftRange.first);
        p_filterMaxFreqSpinBox->setValue(ftRange.second);
    } else {
        p_filterMinFreqSpinBox->setValue(get(BC::Key::CatalogWidget::filterMinFreqMHz, DEFAULT_FILTER_MIN_FREQ));
        p_filterMaxFreqSpinBox->setValue(get(BC::Key::CatalogWidget::filterMaxFreqMHz, DEFAULT_FILTER_MAX_FREQ));
    }
    
    filterRangeLayout->addWidget(p_filterMinFreqSpinBox);
    filterRangeLayout->addWidget(new QLabel("to"));
    filterRangeLayout->addWidget(p_filterMaxFreqSpinBox);
    filterRangeLayout->addStretch();
    sourceLayout->addRow("Filter Range:", filterRangeLayout);
}

void CatalogOverlayWidget::createTypeSpecificSettingsUI(QGroupBox *parent)
{
    auto mainLayout = new QVBoxLayout(parent);
    mainLayout->setSpacing(8);
    parent->setTitle("Catalog Settings");
    
    // Master collapsible group: Convolution Controls
    p_convolutionGroupBox = new QGroupBox("Convolution Enabled", parent);
    p_convolutionGroupBox->setFlat(true);
    p_convolutionGroupBox->setCheckable(true);
    p_convolutionGroupBox->setChecked(get(BC::Key::CatalogWidget::convolutionEnabled, DEFAULT_CONVOLUTION_ENABLED));
    mainLayout->addWidget(p_convolutionGroupBox);
    
    // Collapsible content widget for convolution controls
    auto convolutionContentWidget = new QWidget(p_convolutionGroupBox);
    auto convolutionContentLayout = new QVBoxLayout(convolutionContentWidget);
    convolutionContentLayout->setSpacing(4); // Reduced spacing between flat GroupBoxes
    convolutionContentLayout->setContentsMargins(0, 0, 0, 0);
    
    auto convolutionGroupLayout = new QVBoxLayout(p_convolutionGroupBox);
    convolutionGroupLayout->addWidget(convolutionContentWidget);
    
    // Group 1: Line Shape Parameters
    auto shapeGroup = new QGroupBox("Line Shape", convolutionContentWidget);
    shapeGroup->setFlat(true); // Flat appearance for nested GroupBox
    auto shapeLayout = new QGridLayout(shapeGroup);
    shapeLayout->setContentsMargins(3, 2, 3, 3); // Reduced margins for flat GroupBox
    shapeLayout->setSpacing(3); // Reduced spacing for compact appearance
    
    p_lineshapeComboBox = new QComboBox(shapeGroup);
    p_lineshapeComboBox->addItems({"Lorentzian", "Gaussian"});
    p_lineshapeComboBox->setCurrentIndex(get(BC::Key::CatalogWidget::lineshapeType, DEFAULT_LINESHAPE_TYPE));
    
    p_linewidthSpinBox = new QDoubleSpinBox(shapeGroup);
    configureSpinBox(p_linewidthSpinBox, 
                     BC::Key::CatalogWidget::linewidthMin, BC::Key::CatalogWidget::linewidthMax,
                     BC::Key::CatalogWidget::linewidthDecimals, BC::Key::CatalogWidget::linewidthStep,
                     0.1, 10000.0, 1, 10.0);
    p_linewidthSpinBox->setSuffix(" kHz");
    p_linewidthSpinBox->setValue(get(BC::Key::CatalogWidget::linewidthKHz, DEFAULT_LINEWIDTH));
    p_linewidthSpinBox->setKeyboardTracking(false);
    
    auto typeLabel = new QLabel("Type:", shapeGroup);
    typeLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    shapeLayout->addWidget(typeLabel, 0, 0);
    shapeLayout->addWidget(p_lineshapeComboBox, 0, 1);
    auto widthLabel = new QLabel("Width (FWHM):", shapeGroup);
    widthLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    shapeLayout->addWidget(widthLabel, 1, 0);
    shapeLayout->addWidget(p_linewidthSpinBox, 1, 1);
    
    // Group 2: Frequency Range & Resolution
    auto rangeGroup = new QGroupBox("Frequency Range & Resolution", convolutionContentWidget);
    rangeGroup->setFlat(true); // Flat appearance for nested GroupBox
    auto rangeLayout = new QGridLayout(rangeGroup);
    rangeLayout->setContentsMargins(3, 2, 3, 3); // Reduced margins for flat GroupBox
    rangeLayout->setSpacing(3); // Reduced spacing for compact appearance
    
    // Frequency range controls
    p_convMinFreqSpinBox = new QDoubleSpinBox(rangeGroup);
    configureSpinBox(p_convMinFreqSpinBox,
                     BC::Key::CatalogWidget::freqMin, BC::Key::CatalogWidget::freqMax,
                     BC::Key::CatalogWidget::freqDecimals, BC::Key::CatalogWidget::freqStep,
                     0.0, 10000000.0, 3, 1.0);
    p_convMinFreqSpinBox->setSuffix(" MHz");
    p_convMinFreqSpinBox->setKeyboardTracking(false);
    
    p_convMaxFreqSpinBox = new QDoubleSpinBox(rangeGroup);
    configureSpinBox(p_convMaxFreqSpinBox,
                     BC::Key::CatalogWidget::freqMin, BC::Key::CatalogWidget::freqMax,
                     BC::Key::CatalogWidget::freqDecimals, BC::Key::CatalogWidget::freqStep,
                     0.0, 10000000.0, 3, 1.0);
    p_convMaxFreqSpinBox->setSuffix(" MHz");
    p_convMaxFreqSpinBox->setKeyboardTracking(false);
    
    p_numPointsSpinBox = new QSpinBox(rangeGroup);
    p_numPointsSpinBox->setMinimum(get(BC::Key::CatalogWidget::numPointsMin, 100));
    p_numPointsSpinBox->setMaximum(get(BC::Key::CatalogWidget::numPointsMax, 10000000));
    p_numPointsSpinBox->setSingleStep(get(BC::Key::CatalogWidget::numPointsStep, 100));
    p_numPointsSpinBox->setKeyboardTracking(false);
    
    p_spacingDisplayLabel = new QLabel("0.000 MHz", rangeGroup);
    p_spacingDisplayLabel->setStyleSheet(QString("QLabel { color: %1; font-size: 11px; }").arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
    
    auto rangeRangeLabel = new QLabel("Range:", rangeGroup);
    rangeRangeLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    rangeLayout->addWidget(rangeRangeLabel, 0, 0);
    rangeLayout->addWidget(p_convMinFreqSpinBox, 0, 1);
    auto toLabel = new QLabel("to", rangeGroup);
    toLabel->setAlignment(Qt::AlignCenter | Qt::AlignVCenter);
    rangeLayout->addWidget(toLabel, 0, 2);
    rangeLayout->addWidget(p_convMaxFreqSpinBox, 0, 3);
    auto pointsLabel = new QLabel("Points:", rangeGroup);
    pointsLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    rangeLayout->addWidget(pointsLabel, 1, 0);
    rangeLayout->addWidget(p_numPointsSpinBox, 1, 1);
    auto spacingLabel = new QLabel("Spacing:", rangeGroup);
    spacingLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    rangeLayout->addWidget(spacingLabel, 1, 2);
    rangeLayout->addWidget(p_spacingDisplayLabel, 1, 3);
    
    // Group 3: Action Controls
    auto actionLayout = new QHBoxLayout();
    p_convolveButton = new QPushButton("Convolve", convolutionContentWidget);
    p_convolveButton->setEnabled(false);
    p_convolveButton->setMinimumHeight(30);
    actionLayout->addWidget(p_convolveButton);
    actionLayout->addStretch();
    
    // Add groups to convolution content layout
    convolutionContentLayout->addWidget(shapeGroup);
    convolutionContentLayout->addWidget(rangeGroup);
    convolutionContentLayout->addLayout(actionLayout);
    convolutionContentLayout->addStretch();
    
    // Connect GroupBox toggled signal to hide/show content widget for true collapsible behavior
    connect(p_convolutionGroupBox, &QGroupBox::toggled, convolutionContentWidget, &QWidget::setVisible);
    
    // Set initial visibility based on GroupBox state
    convolutionContentWidget->setVisible(p_convolutionGroupBox->isChecked());

    // Initialize with intelligent defaults from Ft data
    if (!d_currentFt.isEmpty()) {
        auto ftRange = d_currentFt.xRange();
        p_convMinFreqSpinBox->setValue(ftRange.first);
        p_convMaxFreqSpinBox->setValue(ftRange.second);
        // Calculate intelligent default number of points based on range and FT spacing
        double range = ftRange.second - ftRange.first;
        double ftSpacing = d_currentFt.xSpacing();
        int intelligentPoints = qMin(10000000, qMax(100, static_cast<int>(range / ftSpacing)));
        p_numPointsSpinBox->setValue(intelligentPoints);
    } else {
        p_convMinFreqSpinBox->setValue(get(BC::Key::CatalogWidget::convMinFreqMHz, DEFAULT_MIN_FREQ));
        p_convMaxFreqSpinBox->setValue(get(BC::Key::CatalogWidget::convMaxFreqMHz, DEFAULT_MAX_FREQ));
        p_numPointsSpinBox->setValue(get(BC::Key::CatalogWidget::numConvolutionPoints, DEFAULT_NUM_POINTS));
    }
    updateSpacingDisplay();
}

CatalogOverlayWidget::ConvolutionState CatalogOverlayWidget::getCurrentConvolutionState() const
{
    ConvolutionState current;
    current.enabled = p_convolutionGroupBox->isChecked();
    current.lineshapeType = p_lineshapeComboBox->currentIndex();
    current.linewidthKHz = p_linewidthSpinBox->value();
    current.minFreqMHz = p_convMinFreqSpinBox->value();
    current.maxFreqMHz = p_convMaxFreqSpinBox->value();
    current.numPoints = p_numPointsSpinBox->value();
    return current;
}

void CatalogOverlayWidget::updateConvolutionButtonState()
{
    if (!p_convolveButton) {
        return;
    }
    
    ConvolutionState current = getCurrentConvolutionState();
    
    // Enable button if:
    // 1. No convolution has been performed yet, OR
    // 2. Current settings differ from last performed convolution
    bool shouldEnable = !d_lastConvolutionState.convolutionPerformed || 
                        (current != d_lastConvolutionState);
    
    // Only enable if convolution is enabled and we have valid data
    shouldEnable = shouldEnable && current.enabled && isDataValid();
    
    // Don't enable during convolution processing
    shouldEnable = shouldEnable && !d_convolutionInProgress;
    
    p_convolveButton->setEnabled(shouldEnable);
}

bool CatalogOverlayWidget::hasUnsavedConvolutionChanges() const
{
    if (!p_convolveButton) {
        return false;
    }
    
    // If the button is enabled, it means there are unsaved changes
    return p_convolveButton->isEnabled();
}

bool CatalogOverlayWidget::overlayHasMatchingConvolutionData() const
{
    if (d_context != Context::Settings || !d_overlay) {
        return false; // Only relevant in settings context with an existing overlay
    }
    
    auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(d_overlay);
    if (!catalogOverlay) {
        return false;
    }
    
    // Check if overlay has convolved data and if the settings match current settings
    if (!catalogOverlay->hasConvolvedData()) {
        return false;
    }
    
    ConvolutionState currentSettings = getCurrentConvolutionState();
    ConvolutionState overlaySettings;
    overlaySettings.enabled = catalogOverlay->convolutionEnabled();
    overlaySettings.lineshapeType = static_cast<int>(catalogOverlay->lineshapeType());
    overlaySettings.linewidthKHz = catalogOverlay->linewidth();
    overlaySettings.minFreqMHz = catalogOverlay->convolutionMinFreq();
    overlaySettings.maxFreqMHz = catalogOverlay->convolutionMaxFreq();
    overlaySettings.numPoints = catalogOverlay->numConvolutionPoints();
    
    return currentSettings == overlaySettings;
}

void CatalogOverlayWidget::onConvolveButtonClicked()
{
    if (!isDataValid() || d_convolutionInProgress) {
        return;
    }
    
    // Store current settings as the last performed convolution
    d_lastConvolutionState = getCurrentConvolutionState();
    d_lastConvolutionState.convolutionPerformed = true;
    
    // Update button state (should disable it)
    updateConvolutionButtonState();
    
    // Trigger the background convolution
    triggerBackgroundConvolution();
}

void CatalogOverlayWidget::configureForCreationContext()
{
    // Creation context: Emphasize catalog file selection and discovery
    if (p_filePathLineEdit) {
        p_filePathLineEdit->setPlaceholderText("Select a catalog file to create overlay...");
    }
    
    // Show helpful status message for file selection in molecule label
    if (p_moleculeLabel) {
        p_moleculeLabel->setText("Select a catalog file to begin");
        p_moleculeLabel->setStyleSheet(QString("QLabel { color: %1; font-style: italic; }").arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
    }
    
    // Set defaults from settings for creation context
    if (p_linewidthSpinBox) {
        double defaultLinewidth = get(BC::Key::CatalogWidget::linewidthKHz, 100.0); // Default 100 kHz
        p_linewidthSpinBox->setValue(defaultLinewidth);
    }
    
    // Make convolution button more prominent
    if (p_convolveButton) {
        p_convolveButton->setStyleSheet("QPushButton { font-weight: bold; }");
    }
}

void CatalogOverlayWidget::configureForSettingsContext()
{
    // Settings context: Show existing catalog information and convolution status
    if (d_overlay) {
        auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(d_overlay);
        if (catalogOverlay) {
            // Don't override molecule name with status message - keep actual data visible
            // The molecule name should always show the actual catalog molecule name
            // Status information should be shown elsewhere or not override data fields
            
            // Don't override transition count with convolution status
            // The transition count should always show the actual number of transitions
            // Convolution status is already visible in the convolution controls section
        }
    }
    
    // Reduce emphasis on convolution button in settings mode
    if (p_convolveButton) {
        p_convolveButton->setStyleSheet("");
    }
}
