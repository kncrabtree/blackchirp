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
#include <gui/widget/settingstable.h>
#include <gui/plot/curveappearancepresetmanager.h>
#include <data/processing/overlayoperation.h>
#include <data/processing/parsers/catalogparser.h>
#include <gui/style/themecolors.h>

namespace {
// Apply a themed foreground color to a label. A widget-scoped
// stylesheet (themed via ThemeColors::getCSSColor) is used rather than
// a palette: a QLabel hosted in a QTableWidget cell does not reliably
// pick up a palette WindowText override, but a stylesheet color always
// wins. The color is still theme-derived, not a hard-coded string.
void styleSubtleLabel(QLabel *label, ThemeColors::ColorRole role,
                      bool italic = false, bool bold = false)
{
    label->setStyleSheet(QString("color:%1;")
        .arg(ThemeColors::getCSSColor(role, label)));
    QFont f = label->font();
    f.setItalic(italic);
    f.setBold(bold);
    label->setFont(f);
}
}

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

bool CatalogOverlayWidget::isConvolutionEnabled() const
{
    return p_convolutionSectionBox && p_convolutionSectionBox->isChecked();
}

void CatalogOverlayWidget::setConvolutionEnabled(bool enabled)
{
    // A state change emits QCheckBox::toggled, which drives both the
    // bound-row collapse (bindSectionRows) and the
    // convolutionEnabledChanged() relay feeding the convolution state
    // machine.
    if (p_convolutionSectionBox)
        p_convolutionSectionBox->setChecked(enabled);
}

void CatalogOverlayWidget::setupForCreation()
{
    d_context = Context::Creation;
    d_overlay.reset();  
    
    // Load default settings for creation context
    loadSettings();

    //Convolution should be disabled for new overlay
    setConvolutionEnabled(false);
    
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
            setConvolutionEnabled(catalogOverlay->convolutionEnabled());
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
    catalogOverlay->setConvolutionEnabled(isConvolutionEnabled());
    catalogOverlay->setLineshapeType(static_cast<CatalogOverlay::LineshapeType>(p_lineshapeComboBox->currentIndex()));
    catalogOverlay->setLinewidth(p_linewidthSpinBox->value());
    catalogOverlay->setConvolutionFreqRange(p_convMinFreqSpinBox->value(), p_convMaxFreqSpinBox->value());
    catalogOverlay->setNumConvolutionPoints(p_numPointsSpinBox->value());
    
    // Apply filtering range settings
    catalogOverlay->setFilterRange(p_filterMinFreqSpinBox->value(), p_filterMaxFreqSpinBox->value());
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
    QString path = getStoredFullSourceFilePath();
    
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
    settings[BC::Key::CatalogWidget::convolutionEnabled] = isConvolutionEnabled();
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
                isConvolutionEnabled(),
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
        // Refresh the base Source File Settings tier enable-state
        // (filtering) for the now-invalid source.
        validateSourceFile();
        emit dataValidityChanged(isDataValid());
        return;
    }

    emit progressOperationStarted("Loading catalog file...");

    loadCatalogFile(d_filePath);
    updateFileInfo();

    // Re-run base source-file validation so the Source File Settings
    // tier (catalog filtering) enables as soon as a valid catalog is
    // selected in Creation, not only in Settings/edit mode.
    validateSourceFile();

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
    updateConvolutionButtonState();
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
    connect(this, &CatalogOverlayWidget::convolutionEnabledChanged, this, &CatalogOverlayWidget::onConvolutionEnabledToggled);
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
    setConvolutionEnabled(get(BC::Key::CatalogWidget::convolutionEnabled, DEFAULT_CONVOLUTION_ENABLED));
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
    set(BC::Key::CatalogWidget::convolutionEnabled, isConvolutionEnabled());
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

void CatalogOverlayWidget::applyDetailRowVisibility()
{
    // Detail rows are shown only when a parsed catalog is present and
    // the source-file-config section is expanded (always so for the
    // non-checkable Creation heading; gated by the checkbox otherwise).
    const bool show = (d_catalogData.size() > 0) && isSourceConfigEnabled();
    for (int r : d_fileDetailRows)
        p_settingsTable->setRowHidden(r, !show);
}

void CatalogOverlayWidget::refreshSourceFileConfigState()
{
    // Re-assert detail-row visibility after the base applies section
    // context state (which, for the non-checkable Creation heading,
    // expands every bound row including these).
    applyDetailRowVisibility();
}

void CatalogOverlayWidget::updateFileInfo()
{
    if (d_catalogData.isEmpty()) {
        p_formatLabel->setText("-");
        p_moleculeLabel->setText("-");
        p_transitionCountLabel->setText("-");
        p_frequencyRangeLabel->setText("-");
        setConvolutionRegionEnabled(false);

        // Hide detail rows when no valid file
        applyDetailRowVisibility();
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
        setConvolutionRegionEnabled(true);
    }
    else {
        setConvolutionRegionEnabled(false);
    }

    // Detail rows track parsed data and section-expand state.
    applyDetailRowVisibility();
}

void CatalogOverlayWidget::updateConvolutionControls()
{
    bool convolutionEnabled = isConvolutionEnabled();

    // The convolution tier is usable once there is something to
    // convolve. In Settings/Edit that is an existing overlay; in
    // Creation no overlay exists until the dialog is accepted, so the
    // gate is a parsed catalog instead. Gating Creation on d_overlay
    // would disable the section checkbox the moment it is checked,
    // wedging it in the checked state.
    bool hasConvolvableData = isCreationContext() ? isDataValid()
                                                  : (d_overlay != nullptr);

    setConvolutionRegionEnabled(hasConvolvableData);
    
    // Individual control states based on checkbox (preserved when group is re-enabled)
    p_lineshapeComboBox->setEnabled(convolutionEnabled);
    p_linewidthSpinBox->setEnabled(convolutionEnabled);
    p_convMinFreqSpinBox->setEnabled(convolutionEnabled);
    p_convMaxFreqSpinBox->setEnabled(convolutionEnabled);
    p_numPointsSpinBox->setEnabled(convolutionEnabled);
    p_spacingDisplayLabel->setEnabled(convolutionEnabled);
}

void CatalogOverlayWidget::setConvolutionRegionEnabled(bool enabled)
{
    // Stands in for the old whole-QGroupBox enable: grey the
    // convolution heading + its bound rows and block the section
    // checkbox so the tier cannot be toggled when there is no data /
    // overlay to convolve.
    if (d_convolutionSection >= 0)
        p_settingsTable->setBoundRowsEnabled(d_convolutionSection, enabled);
    if (p_convolutionSectionBox)
        p_convolutionSectionBox->setEnabled(enabled);
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
            styleSubtleLabel(p_spacingDisplayLabel, ThemeColors::StatusWarning, false, true);
            p_spacingDisplayLabel->setToolTip("Warning: Large number of points may cause slow performance");
        } else if (numPoints > 2000000) {
            styleSubtleLabel(p_spacingDisplayLabel, ThemeColors::StatusError, false, true);
            p_spacingDisplayLabel->setToolTip("Warning: Very large number of points will cause slow performance");
        } else {
            styleSubtleLabel(p_spacingDisplayLabel, ThemeColors::SubtleText);
            p_spacingDisplayLabel->setToolTip("");
        }
    } else {
        p_spacingDisplayLabel->setText("0.000 MHz");
        styleSubtleLabel(p_spacingDisplayLabel, ThemeColors::SubtleText);
        p_spacingDisplayLabel->setToolTip("");
    }
}



bool CatalogOverlayWidget::validateConvolutionSettings(QString &errorMessage) const
{
    if (!isConvolutionEnabled()) {
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
        isConvolutionEnabled(),
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
    
    // Update button state to show "Cancel"
    updateConvolutionButtonState();
    
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
        
        // Reset cache state on cancellation
        auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(d_overlay);
        if (catalogOverlay) {
            catalogOverlay->invalidateConvolutionCache();
        }
        
        // Reset convolution state so button becomes enabled again
        d_lastConvolutionState.convolutionPerformed = false;
        
        // Update button state back to normal
        updateConvolutionButtonState();
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
    
    // Update button state back to normal
    updateConvolutionButtonState();
    
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
    
    // Reset convolution state so button becomes enabled again
    d_lastConvolutionState.convolutionPerformed = false;
    
    // Update button state back to normal
    updateConvolutionButtonState();
    
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
    
    // Reset convolution state so button becomes enabled again
    d_lastConvolutionState.convolutionPerformed = false;
    
    // Update button state back to normal
    updateConvolutionButtonState();
    
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

void CatalogOverlayWidget::populateSourceFileConfigRows(SettingsTable *table)
{
    // Fixed source-file row: path + browse. The base has already added
    // the checkable section row above.
    p_filePathLineEdit = new QLineEdit(table);
    p_filePathLineEdit->setPlaceholderText("Select catalog file (.cat, .xo, .out)...");
    p_filePathLineEdit->setMinimumWidth(250); // Ensure adequate width

    p_browseButton = new QToolButton(table);
    p_browseButton->setIcon(ThemeColors::createThemedIcon(
        ":/icons/folder-open.svg", ThemeColors::IconSecondary, this));
    p_browseButton->setToolTip("Browse for catalog file");

    table->addSettingRow("File", p_filePathLineEdit, p_browseButton);

    // Parsed-file detail rows. They live directly in the config table
    // and are hidden via setRowHidden() until a valid catalog is
    // loaded (replacing the old object-named details frame).
    p_formatLabel = new QLabel("-", table);
    p_moleculeLabel = new QLabel("-", table);
    p_transitionCountLabel = new QLabel("-", table);
    p_frequencyRangeLabel = new QLabel("-", table);

    auto configureDetailLabel = [](QLabel* label) {
        label->setWordWrap(false);
        label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        styleSubtleLabel(label, ThemeColors::SubtleText);
    };

    configureDetailLabel(p_formatLabel);
    configureDetailLabel(p_moleculeLabel);
    configureDetailLabel(p_transitionCountLabel);
    configureDetailLabel(p_frequencyRangeLabel);

    d_fileDetailRows = {
        table->addSettingRow("Format", p_formatLabel),
        table->addSettingRow("Molecule", p_moleculeLabel),
        table->addSettingRow("Transitions", p_transitionCountLabel),
        table->addSettingRow("Range", p_frequencyRangeLabel)
    };

    // Initially hidden - shown by updateFileInfo() once a file loads.
    for (int r : d_fileDetailRows)
        table->setRowHidden(r, true);
}

void CatalogOverlayWidget::populateSourceFileSettingsRows(SettingsTable *table)
{
    table->addSectionRow("Filtering");

    // Save range only option (source-dependent)
    p_saveRangeOnlyCheckBox = new QCheckBox("(recommended)", table);
    p_saveRangeOnlyCheckBox->setChecked(get(BC::Key::CatalogWidget::saveRangeOnly, DEFAULT_SAVE_RANGE_ONLY));
    p_saveRangeOnlyCheckBox->setToolTip("When enabled, only saves catalog transitions within the frequency range, reducing file size and improving performance.");
    table->addSettingRow("Limit to Range", p_saveRangeOnlyCheckBox);

    // Filtering frequency range spinboxes
    p_filterMinFreqSpinBox = new QDoubleSpinBox(table);
    configureSpinBox(p_filterMinFreqSpinBox,
                     BC::Key::CatalogWidget::freqMin, BC::Key::CatalogWidget::freqMax,
                     BC::Key::CatalogWidget::freqDecimals, BC::Key::CatalogWidget::freqStep,
                     0.0, 10000000.0, 3, 1.0);
    p_filterMinFreqSpinBox->setSuffix(" MHz");

    p_filterMaxFreqSpinBox = new QDoubleSpinBox(table);
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

    auto rangeCell = new QWidget(table);
    auto rangeRow = new QHBoxLayout(rangeCell);
    rangeRow->setContentsMargins(0, 0, 0, 0);
    rangeRow->addWidget(p_filterMinFreqSpinBox);
    rangeRow->addWidget(new QLabel("to", rangeCell));
    rangeRow->addWidget(p_filterMaxFreqSpinBox);
    rangeRow->addStretch();
    table->addSettingRow("Filter Range", rangeCell);
}

void CatalogOverlayWidget::populateTypeSpecificRows(SettingsTable *table)
{
    // Retitle the base-provided tier heading.
    table->setSectionTitle(d_typeSpecificSection, "Catalog Settings");

    // Convolution enable is a checkable section row nested in the
    // shared table. Its bound rows (the Line Shape / Frequency Range
    // sub-headings, value rows, and the Convolve button) collapse with
    // the section, reproducing the old checkable-QGroupBox behavior.
    // The convolution state machine talks to the section checkbox
    // through isConvolutionEnabled()/setConvolutionEnabled() and the
    // convolutionEnabledChanged() relay. setSectionVisible() on the
    // enclosing tier is collapse-aware, so showing the tier does not
    // fight this collapse.
    d_convolutionSection = table->addCheckableSectionRow(
        "Convolution Enabled",
        get(BC::Key::CatalogWidget::convolutionEnabled, DEFAULT_CONVOLUTION_ENABLED),
        &p_convolutionSectionBox);

    const int firstConvRow = table->rowCount();

    // --- Line Shape ---
    table->addSectionRow("Line Shape");

    p_lineshapeComboBox = new QComboBox(table);
    p_lineshapeComboBox->addItems({"Lorentzian", "Gaussian"});
    p_lineshapeComboBox->setCurrentIndex(get(BC::Key::CatalogWidget::lineshapeType, DEFAULT_LINESHAPE_TYPE));
    table->addSettingRow("Type", p_lineshapeComboBox);

    p_linewidthSpinBox = new QDoubleSpinBox(table);
    configureSpinBox(p_linewidthSpinBox,
                     BC::Key::CatalogWidget::linewidthMin, BC::Key::CatalogWidget::linewidthMax,
                     BC::Key::CatalogWidget::linewidthDecimals, BC::Key::CatalogWidget::linewidthStep,
                     0.1, 10000.0, 1, 10.0);
    p_linewidthSpinBox->setSuffix(" kHz");
    p_linewidthSpinBox->setValue(get(BC::Key::CatalogWidget::linewidthKHz, DEFAULT_LINEWIDTH));
    p_linewidthSpinBox->setKeyboardTracking(false);
    table->addSettingRow("Width (FWHM)", p_linewidthSpinBox);

    // --- Frequency Range & Resolution ---
    table->addSectionRow("Frequency Range & Resolution");

    p_convMinFreqSpinBox = new QDoubleSpinBox(table);
    configureSpinBox(p_convMinFreqSpinBox,
                     BC::Key::CatalogWidget::freqMin, BC::Key::CatalogWidget::freqMax,
                     BC::Key::CatalogWidget::freqDecimals, BC::Key::CatalogWidget::freqStep,
                     0.0, 10000000.0, 3, 1.0);
    p_convMinFreqSpinBox->setSuffix(" MHz");
    p_convMinFreqSpinBox->setKeyboardTracking(false);

    p_convMaxFreqSpinBox = new QDoubleSpinBox(table);
    configureSpinBox(p_convMaxFreqSpinBox,
                     BC::Key::CatalogWidget::freqMin, BC::Key::CatalogWidget::freqMax,
                     BC::Key::CatalogWidget::freqDecimals, BC::Key::CatalogWidget::freqStep,
                     0.0, 10000000.0, 3, 1.0);
    p_convMaxFreqSpinBox->setSuffix(" MHz");
    p_convMaxFreqSpinBox->setKeyboardTracking(false);

    p_numPointsSpinBox = new QSpinBox(table);
    p_numPointsSpinBox->setMinimum(get(BC::Key::CatalogWidget::numPointsMin, 100));
    p_numPointsSpinBox->setMaximum(get(BC::Key::CatalogWidget::numPointsMax, 10000000));
    p_numPointsSpinBox->setSingleStep(get(BC::Key::CatalogWidget::numPointsStep, 100));
    p_numPointsSpinBox->setKeyboardTracking(false);

    p_spacingDisplayLabel = new QLabel("0.000 MHz", table);
    styleSubtleLabel(p_spacingDisplayLabel, ThemeColors::SubtleText);

    auto rangeCell = new QWidget(table);
    auto rangeRow = new QHBoxLayout(rangeCell);
    rangeRow->setContentsMargins(0, 0, 0, 0);
    rangeRow->addWidget(p_convMinFreqSpinBox);
    rangeRow->addWidget(new QLabel("to", rangeCell));
    rangeRow->addWidget(p_convMaxFreqSpinBox);
    rangeRow->addStretch();
    table->addSettingRow("Range", rangeCell);

    table->addSettingRow("Points", p_numPointsSpinBox);
    table->addSettingRow("Spacing", p_spacingDisplayLabel);

    // Action control: a trailing bound row so the Convolve button
    // collapses with the section exactly as the old content widget did.
    p_convolveButton = new QPushButton("Convolve", table);
    p_convolveButton->setIcon(ThemeColors::createThemedIcon(":/icons/calculator.svg", ThemeColors::IconPrimary, this));
    p_convolveButton->setEnabled(false);
    p_convolveButton->setMinimumHeight(30);
    table->addSettingRow("Action", p_convolveButton);

    const int lastConvRow = table->rowCount();
    d_convolutionRows.clear();
    for (int r = firstConvRow; r < lastConvRow; ++r)
        d_convolutionRows.append(r);
    table->bindSectionRows(d_convolutionSection, d_convolutionRows);

    // Relay the section checkbox toggle through one internal slot, the
    // single point the convolution state machine connects to.
    connect(p_convolutionSectionBox, &QCheckBox::toggled, this,
            [this](bool on) { emit convolutionEnabledChanged(on); });

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
    current.enabled = isConvolutionEnabled();
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
    
    if (d_convolutionInProgress) {
        // Show cancel button during processing
        p_convolveButton->setText("Cancel");
        p_convolveButton->setIcon(ThemeColors::createThemedIcon(":/icons/x-mark.svg", ThemeColors::StatusError, this));
        p_convolveButton->setToolTip("Cancel ongoing convolution");
        p_convolveButton->setEnabled(true);
        return;
    }
    
    // Normal convolution button state
    p_convolveButton->setText("Convolve");
    p_convolveButton->setIcon(ThemeColors::createThemedIcon(":/icons/calculator.svg", ThemeColors::IconPrimary, this));
    p_convolveButton->setToolTip("Apply convolution to catalog data");
    
    ConvolutionState current = getCurrentConvolutionState();
    
    // Enable button if:
    // 1. No convolution has been performed yet, OR
    // 2. Current settings differ from last performed convolution
    bool shouldEnable = !d_lastConvolutionState.convolutionPerformed || 
                        (current != d_lastConvolutionState);
    
    // Only enable if convolution is enabled and we have valid data
    shouldEnable = shouldEnable && current.enabled && isDataValid();
    
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
    if (d_convolutionInProgress) {
        // Cancel operation
        cancelPendingConvolution();
        return;
    }
    
    if (!isDataValid()) {
        return;
    }
    
    // Store current settings as the last performed convolution
    d_lastConvolutionState = getCurrentConvolutionState();
    d_lastConvolutionState.convolutionPerformed = true;
    
    // Update button state (should show "Cancel")
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
        styleSubtleLabel(p_moleculeLabel, ThemeColors::SubtleText, true);
    }

    // Set defaults from settings for creation context
    if (p_linewidthSpinBox) {
        double defaultLinewidth = get(BC::Key::CatalogWidget::linewidthKHz, 100.0); // Default 100 kHz
        p_linewidthSpinBox->setValue(defaultLinewidth);
    }

    // Make convolution button more prominent
    if (p_convolveButton) {
        QFont bf = p_convolveButton->font();
        bf.setBold(true);
        p_convolveButton->setFont(bf);
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
        QFont bf = p_convolveButton->font();
        bf.setBold(false);
        p_convolveButton->setFont(bf);
    }
}
