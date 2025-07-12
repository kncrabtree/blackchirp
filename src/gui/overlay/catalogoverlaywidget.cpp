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

CatalogOverlayWidget::CatalogOverlayWidget(QWidget *parent)
    : OverlayTypeSpecificWidget(parent), SettingsStorage(BC::Key::CatalogWidget::key),
      p_sourceFileConfigWidget(nullptr),
      p_sourceFileSettingsWidget(nullptr),
      p_overlaySettingsWidget(nullptr),
      d_fileValid(false),
      d_hasFtData(false),
      d_ftYMax(1.0),
      d_convolutionInProgress(false)
{
    setupUI();
    setupConnections();
}

CatalogOverlayWidget::~CatalogOverlayWidget() = default;

void CatalogOverlayWidget::setupForCreation()
{
    d_context = Context::Creation;
    d_overlay.reset();
    
    // Get current Ft data for intelligent defaults
    auto ftmwParent = qobject_cast<FtmwViewWidget*>(parent());
    if (ftmwParent) {
        Ft mainFt = ftmwParent->getMainPlotFt();
        d_hasFtData = !mainFt.isEmpty();
        if (d_hasFtData) {
            d_ftYMax = mainFt.yMax();
            
            // Initialize frequency range with Ft data
            auto xRange = mainFt.xRange();
            p_minFreqSpinBox->setValue(xRange.first);
            p_maxFreqSpinBox->setValue(xRange.second);
        }
    }
    
    // Load default settings for creation context
    loadSettings();
    
    // Initialize defaults
    resetToDefaults();
    
    updateConvolutionControls();
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
            
            // Set source file path
            p_filePathLineEdit->setText(d_filePath);
            
            // Load convolution settings
            p_convolutionEnabledCheckBox->setChecked(catalogOverlay->convolutionEnabled());
            p_lineshapeComboBox->setCurrentIndex(static_cast<int>(catalogOverlay->lineshapeType()));
            p_linewidthSpinBox->setValue(catalogOverlay->linewidth());
            p_minFreqSpinBox->setValue(catalogOverlay->convolutionMinFreq());
            p_maxFreqSpinBox->setValue(catalogOverlay->convolutionMaxFreq());
            p_pointSpacingSpinBox->setValue(catalogOverlay->pointSpacing());
        }
    }
    
    updateFileInfo();
    updateConvolutionControls();
}

std::shared_ptr<OverlayBase> CatalogOverlayWidget::createOverlay() const
{
    if (d_context != Context::Creation) {
        return nullptr;
    }
    
    if (!isDataValid()) {
        return nullptr;
    }
    
    auto overlay = std::make_shared<CatalogOverlay>();
    overlay->setCatalogData(d_catalogData);
    overlay->setSourceFile(d_filePath);
    
    // Apply convolution settings
    overlay->setConvolutionEnabled(p_convolutionEnabledCheckBox->isChecked());
    overlay->setLineshapeType(static_cast<CatalogOverlay::LineshapeType>(p_lineshapeComboBox->currentIndex()));
    overlay->setLinewidth(p_linewidthSpinBox->value());
    overlay->setConvolutionFreqRange(p_minFreqSpinBox->value(), p_maxFreqSpinBox->value());
    overlay->setPointSpacing(p_pointSpacingSpinBox->value());
    
    // Apply curve appearance preset based on convolution mode
    QString presetName;
    if (p_convolutionEnabledCheckBox->isChecked()) {
        presetName = "Curve - Secondary";  // Smooth curve for convolved data
    } else {
        presetName = "Stem - Secondary";   // Stem plot for discrete transitions
    }
    
    auto presetManager = CurveAppearancePresetManager::instance();
    if (presetManager && presetManager->hasPreset(presetName)) {
        auto preset = presetManager->getPreset(presetName);
        overlay->setCurveAppearanceMetadata(preset.appearance);
    }
    
    return overlay;
}

void CatalogOverlayWidget::applyToOverlay(std::shared_ptr<OverlayBase> overlay) const
{
    if (!overlay || d_context != Context::Settings) {
        return;
    }
    
    auto catalogOverlay = std::dynamic_pointer_cast<CatalogOverlay>(overlay);
    if (!catalogOverlay) {
        return;
    }
    
    // Apply current settings to the overlay
    catalogOverlay->setCatalogData(d_catalogData);
    catalogOverlay->setSourceFile(d_filePath);
    
    // Apply convolution settings
    catalogOverlay->setConvolutionEnabled(p_convolutionEnabledCheckBox->isChecked());
    catalogOverlay->setLineshapeType(static_cast<CatalogOverlay::LineshapeType>(p_lineshapeComboBox->currentIndex()));
    catalogOverlay->setLinewidth(p_linewidthSpinBox->value());
    catalogOverlay->setConvolutionFreqRange(p_minFreqSpinBox->value(), p_maxFreqSpinBox->value());
    catalogOverlay->setPointSpacing(p_pointSpacingSpinBox->value());
}

bool CatalogOverlayWidget::validateSettings(QString &errorMessage) const
{
    if (!d_fileValid) {
        errorMessage = "Please select a valid catalog file.";
        return false;
    }
    
    if (d_catalogData.isEmpty()) {
        errorMessage = "Selected catalog file contains no transitions.";
        return false;
    }
    
    return validateConvolutionSettings(errorMessage);
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
    
    p_filePathLineEdit->setText(path);
    onFilePathChanged();
}

bool CatalogOverlayWidget::validateSourceFile(QString &errorMessage)
{
    QString path = p_filePathLineEdit->text().trimmed();
    
    if (path.isEmpty()) {
        errorMessage = "Please select a catalog file.";
        d_fileValid = false;
        return false;
    }
    
    if (!QFile::exists(path)) {
        errorMessage = QString("Catalog file does not exist: %1").arg(path);
        d_fileValid = false;
        return false;
    }
    
    // Try to parse the file
    auto registry = CatalogParserRegistry::instance();
    auto parser = registry->findParser(path);
    
    if (!parser) {
        errorMessage = QString("No suitable parser found for file: %1").arg(path);
        d_fileValid = false;
        return false;
    }
    
    try {
        CatalogData testData = parser->parse(path);
        if (testData.isEmpty()) {
            errorMessage = QString("Catalog file contains no valid transitions: %1").arg(path);
            d_fileValid = false;
            return false;
        }
        d_fileValid = true;
        return true;
    } catch (const std::exception &e) {
        errorMessage = QString("Error parsing catalog file: %1").arg(e.what());
        d_fileValid = false;
        return false;
    } catch (...) {
        errorMessage = QString("Unknown error parsing catalog file: %1").arg(path);
        d_fileValid = false;
        return false;
    }
}

void CatalogOverlayWidget::resetToDefaults()
{
    // Clear file selection
    p_filePathLineEdit->clear();
    d_filePath.clear();
    d_fileValid = false;
    d_catalogData = CatalogData();
    
    // Reset convolution settings to defaults/settings
    p_convolutionEnabledCheckBox->setChecked(get(BC::Key::CatalogWidget::convolutionEnabled, DEFAULT_CONVOLUTION_ENABLED));
    p_lineshapeComboBox->setCurrentIndex(get(BC::Key::CatalogWidget::lineshapeType, DEFAULT_LINESHAPE_TYPE));
    p_linewidthSpinBox->setValue(get(BC::Key::CatalogWidget::linewidthKHz, DEFAULT_LINEWIDTH));
    p_pointSpacingSpinBox->setValue(get(BC::Key::CatalogWidget::pointSpacingMHz, DEFAULT_POINT_SPACING));
    p_saveRangeOnlyCheckBox->setChecked(get(BC::Key::CatalogWidget::saveRangeOnly, DEFAULT_SAVE_RANGE_ONLY));
    
    // Reset frequency range (will be set from Ft data if available)
    if (!d_hasFtData) {
        p_minFreqSpinBox->setValue(get(BC::Key::CatalogWidget::minFreqMHz, DEFAULT_MIN_FREQ));
        p_maxFreqSpinBox->setValue(get(BC::Key::CatalogWidget::maxFreqMHz, DEFAULT_MAX_FREQ));
    }
    
    updateFileInfo();
    updateConvolutionControls();
}

QHash<QString, QVariant> CatalogOverlayWidget::getSettingsHash() const
{
    QHash<QString, QVariant> settings;
    
    // File selection settings
    settings["filePath"] = d_filePath;
    settings["fileValid"] = d_fileValid;
    
    // Convolution settings (including frequency range for convolution)
    settings["convolutionEnabled"] = p_convolutionEnabledCheckBox->isChecked();
    settings["lineshapeType"] = p_lineshapeComboBox->currentIndex();
    settings["linewidthKHz"] = p_linewidthSpinBox->value();
    settings["pointSpacingMHz"] = p_pointSpacingSpinBox->value();
    settings["minFreqMHz"] = p_minFreqSpinBox->value();
    settings["maxFreqMHz"] = p_maxFreqSpinBox->value();
    
    // Other settings
    settings["saveRangeOnly"] = p_saveRangeOnlyCheckBox->isChecked();
    
    // Catalog data fingerprint (for detecting data changes)
    if (d_catalogData.size() > 0) {
        settings["catalogSize"] = d_catalogData.size();
        settings["catalogSourceProgram"] = d_catalogData.sourceProgram();
        settings["catalogMoleculeName"] = d_catalogData.moleculeName();
    }
    
    return settings;
}

QWidget* CatalogOverlayWidget::getSourceFileConfigWidget()
{
    return p_sourceFileConfigWidget;
}

QWidget* CatalogOverlayWidget::getSourceFileSettingsWidget()
{
    return p_sourceFileSettingsWidget;
}

QWidget* CatalogOverlayWidget::getOverlaySettingsWidget()
{
    return p_overlaySettingsWidget;
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
        p_filePathLineEdit->setText(filePath);
        onFilePathChanged();
    }
}

void CatalogOverlayWidget::onFilePathChanged()
{
    d_filePath = p_filePathLineEdit->text().trimmed();
    
    if (d_filePath.isEmpty()) {
        d_fileValid = false;
        d_catalogData = CatalogData();
        updateFileInfo();
        emit sourceFileChanged();
        emit dataValidityChanged(isDataValid());
        return;
    }
    
    emit progressOperationStarted("Loading catalog file...");
    
    loadCatalogFile(d_filePath);
    updateFileInfo();
    
    // Auto-set frequency range if catalog loaded successfully and ranges are at defaults
    if (d_fileValid && qAbs(p_minFreqSpinBox->value() - DEFAULT_MIN_FREQ) < 1e-6 &&
        qAbs(p_maxFreqSpinBox->value() - DEFAULT_MAX_FREQ) < 1e-6) {
        autoSetFrequencyRange();
    }
    
    emit progressOperationFinished();
    emit sourceFileChanged();
    emit dataValidityChanged(isDataValid());
    emit settingsChanged();
}

void CatalogOverlayWidget::onConvolutionEnabledToggled(bool enabled)
{
    Q_UNUSED(enabled);
    updateConvolutionControls();
    emit settingsChanged();
}

void CatalogOverlayWidget::onLineshapeTypeChanged(int index)
{
    Q_UNUSED(index);
    emit settingsChanged();
}

void CatalogOverlayWidget::onConvolutionSettingsChanged()
{
    // Recalculate default Y scale when convolution settings change
    if (d_context == Context::Creation) {
        calculateDefaultYScale();
    }
    
    // In settings context (including preview mode), trigger background convolution for real-time updates
    if (d_context == Context::Settings && p_convolutionEnabledCheckBox->isChecked()) {
        triggerBackgroundConvolution();
    }
    
    emit settingsChanged();
}

void CatalogOverlayWidget::onAutoRangeClicked()
{
    autoSetFrequencyRange();
    emit settingsChanged();
}

void CatalogOverlayWidget::onSaveRangeOnlyToggled(bool enabled)
{
    Q_UNUSED(enabled);
    emit settingsChanged();
}

void CatalogOverlayWidget::setupUI()
{
    // Load settings first to configure spinboxes
    loadSettings();
    
    // Create main layout
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);
    
    // Create the three-tier widgets
    setupFileSelectionUI();
    setupSourceFileSettingsUI();
    setupConvolutionSettingsUI();
    
    // Add all widgets to main layout (they will be reparented by UnifiedOverlayWidget)
    mainLayout->addWidget(p_sourceFileConfigWidget);
    mainLayout->addWidget(p_sourceFileSettingsWidget);
    mainLayout->addWidget(p_overlaySettingsWidget);
}

void CatalogOverlayWidget::setupConnections()
{
    connect(p_browseButton, &QToolButton::clicked, this, &CatalogOverlayWidget::onBrowseButtonClicked);
    connect(p_filePathLineEdit, &QLineEdit::textChanged, this, &CatalogOverlayWidget::onFilePathChanged);
    connect(p_convolutionEnabledCheckBox, &QCheckBox::toggled, this, &CatalogOverlayWidget::onConvolutionEnabledToggled);
    connect(p_lineshapeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CatalogOverlayWidget::onLineshapeTypeChanged);
    connect(p_linewidthSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayWidget::onConvolutionSettingsChanged);
    connect(p_minFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayWidget::onConvolutionSettingsChanged);
    connect(p_maxFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayWidget::onConvolutionSettingsChanged);
    connect(p_pointSpacingSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayWidget::onConvolutionSettingsChanged);
    connect(p_autoRangeButton, &QPushButton::clicked, this, &CatalogOverlayWidget::onAutoRangeClicked);
    connect(p_saveRangeOnlyCheckBox, &QCheckBox::toggled, this, &CatalogOverlayWidget::onSaveRangeOnlyToggled);
    
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
    // Settings are loaded in the UI setup methods and resetToDefaults()
    // This is mainly a placeholder for consistency with the interface
}

void CatalogOverlayWidget::saveSettings()
{
    // Save current file path
    set(BC::Key::CatalogWidget::lastFilePath, d_filePath);
    
    // Save convolution settings
    set(BC::Key::CatalogWidget::convolutionEnabled, p_convolutionEnabledCheckBox->isChecked());
    set(BC::Key::CatalogWidget::lineshapeType, p_lineshapeComboBox->currentIndex());
    set(BC::Key::CatalogWidget::linewidthKHz, p_linewidthSpinBox->value());
    set(BC::Key::CatalogWidget::minFreqMHz, p_minFreqSpinBox->value());
    set(BC::Key::CatalogWidget::maxFreqMHz, p_maxFreqSpinBox->value());
    set(BC::Key::CatalogWidget::pointSpacingMHz, p_pointSpacingSpinBox->value());
    set(BC::Key::CatalogWidget::saveRangeOnly, p_saveRangeOnlyCheckBox->isChecked());
}

void CatalogOverlayWidget::setupFileSelectionUI()
{
    p_sourceFileConfigWidget = new QWidget(this);
    QVBoxLayout *configLayout = new QVBoxLayout(p_sourceFileConfigWidget);
    configLayout->setContentsMargins(0, 0, 0, 0);
    
    p_fileSelectionGroup = new QGroupBox("Catalog File Selection", p_sourceFileConfigWidget);
    QFormLayout *fileLayout = new QFormLayout(p_fileSelectionGroup);
    
    // File path selection
    QHBoxLayout *pathLayout = new QHBoxLayout();
    p_filePathLineEdit = new QLineEdit(p_fileSelectionGroup);
    p_filePathLineEdit->setPlaceholderText("Select catalog file (.cat, .xo, .out)...");
    p_browseButton = new QToolButton(p_fileSelectionGroup);
    p_browseButton->setText("Browse...");
    p_browseButton->setMinimumSize(80, 0);
    
    pathLayout->addWidget(p_filePathLineEdit);
    pathLayout->addWidget(p_browseButton);
    fileLayout->addRow("File:", pathLayout);
    
    // File information display
    p_formatLabel = new QLabel("-", p_fileSelectionGroup);
    p_moleculeLabel = new QLabel("-", p_fileSelectionGroup);
    p_transitionCountLabel = new QLabel("-", p_fileSelectionGroup);
    p_frequencyRangeLabel = new QLabel("-", p_fileSelectionGroup);
    
    fileLayout->addRow("Format:", p_formatLabel);
    fileLayout->addRow("Molecule:", p_moleculeLabel);
    fileLayout->addRow("Transitions:", p_transitionCountLabel);
    fileLayout->addRow("Frequency Range:", p_frequencyRangeLabel);
    
    configLayout->addWidget(p_fileSelectionGroup);
}

void CatalogOverlayWidget::setupSourceFileSettingsUI()
{
    p_sourceFileSettingsWidget = new QWidget(this);
    QVBoxLayout *settingsLayout = new QVBoxLayout(p_sourceFileSettingsWidget);
    settingsLayout->setContentsMargins(0, 0, 0, 0);
    
    p_sourceFileGroup = new QGroupBox("Source File Settings", p_sourceFileSettingsWidget);
    QFormLayout *sourceLayout = new QFormLayout(p_sourceFileGroup);
    
    // Save range only option (source-dependent)
    p_saveRangeOnlyCheckBox = new QCheckBox("Save only transitions within frequency range (recommended)");
    p_saveRangeOnlyCheckBox->setChecked(get(BC::Key::CatalogWidget::saveRangeOnly, DEFAULT_SAVE_RANGE_ONLY));
    p_saveRangeOnlyCheckBox->setToolTip("When enabled, only saves catalog transitions within the frequency range, reducing file size and improving performance.");
    sourceLayout->addRow(p_saveRangeOnlyCheckBox);
    
    // Auto-range button (source-dependent - needs catalog data)
    p_autoRangeButton = new QPushButton("Auto-set frequency range from catalog");
    p_autoRangeButton->setEnabled(false); // Will be enabled when catalog is loaded
    sourceLayout->addRow("Frequency Range:", p_autoRangeButton);
    
    settingsLayout->addWidget(p_sourceFileGroup);
}

void CatalogOverlayWidget::setupConvolutionSettingsUI()
{
    p_overlaySettingsWidget = new QWidget(this);
    QVBoxLayout *overlayLayout = new QVBoxLayout(p_overlaySettingsWidget);
    overlayLayout->setContentsMargins(0, 0, 0, 0);
    
    p_convolutionGroup = new QGroupBox("Convolution Settings", p_overlaySettingsWidget);
    QFormLayout *convLayout = new QFormLayout(p_convolutionGroup);
    
    // Enable convolution
    p_convolutionEnabledCheckBox = new QCheckBox("Enable convolution");
    p_convolutionEnabledCheckBox->setChecked(get(BC::Key::CatalogWidget::convolutionEnabled, DEFAULT_CONVOLUTION_ENABLED));
    convLayout->addRow(p_convolutionEnabledCheckBox);
    
    // Lineshape type
    p_lineshapeComboBox = new QComboBox(p_convolutionGroup);
    p_lineshapeComboBox->addItems({"Lorentzian", "Gaussian"});
    p_lineshapeComboBox->setCurrentIndex(get(BC::Key::CatalogWidget::lineshapeType, DEFAULT_LINESHAPE_TYPE));
    convLayout->addRow("Lineshape:", p_lineshapeComboBox);
    
    // Linewidth
    p_linewidthSpinBox = new QDoubleSpinBox(p_convolutionGroup);
    configureSpinBox(p_linewidthSpinBox, 
                     BC::Key::CatalogWidget::linewidthMin, BC::Key::CatalogWidget::linewidthMax,
                     BC::Key::CatalogWidget::linewidthDecimals, BC::Key::CatalogWidget::linewidthStep,
                     0.1, 10000.0, 1, 10.0);
    p_linewidthSpinBox->setSuffix(" kHz");
    p_linewidthSpinBox->setValue(get(BC::Key::CatalogWidget::linewidthKHz, DEFAULT_LINEWIDTH));
    convLayout->addRow("Linewidth (FWHM):", p_linewidthSpinBox);
    
    // Frequency range
    QHBoxLayout *freqRangeLayout = new QHBoxLayout();
    p_minFreqSpinBox = new QDoubleSpinBox(p_convolutionGroup);
    configureSpinBox(p_minFreqSpinBox,
                     BC::Key::CatalogWidget::freqMin, BC::Key::CatalogWidget::freqMax,
                     BC::Key::CatalogWidget::freqDecimals, BC::Key::CatalogWidget::freqStep,
                     0.0, 100000.0, 3, 1.0);
    p_minFreqSpinBox->setSuffix(" MHz");
    p_minFreqSpinBox->setValue(get(BC::Key::CatalogWidget::minFreqMHz, DEFAULT_MIN_FREQ));
    
    p_maxFreqSpinBox = new QDoubleSpinBox(p_convolutionGroup);
    configureSpinBox(p_maxFreqSpinBox,
                     BC::Key::CatalogWidget::freqMin, BC::Key::CatalogWidget::freqMax,
                     BC::Key::CatalogWidget::freqDecimals, BC::Key::CatalogWidget::freqStep,
                     0.0, 100000.0, 3, 1.0);
    p_maxFreqSpinBox->setSuffix(" MHz");
    p_maxFreqSpinBox->setValue(get(BC::Key::CatalogWidget::maxFreqMHz, DEFAULT_MAX_FREQ));
    
    freqRangeLayout->addWidget(p_minFreqSpinBox);
    freqRangeLayout->addWidget(new QLabel("to"));
    freqRangeLayout->addWidget(p_maxFreqSpinBox);
    convLayout->addRow("Frequency Range:", freqRangeLayout);
    
    // Point spacing
    p_pointSpacingSpinBox = new QDoubleSpinBox(p_convolutionGroup);
    configureSpinBox(p_pointSpacingSpinBox,
                     BC::Key::CatalogWidget::pointSpacingMin, BC::Key::CatalogWidget::pointSpacingMax,
                     BC::Key::CatalogWidget::pointSpacingDecimals, BC::Key::CatalogWidget::pointSpacingStep,
                     0.001, 1.0, 3, 0.001);
    p_pointSpacingSpinBox->setSuffix(" MHz");
    p_pointSpacingSpinBox->setValue(get(BC::Key::CatalogWidget::pointSpacingMHz, DEFAULT_POINT_SPACING));
    convLayout->addRow("Point Spacing:", p_pointSpacingSpinBox);
    
    overlayLayout->addWidget(p_convolutionGroup);
}

void CatalogOverlayWidget::loadCatalogFile(const QString &filePath)
{
    d_fileValid = false;
    d_catalogData = CatalogData();
    
    if (!QFile::exists(filePath)) {
        return;
    }
    
    auto registry = CatalogParserRegistry::instance();
    auto parser = registry->findParser(filePath);
    
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
}

void CatalogOverlayWidget::updateFileInfo()
{
    if (!d_fileValid || d_catalogData.isEmpty()) {
        p_formatLabel->setText("-");
        p_moleculeLabel->setText("-");
        p_transitionCountLabel->setText("-");
        p_frequencyRangeLabel->setText("-");
        p_autoRangeButton->setEnabled(false);
        return;
    }
    
    // Show format and molecule info
    p_formatLabel->setText(d_catalogData.sourceProgram());
    p_moleculeLabel->setText(d_catalogData.moleculeName());
    p_transitionCountLabel->setText(QString::number(d_catalogData.size()));
    p_autoRangeButton->setEnabled(true);
    
    // Calculate and set reasonable default yscale (only in creation context)
    if (d_context == Context::Creation) {
        calculateDefaultYScale();
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
    }
}

void CatalogOverlayWidget::updateConvolutionControls()
{
    bool enabled = p_convolutionEnabledCheckBox->isChecked();
    
    p_lineshapeComboBox->setEnabled(enabled);
    p_linewidthSpinBox->setEnabled(enabled);
    p_minFreqSpinBox->setEnabled(enabled);
    p_maxFreqSpinBox->setEnabled(enabled);
    p_pointSpacingSpinBox->setEnabled(enabled);
}

void CatalogOverlayWidget::autoSetFrequencyRange()
{
    double rangeMin, rangeMax;
    
    // Prefer Ft data range if available, otherwise use settings defaults
    if (d_hasFtData) {
        auto ftmwParent = qobject_cast<FtmwViewWidget*>(parent());
        if (ftmwParent) {
            Ft mainFt = ftmwParent->getMainPlotFt();
            auto xRange = mainFt.xRange();
            rangeMin = xRange.first;
            rangeMax = xRange.second;
        } else {
            // Fallback to settings values
            rangeMin = get(BC::Key::CatalogWidget::minFreqMHz, DEFAULT_MIN_FREQ);
            rangeMax = get(BC::Key::CatalogWidget::maxFreqMHz, DEFAULT_MAX_FREQ);
        }
    } else {
        // Use settings values when no Ft data is available
        rangeMin = get(BC::Key::CatalogWidget::minFreqMHz, DEFAULT_MIN_FREQ);
        rangeMax = get(BC::Key::CatalogWidget::maxFreqMHz, DEFAULT_MAX_FREQ);
    }
    
    p_minFreqSpinBox->setValue(rangeMin);
    p_maxFreqSpinBox->setValue(rangeMax);
}

void CatalogOverlayWidget::calculateDefaultYScale()
{
    // This would need access to OverlayBaseOptionsWidget, which will be handled
    // by the UnifiedOverlayWidget when it integrates this type-specific widget
    // For now, this is a placeholder that maintains the interface
}

bool CatalogOverlayWidget::validateConvolutionSettings(QString &errorMessage) const
{
    if (!p_convolutionEnabledCheckBox->isChecked()) {
        return true; // No validation needed if convolution disabled
    }
    
    double minFreq = p_minFreqSpinBox->value();
    double maxFreq = p_maxFreqSpinBox->value();
    
    if (minFreq >= maxFreq) {
        errorMessage = "Minimum frequency must be less than maximum frequency.";
        return false;
    }
    
    if (p_linewidthSpinBox->value() <= 0) {
        errorMessage = "Linewidth must be positive.";
        return false;
    }
    
    if (p_pointSpacingSpinBox->value() <= 0) {
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
        p_convolutionEnabledCheckBox->isChecked(),
        static_cast<CatalogOverlay::LineshapeType>(p_lineshapeComboBox->currentIndex()),
        p_linewidthSpinBox->value(),
        p_minFreqSpinBox->value(),
        p_maxFreqSpinBox->value(),
        p_pointSpacingSpinBox->value(),
        this
    );
    
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
    if (result && d_overlay) {
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
    
    emit progressOperationFinished();
}