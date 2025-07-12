#include "catalogoverlaydialog.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QStandardPaths>
#include <QCloseEvent>
#include <QSplitter>
#include <QDebug>

#include <gui/widget/ftmwviewwidget.h>
#include <gui/plot/curveappearancepresetmanager.h>

CatalogOverlayDialog::CatalogOverlayDialog(FtmwViewWidget *parent)
    : OverlayConfigDialog(parent), SettingsStorage(BC::Key::CatalogDialog::key), 
      d_fileValid(false), d_hasFtData(false), d_ftYMax(1.0)
{
    setWindowTitle("Create Catalog Overlay");
    
    // Get current Ft data for intelligent defaults
    if (parent) {
        Ft mainFt = parent->getMainPlotFt();
        d_hasFtData = !mainFt.isEmpty();
        if (d_hasFtData) {
            d_ftYMax = mainFt.yMax();
        }
    }
}

CatalogOverlayDialog::~CatalogOverlayDialog() = default;

std::shared_ptr<OverlayBase> CatalogOverlayDialog::createTypeSpecificOverlay() const
{
    if (!d_fileValid || d_catalogData.isEmpty()) {
        return nullptr;
    }
    
    auto overlay = std::make_shared<CatalogOverlay>();
    overlay->setCatalogData(d_catalogData);
    
    // Set the source file path (type-specific behavior)
    overlay->setSourceFile(d_filePath);
    
    return overlay;
}

void CatalogOverlayDialog::configureTypeSpecificOverlay(std::shared_ptr<OverlayBase> overlay) const
{
    // Cast to specific type for type-specific configuration
    auto catalogOverlay = std::static_pointer_cast<CatalogOverlay>(overlay);
    
    // Apply convolution settings (type-specific configuration)
    catalogOverlay->setConvolutionEnabled(p_convolutionEnabledCheckBox->isChecked());
    catalogOverlay->setLineshapeType(static_cast<CatalogOverlay::LineshapeType>(p_lineshapeComboBox->currentIndex()));
    catalogOverlay->setLinewidth(p_linewidthSpinBox->value());
    catalogOverlay->setConvolutionFreqRange(p_minFreqSpinBox->value(), p_maxFreqSpinBox->value());
    catalogOverlay->setPointSpacing(p_pointSpacingSpinBox->value());
    
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
}

void CatalogOverlayDialog::accept()
{
    saveSettings();
    OverlayConfigDialog::accept();
}

void CatalogOverlayDialog::reject()
{
    // Discard any unsaved settings changes
    discardChanges(true);
    OverlayConfigDialog::reject();
}

void CatalogOverlayDialog::closeEvent(QCloseEvent *event)
{
    // Only save geometry on close
    set(BC::Key::CatalogDialog::geometry, saveGeometry(), true);
    event->accept();
}

void CatalogOverlayDialog::setupTypeSpecificUI()
{
    // Load settings first to configure spinboxes
    loadSettings();
    
    setupFileSelection();
    setupConvolutionSettings();
    
    // Update initial state
    updateConvolutionControls();
    updateFileInfo();
}

void CatalogOverlayDialog::setupTypeSpecificConnections()
{
    connect(p_browseButton, &QToolButton::clicked, this, &CatalogOverlayDialog::onBrowseButtonClicked);
    connect(p_filePathLineEdit, &QLineEdit::textChanged, this, &CatalogOverlayDialog::onFilePathChanged);
    connect(p_convolutionEnabledCheckBox, &QCheckBox::toggled, this, &CatalogOverlayDialog::onConvolutionEnabledToggled);
    connect(p_lineshapeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CatalogOverlayDialog::onLineshapeTypeChanged);
    connect(p_linewidthSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayDialog::onConvolutionSettingsChanged);
    connect(p_minFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayDialog::onConvolutionSettingsChanged);
    connect(p_maxFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayDialog::onConvolutionSettingsChanged);
    connect(p_pointSpacingSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CatalogOverlayDialog::onConvolutionSettingsChanged);
    connect(p_autoRangeButton, &QPushButton::clicked, this, &CatalogOverlayDialog::onAutoRangeClicked);
    connect(p_saveRangeOnlyCheckBox, &QCheckBox::toggled, this, &CatalogOverlayDialog::onConvolutionSettingsChanged);
}

void CatalogOverlayDialog::initializeTypeSpecificDefaults()
{
    // Don't pre-populate the file path line edit
    // The last path is only used for the file browser's default location
}

bool CatalogOverlayDialog::validateTypeSpecificSettings(QString &errorMessage)
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

bool CatalogOverlayDialog::isTypeSpecificDataValid() const
{
    return d_fileValid && !d_catalogData.isEmpty();
}

void CatalogOverlayDialog::setupFileSelection()
{
    auto fileGroup = new QGroupBox("Catalog File");
    auto fileLayout = new QFormLayout(fileGroup);
    
    // File path selection
    auto pathLayout = new QHBoxLayout();
    p_filePathLineEdit = new QLineEdit();
    p_filePathLineEdit->setPlaceholderText("Select catalog file (.cat, .xo, .out)...");
    p_browseButton = new QToolButton();
    p_browseButton->setText("Browse...");
    p_browseButton->setMinimumSize(80, 0);
    
    pathLayout->addWidget(p_filePathLineEdit);
    pathLayout->addWidget(p_browseButton);
    fileLayout->addRow("File:", pathLayout);
    
    // File information display
    p_formatLabel = new QLabel("-");
    p_moleculeLabel = new QLabel("-");
    p_transitionCountLabel = new QLabel("-");
    p_frequencyRangeLabel = new QLabel("-");
    
    fileLayout->addRow("Format:", p_formatLabel);
    fileLayout->addRow("Molecule:", p_moleculeLabel);
    fileLayout->addRow("Transitions:", p_transitionCountLabel);
    fileLayout->addRow("Frequency Range:", p_frequencyRangeLabel);
    
    // Add to main layout
    static_cast<QVBoxLayout*>(layout())->insertWidget(1, fileGroup);
}

void CatalogOverlayDialog::setupConvolutionSettings()
{
    auto convGroup = new QGroupBox("Convolution Settings");
    auto convLayout = new QFormLayout(convGroup);
    
    // Enable convolution
    p_convolutionEnabledCheckBox = new QCheckBox("Enable convolution");
    p_convolutionEnabledCheckBox->setChecked(get(BC::Key::CatalogDialog::convolutionEnabled, DEFAULT_CONVOLUTION_ENABLED));
    convLayout->addRow(p_convolutionEnabledCheckBox);
    
    // Lineshape type
    p_lineshapeComboBox = new QComboBox();
    p_lineshapeComboBox->addItems({"Lorentzian", "Gaussian"});
    p_lineshapeComboBox->setCurrentIndex(get(BC::Key::CatalogDialog::lineshapeType, DEFAULT_LINESHAPE_TYPE));
    convLayout->addRow("Lineshape:", p_lineshapeComboBox);
    
    // Linewidth
    p_linewidthSpinBox = new QDoubleSpinBox();
    configureSpinBox(p_linewidthSpinBox, 
                     BC::Key::CatalogDialog::linewidthMin, BC::Key::CatalogDialog::linewidthMax,
                     BC::Key::CatalogDialog::linewidthDecimals, BC::Key::CatalogDialog::linewidthStep,
                     0.1, 10000.0, 1, 10.0);
    p_linewidthSpinBox->setSuffix(" kHz");
    p_linewidthSpinBox->setValue(get(BC::Key::CatalogDialog::linewidthKHz, DEFAULT_LINEWIDTH));
    convLayout->addRow("Linewidth (FWHM):", p_linewidthSpinBox);
    
    // Frequency range
    auto freqRangeLayout = new QHBoxLayout();
    p_minFreqSpinBox = new QDoubleSpinBox();
    configureSpinBox(p_minFreqSpinBox,
                     BC::Key::CatalogDialog::freqMin, BC::Key::CatalogDialog::freqMax,
                     BC::Key::CatalogDialog::freqDecimals, BC::Key::CatalogDialog::freqStep,
                     0.0, 100000.0, 3, 1.0);
    p_minFreqSpinBox->setSuffix(" MHz");
    // Initialize frequency range with Ft data if available, otherwise use settings
    double initialMinFreq, initialMaxFreq;
    if (d_hasFtData) {
        auto ftmwParent = qobject_cast<FtmwViewWidget*>(parent());
        if (ftmwParent) {
            Ft mainFt = ftmwParent->getMainPlotFt();
            auto xRange = mainFt.xRange();
            initialMinFreq = xRange.first;
            initialMaxFreq = xRange.second;
        } else {
            initialMinFreq = get(BC::Key::CatalogDialog::minFreqMHz, DEFAULT_MIN_FREQ);
            initialMaxFreq = get(BC::Key::CatalogDialog::maxFreqMHz, DEFAULT_MAX_FREQ);
        }
    } else {
        initialMinFreq = get(BC::Key::CatalogDialog::minFreqMHz, DEFAULT_MIN_FREQ);
        initialMaxFreq = get(BC::Key::CatalogDialog::maxFreqMHz, DEFAULT_MAX_FREQ);
    }
    p_minFreqSpinBox->setValue(initialMinFreq);
    
    p_maxFreqSpinBox = new QDoubleSpinBox();
    configureSpinBox(p_maxFreqSpinBox,
                     BC::Key::CatalogDialog::freqMin, BC::Key::CatalogDialog::freqMax,
                     BC::Key::CatalogDialog::freqDecimals, BC::Key::CatalogDialog::freqStep,
                     0.0, 100000.0, 3, 1.0);
    p_maxFreqSpinBox->setSuffix(" MHz");
    p_maxFreqSpinBox->setValue(initialMaxFreq);
    
    p_autoRangeButton = new QPushButton("Auto");
    p_autoRangeButton->setMaximumWidth(60);
    
    freqRangeLayout->addWidget(p_minFreqSpinBox);
    freqRangeLayout->addWidget(new QLabel("to"));
    freqRangeLayout->addWidget(p_maxFreqSpinBox);
    freqRangeLayout->addWidget(p_autoRangeButton);
    convLayout->addRow("Frequency Range:", freqRangeLayout);
    
    // Point spacing
    p_pointSpacingSpinBox = new QDoubleSpinBox();
    configureSpinBox(p_pointSpacingSpinBox,
                     BC::Key::CatalogDialog::pointSpacingMin, BC::Key::CatalogDialog::pointSpacingMax,
                     BC::Key::CatalogDialog::pointSpacingDecimals, BC::Key::CatalogDialog::pointSpacingStep,
                     0.001, 1.0, 3, 0.001);
    p_pointSpacingSpinBox->setSuffix(" MHz");
    p_pointSpacingSpinBox->setValue(get(BC::Key::CatalogDialog::pointSpacingMHz, DEFAULT_POINT_SPACING));
    convLayout->addRow("Point Spacing:", p_pointSpacingSpinBox);
    
    // Save range only option
    p_saveRangeOnlyCheckBox = new QCheckBox("Save only transitions within x range (recommended)");
    p_saveRangeOnlyCheckBox->setChecked(get(BC::Key::CatalogDialog::saveRangeOnly, DEFAULT_SAVE_RANGE_ONLY));
    p_saveRangeOnlyCheckBox->setToolTip("When enabled, only saves catalog transitions within the frequency range, reducing file size and improving performance.");
    convLayout->addRow(p_saveRangeOnlyCheckBox);
    
    // Add to main layout
    static_cast<QVBoxLayout*>(layout())->insertWidget(2, convGroup);
}

void CatalogOverlayDialog::onBrowseButtonClicked()
{
    QString lastPath = get(BC::Key::CatalogDialog::lastFilePath, 
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

void CatalogOverlayDialog::onFilePathChanged()
{
    d_filePath = p_filePathLineEdit->text().trimmed();
    
    if (d_filePath.isEmpty()) {
        d_fileValid = false;
        d_catalogData = CatalogData();
        updateFileInfo();
        updateOkButtonState();
        return;
    }
    
    loadCatalogFile(d_filePath);
    updateFileInfo();
    updateOkButtonState();
}

void CatalogOverlayDialog::onConvolutionEnabledToggled(bool enabled)
{
    Q_UNUSED(enabled)
    updateConvolutionControls();
    updateOkButtonState();
}

void CatalogOverlayDialog::onLineshapeTypeChanged(int index)
{
    Q_UNUSED(index)
    updateOkButtonState();
}

void CatalogOverlayDialog::onConvolutionSettingsChanged()
{
    updateOkButtonState();
}

void CatalogOverlayDialog::onAutoRangeClicked()
{
    autoSetFrequencyRange();
}

void CatalogOverlayDialog::loadCatalogFile(const QString &filePath)
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
        
        if (d_fileValid) {
            // Auto-set frequency range if currently at defaults
            if (qAbs(p_minFreqSpinBox->value() - DEFAULT_MIN_FREQ) < 1e-6 &&
                qAbs(p_maxFreqSpinBox->value() - DEFAULT_MAX_FREQ) < 1e-6) {
                autoSetFrequencyRange();
            }
        }
    } catch (const std::exception &e) {
        d_fileValid = false;
        d_catalogData = CatalogData();
    } catch (...) {
        d_fileValid = false;
        d_catalogData = CatalogData();
    }
}

void CatalogOverlayDialog::updateFileInfo()
{
    if (!d_fileValid || d_catalogData.isEmpty()) {
        p_formatLabel->setText("-");
        p_moleculeLabel->setText("-");
        p_transitionCountLabel->setText("-");
        p_frequencyRangeLabel->setText("-");
        return;
    }
    
    // Show format and molecule info
    p_formatLabel->setText(d_catalogData.sourceProgram());
    p_moleculeLabel->setText(d_catalogData.moleculeName());
    p_transitionCountLabel->setText(QString::number(d_catalogData.size()));
    
    // Auto-set overlay label from molecule name
    if (p_overlayOptionsWidget) {
        QString moleculeName = d_catalogData.moleculeName();
        if (!moleculeName.isEmpty()) {
            p_overlayOptionsWidget->setLabel(moleculeName);
        }
        
        // Enable x range checkboxes by default when catalog data is loaded
        p_overlayOptionsWidget->setMinFreqLimit(true, p_overlayOptionsWidget->getMinFreqValue());
        p_overlayOptionsWidget->setMaxFreqLimit(true, p_overlayOptionsWidget->getMaxFreqValue());
    }
    
    // Calculate and set reasonable default yscale
    calculateDefaultYScale();
    
    // Curve appearance will be set when overlay is created
    
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

void CatalogOverlayDialog::updateConvolutionControls()
{
    bool enabled = p_convolutionEnabledCheckBox->isChecked();
    
    p_lineshapeComboBox->setEnabled(enabled);
    p_linewidthSpinBox->setEnabled(enabled);
    p_minFreqSpinBox->setEnabled(enabled);
    p_maxFreqSpinBox->setEnabled(enabled);
    p_pointSpacingSpinBox->setEnabled(enabled);
    p_autoRangeButton->setEnabled(enabled && d_fileValid);
}

void CatalogOverlayDialog::autoSetFrequencyRange()
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
            rangeMin = get(BC::Key::CatalogDialog::minFreqMHz, DEFAULT_MIN_FREQ);
            rangeMax = get(BC::Key::CatalogDialog::maxFreqMHz, DEFAULT_MAX_FREQ);
        }
    } else {
        // Use settings values when no Ft data is available
        rangeMin = get(BC::Key::CatalogDialog::minFreqMHz, DEFAULT_MIN_FREQ);
        rangeMax = get(BC::Key::CatalogDialog::maxFreqMHz, DEFAULT_MAX_FREQ);
    }
    
    p_minFreqSpinBox->setValue(rangeMin);
    p_maxFreqSpinBox->setValue(rangeMax);
}

bool CatalogOverlayDialog::validateConvolutionSettings(QString &errorMessage)
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

QString CatalogOverlayDialog::formatFrequencyRange(double min, double max) const
{
    return QString("%1 - %2 MHz").arg(min, 0, 'f', 1).arg(max, 0, 'f', 1);
}

void CatalogOverlayDialog::loadSettings()
{
    QByteArray geom = get(BC::Key::CatalogDialog::geometry).toByteArray();
    if (!geom.isEmpty()) {
        restoreGeometry(geom);
    }
}

void CatalogOverlayDialog::saveSettings()
{
    // Save current file path
    set(BC::Key::CatalogDialog::lastFilePath, d_filePath);
    
    // Save convolution settings
    set(BC::Key::CatalogDialog::convolutionEnabled, p_convolutionEnabledCheckBox->isChecked());
    set(BC::Key::CatalogDialog::lineshapeType, p_lineshapeComboBox->currentIndex());
    set(BC::Key::CatalogDialog::linewidthKHz, p_linewidthSpinBox->value());
    set(BC::Key::CatalogDialog::minFreqMHz, p_minFreqSpinBox->value());
    set(BC::Key::CatalogDialog::maxFreqMHz, p_maxFreqSpinBox->value());
    set(BC::Key::CatalogDialog::pointSpacingMHz, p_pointSpacingSpinBox->value());
    set(BC::Key::CatalogDialog::saveRangeOnly, p_saveRangeOnlyCheckBox->isChecked());
}

void CatalogOverlayDialog::configureSpinBox(QDoubleSpinBox *spinBox, const QString &minKey, const QString &maxKey, 
                                           const QString &decimalsKey, const QString &stepKey, 
                                           double defaultMin, double defaultMax, int defaultDecimals, double defaultStep)
{
    spinBox->setMinimum(get(minKey, defaultMin));
    spinBox->setMaximum(get(maxKey, defaultMax));
    spinBox->setDecimals(get(decimalsKey, defaultDecimals));
    spinBox->setSingleStep(get(stepKey, defaultStep));
}

void CatalogOverlayDialog::calculateDefaultYScale()
{
    if (!p_overlayOptionsWidget || !d_fileValid || d_catalogData.isEmpty() || !d_hasFtData) {
        return; // Can't calculate without valid data
    }
    
    // Get the current frequency range (either from spinboxes or Ft data)
    double rangeMin = p_minFreqSpinBox->value();
    double rangeMax = p_maxFreqSpinBox->value();
    
    // Find the maximum intensity within the frequency range
    double maxIntensityInRange = 0.0;
    bool foundTransitions = false;
    
    for (int i = 0; i < d_catalogData.size(); ++i) {
        const auto &transition = d_catalogData.at(i);
        if (transition.frequency >= rangeMin && transition.frequency <= rangeMax) {
            maxIntensityInRange = qMax(maxIntensityInRange, transition.intensity);
            foundTransitions = true;
        }
    }
    
    if (!foundTransitions || maxIntensityInRange <= 0.0) {
        return; // No transitions in range or invalid intensities
    }
    
    // Calculate yscale: we want the strongest catalog line to be about 20% of the Ft yMax
    // This provides good visibility without overwhelming the spectrum
    double targetHeight = d_ftYMax * 0.2;
    double calculatedYScale = targetHeight / maxIntensityInRange;
    
    // Set the calculated yscale
    p_overlayOptionsWidget->setYScale(calculatedYScale);
}

