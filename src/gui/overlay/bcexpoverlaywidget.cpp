#include "bcexpoverlaywidget.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QStandardPaths>
#include <QDialogButtonBox>
#include <QDialog>

#include <gui/widget/ftmwviewwidget.h>
#include <gui/widget/experimentviewwidget.h>
#include <data/storage/settingsstorage.h>
#include <data/storage/fidstoragebase.h>
#include <data/experiment/experiment.h>
#include <data/storage/blackchirpcsv.h>

using namespace BC::Store;

BCExpOverlayWidget::BCExpOverlayWidget(QWidget *parent)
    : OverlayTypeSpecificWidget(parent),
      p_sourceFileConfigWidget(nullptr),
      p_sourceFileSettingsWidget(nullptr),
      p_overlaySettingsWidget(nullptr),
      d_experimentValid(false),
      d_hasFtData(false)
{
    setupUI();
    setupConnections();
}

BCExpOverlayWidget::~BCExpOverlayWidget() = default;

void BCExpOverlayWidget::setupForCreation()
{
    d_context = Context::Creation;
    d_overlay.reset();
    
    // Load default settings for creation context
    loadSettings();
    
    // Initialize defaults
    resetToDefaults();
    validateExperiment();
}

void BCExpOverlayWidget::setupForSettings(std::shared_ptr<OverlayBase> overlay)
{
    d_context = Context::Settings;
    d_overlay = overlay;
    
    if (overlay) {
        // Load settings from existing overlay
        auto bcexpOverlay = std::dynamic_pointer_cast<BCExpOverlay>(overlay);
        if (bcexpOverlay) {
            // Set source file path
            setSourceFilePath(bcexpOverlay->getSourceFile());
            
            // Load FT configuration if available
            d_configuredFt = bcexpOverlay->getFtData();
            d_hasFtData = !d_configuredFt.isEmpty();
            updateFtStatus();
        }
    }
    
    validateExperiment();
}

std::shared_ptr<OverlayBase> BCExpOverlayWidget::createOverlay() const
{
    if (d_context != Context::Creation) {
        return nullptr;
    }
    
    if (!isDataValid()) {
        return nullptr;
    }
    
    // Create the overlay
    auto overlay = std::make_shared<BCExpOverlay>();
    
    // Set the source file path
    overlay->setSourceFile(getExperimentPath());
    
    // Set the configured FT data
    overlay->setFtData(d_configuredFt);
    
    return overlay;
}

void BCExpOverlayWidget::applyToOverlay(std::shared_ptr<OverlayBase> overlay) const
{
    if (!overlay || d_context != Context::Settings) {
        return;
    }
    
    auto bcexpOverlay = std::dynamic_pointer_cast<BCExpOverlay>(overlay);
    if (!bcexpOverlay) {
        return;
    }
    
    // Apply current settings to the overlay
    bcexpOverlay->setSourceFile(getExperimentPath());
    bcexpOverlay->setFtData(d_configuredFt);
}

bool BCExpOverlayWidget::validateSettings(QString &errorMessage) const
{
    QStringList errors;
    
    // Validate experiment
    if (!d_experimentValid) {
        errors << "Please select a valid experiment";
    }
    
    // Validate FT configuration
    if (!d_hasFtData) {
        errors << "Please configure FT processing by clicking 'Configure FT...'";
    }
    
    if (!errors.isEmpty()) {
        errorMessage = errors.join("\n");
        return false;
    }
    
    return true;
}

bool BCExpOverlayWidget::isDataValid() const
{
    return d_experimentValid && d_hasFtData;
}

bool BCExpOverlayWidget::hasValidSourceFile() const
{
    return d_experimentValid;
}

QString BCExpOverlayWidget::getSourceFilePath() const
{
    return getExperimentPath();
}

void BCExpOverlayWidget::setSourceFilePath(const QString &path)
{
    if (path.isEmpty()) {
        return;
    }
    
    // Try to determine if this is an experiment number or custom path
    QDir dir(path);
    if (dir.exists()) {
        // It's a valid directory path
        p_usePathCheckBox->setChecked(true);
        p_pathLineEdit->setText(path);
    } else {
        // Try to parse as experiment number from path
        QString pathStr = path;
        QRegularExpression expNumRegex(R"(/(\d+)/?$)");
        auto match = expNumRegex.match(pathStr);
        if (match.hasMatch()) {
            int expNum = match.captured(1).toInt();
            if (expNum > 0) {
                p_usePathCheckBox->setChecked(false);
                p_experimentNumberSpinBox->setValue(expNum);
            }
        }
    }
    
    validateExperiment();
}

bool BCExpOverlayWidget::validateSourceFile(QString &errorMessage)
{
    QString path = getExperimentPath();
    bool valid = validateExperimentPath(path, errorMessage);
    d_experimentValid = valid;
    updateExperimentStatus();
    return valid;
}

void BCExpOverlayWidget::resetToDefaults()
{
    // Reset to default experiment number
    SettingsStorage s;
    int lastExperiment = s.get(BC::Key::exptNum, 1);
    p_experimentNumberSpinBox->setRange(1, lastExperiment);
    p_experimentNumberSpinBox->setValue(lastExperiment);
    
    // Use experiment number mode by default
    p_usePathCheckBox->setChecked(false);
    p_pathLineEdit->clear();
    
    // Reset FT configuration
    resetFtConfiguration();
    
    // Validate initial state
    validateExperiment();
}

QHash<QString, QVariant> BCExpOverlayWidget::getSettingsHash() const
{
    QHash<QString, QVariant> settings;
    
    // Experiment selection settings
    settings["experimentNumber"] = p_experimentNumberSpinBox->value();
    settings["useCustomPath"] = p_usePathCheckBox->isChecked();
    settings["customPath"] = p_pathLineEdit->text();
    
    // FT configuration settings
    settings["hasFtData"] = d_hasFtData;
    if (d_hasFtData) {
        // Add FT parameters to settings
        settings["ftEmpty"] = d_configuredFt.isEmpty();
        settings["ftMinFreq"] = d_configuredFt.minFreqMHz();
        settings["ftMaxFreq"] = d_configuredFt.maxFreqMHz();
        settings["ftSize"] = d_configuredFt.size();
    }
    
    return settings;
}

QWidget* BCExpOverlayWidget::getSourceFileConfigWidget()
{
    return p_sourceFileConfigWidget;
}

QWidget* BCExpOverlayWidget::getSourceFileSettingsWidget()
{
    return p_sourceFileSettingsWidget;
}

QWidget* BCExpOverlayWidget::getOverlaySettingsWidget()
{
    return p_overlaySettingsWidget;
}

void BCExpOverlayWidget::onExperimentNumberChanged(int number)
{
    Q_UNUSED(number);
    if (!p_usePathCheckBox->isChecked()) {
        resetFtConfiguration(); // Reset FT config when experiment changes
        validateExperiment();
        emit settingsChanged();
    }
}

void BCExpOverlayWidget::onUsePathToggled(bool enabled)
{
    p_pathLineEdit->setEnabled(enabled);
    p_browseButton->setEnabled(enabled);
    p_experimentNumberSpinBox->setEnabled(!enabled);
    
    resetFtConfiguration(); // Reset FT config when switching modes
    validateExperiment();
    emit settingsChanged();
}

void BCExpOverlayWidget::onBrowseButtonClicked()
{
    QString startPath = p_pathLineEdit->text();
    if (startPath.isEmpty()) {
        startPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    }
    
    QString path = QFileDialog::getExistingDirectory(this, "Select Experiment Directory", startPath);
    if (!path.isEmpty()) {
        p_pathLineEdit->setText(path);
    }
}

void BCExpOverlayWidget::onPathChanged()
{
    resetFtConfiguration(); // Reset FT config when path changes
    validateExperiment();
    emit settingsChanged();
}

void BCExpOverlayWidget::onConfigureFtClicked()
{
    QString experimentPath = getExperimentPath();
    if (experimentPath.isEmpty()) {
        QMessageBox::warning(this, "No Experiment Selected", 
                           "Please select a valid experiment before configuring FT.");
        return;
    }
    
    emit progressOperationStarted("Loading experiment data...");
    
    // Create ExperimentViewWidget with overlays disabled
    ExperimentViewWidget *experimentWidget;
    if (p_usePathCheckBox->isChecked()) {
        experimentWidget = new ExperimentViewWidget(0, p_pathLineEdit->text(), false);
    } else {
        experimentWidget = new ExperimentViewWidget(p_experimentNumberSpinBox->value(), QString(""), false);
    }
    
    // Set the CP-FTMW tab as active
    experimentWidget->setCurrentTab("CP-FTMW");
    
    // Create dialog with ExperimentViewWidget and Ok/Cancel buttons
    QDialog *ftDialog = new QDialog(this);
    ftDialog->setWindowTitle("Configure FT Processing");
    ftDialog->setModal(true);
    ftDialog->resize(800, 600);
    
    QVBoxLayout *dialogLayout = new QVBoxLayout(ftDialog);
    dialogLayout->addWidget(experimentWidget);
    
    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, ftDialog);
    dialogLayout->addWidget(buttonBox);
    
    connect(buttonBox, &QDialogButtonBox::accepted, ftDialog, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, ftDialog, &QDialog::reject);
    
    emit progressOperationFinished();
    
    // Show dialog and handle result
    if (ftDialog->exec() == QDialog::Accepted) {
        emit progressOperationStarted("Extracting FT data...");
        
        // Extract Ft object from the main plot
        d_configuredFt = experimentWidget->getMainPlotFt();
        d_hasFtData = !d_configuredFt.isEmpty();
        
        updateFtStatus();
        
        if (!d_hasFtData) {
            QMessageBox::warning(this, "No FT Data", 
                               "No valid FT data was found in the main plot. Please ensure the experiment data has been processed and is displayed in the main FT plot before configuring the overlay.");
        }
        
        emit progressOperationFinished();
        emit settingsChanged();
        emit dataValidityChanged(isDataValid());
    }
    
    ftDialog->deleteLater();
}

void BCExpOverlayWidget::validateExperiment()
{
    QString errorMessage;
    bool valid = validateSourceFile(errorMessage);
    
    d_experimentValid = valid;
    updateExperimentStatus();
    
    emit sourceFileChanged();
    emit dataValidityChanged(isDataValid());
}

void BCExpOverlayWidget::setupUI()
{
    // Create main layout
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);
    
    // Create the three-tier widgets
    setupExperimentSelectionUI();
    setupFtConfigurationUI();
    setupBCExpSettingsUI();
    
    // Add all widgets to main layout (they will be reparented by UnifiedOverlayWidget)
    mainLayout->addWidget(p_sourceFileConfigWidget);
    mainLayout->addWidget(p_sourceFileSettingsWidget);
    mainLayout->addWidget(p_overlaySettingsWidget);
}

void BCExpOverlayWidget::setupConnections()
{
    // Experiment selection connections
    connect(p_experimentNumberSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &BCExpOverlayWidget::onExperimentNumberChanged);
    connect(p_usePathCheckBox, &QCheckBox::toggled,
            this, &BCExpOverlayWidget::onUsePathToggled);
    connect(p_browseButton, &QToolButton::clicked,
            this, &BCExpOverlayWidget::onBrowseButtonClicked);
    connect(p_pathLineEdit, &QLineEdit::textChanged,
            this, &BCExpOverlayWidget::onPathChanged);
    
    // FT Configuration connections
    connect(p_configureFtButton, &QPushButton::clicked,
            this, &BCExpOverlayWidget::onConfigureFtClicked);
}

void BCExpOverlayWidget::loadSettings()
{
    // Settings loading is minimal for now - mainly defaults
    // Future: Could load last used paths, FT settings, etc.
}

void BCExpOverlayWidget::saveSettings()
{
    // Settings saving is minimal for now
    // Future: Could save last used paths, FT settings, etc.
}

void BCExpOverlayWidget::setupExperimentSelectionUI()
{
    p_sourceFileConfigWidget = new QWidget(this);
    QVBoxLayout *configLayout = new QVBoxLayout(p_sourceFileConfigWidget);
    configLayout->setContentsMargins(0, 0, 0, 0);
    
    p_experimentSelectionGroup = new QGroupBox("Experiment Selection", p_sourceFileConfigWidget);
    QFormLayout *formLayout = new QFormLayout(p_experimentSelectionGroup);
    
    // Experiment number
    p_experimentNumberSpinBox = new QSpinBox(p_experimentSelectionGroup);
    p_experimentNumberSpinBox->setMinimum(1);
    p_experimentNumberSpinBox->setMaximum(999999);
    p_experimentNumberSpinBox->setValue(1);
    formLayout->addRow("Experiment Number:", p_experimentNumberSpinBox);
    
    // Custom path option
    p_usePathCheckBox = new QCheckBox("Use custom path", p_experimentSelectionGroup);
    formLayout->addRow(p_usePathCheckBox);
    
    // Path selection
    QHBoxLayout *pathLayout = new QHBoxLayout();
    p_pathLineEdit = new QLineEdit(p_experimentSelectionGroup);
    p_pathLineEdit->setEnabled(false);
    p_browseButton = new QToolButton(p_experimentSelectionGroup);
    p_browseButton->setText("...");
    p_browseButton->setEnabled(false);
    pathLayout->addWidget(p_pathLineEdit);
    pathLayout->addWidget(p_browseButton);
    formLayout->addRow("Path:", pathLayout);
    
    // Status label
    p_experimentStatusLabel = new QLabel(p_experimentSelectionGroup);
    p_experimentStatusLabel->setWordWrap(true);
    formLayout->addRow("Status:", p_experimentStatusLabel);
    
    configLayout->addWidget(p_experimentSelectionGroup);
}

void BCExpOverlayWidget::setupFtConfigurationUI()
{
    p_sourceFileSettingsWidget = new QWidget(this);
    QVBoxLayout *settingsLayout = new QVBoxLayout(p_sourceFileSettingsWidget);
    settingsLayout->setContentsMargins(0, 0, 0, 0);
    
    p_ftConfigurationGroup = new QGroupBox("FT Configuration", p_sourceFileSettingsWidget);
    QVBoxLayout *ftLayout = new QVBoxLayout(p_ftConfigurationGroup);
    
    // Configure FT button
    p_configureFtButton = new QPushButton("Configure FT...", p_ftConfigurationGroup);
    p_configureFtButton->setMinimumHeight(30);
    
    // Status label
    p_ftStatusLabel = new QLabel("Click to configure FT processing settings for this overlay.", p_ftConfigurationGroup);
    p_ftStatusLabel->setWordWrap(true);
    p_ftStatusLabel->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    
    ftLayout->addWidget(p_configureFtButton);
    ftLayout->addWidget(p_ftStatusLabel);
    
    settingsLayout->addWidget(p_ftConfigurationGroup);
}

void BCExpOverlayWidget::setupBCExpSettingsUI()
{
    p_overlaySettingsWidget = new QWidget(this);
    QVBoxLayout *overlayLayout = new QVBoxLayout(p_overlaySettingsWidget);
    overlayLayout->setContentsMargins(0, 0, 0, 0);
    
    p_bcexpSettingsGroup = new QGroupBox("BCExperiment Settings", p_overlaySettingsWidget);
    QVBoxLayout *bcexpLayout = new QVBoxLayout(p_bcexpSettingsGroup);
    
    // Placeholder for future BCExp-specific settings
    QLabel *placeholderLabel = new QLabel("Future BCExperiment-specific settings will be added here.");
    placeholderLabel->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    bcexpLayout->addWidget(placeholderLabel);
    
    overlayLayout->addWidget(p_bcexpSettingsGroup);
}

void BCExpOverlayWidget::resetFtConfiguration()
{
    d_configuredFt = Ft(); // Reset to empty FT
    d_hasFtData = false;
    updateFtStatus();
}

QString BCExpOverlayWidget::getExperimentPath() const
{
    if (p_usePathCheckBox->isChecked()) {
        return p_pathLineEdit->text();
    } else {
        // Construct path from experiment number
        return BlackchirpCSV::exptDir(p_experimentNumberSpinBox->value()).absolutePath();
    }
}

bool BCExpOverlayWidget::validateExperimentPath(const QString &path, QString &errorMessage)
{
    if (path.isEmpty()) {
        errorMessage = "Please specify an experiment path or number.";
        return false;
    }
    
    QDir expDir(path);
    if (!expDir.exists()) {
        errorMessage = QString("Experiment directory does not exist: %1").arg(path);
        return false;
    }
    
    // Check for header.csv
    if (!expDir.exists("header.csv")) {
        errorMessage = QString("No header.csv file found in: %1").arg(path);
        return false;
    }
    
    // Check for fid folder
    if (!expDir.exists("fid")) {
        errorMessage = QString("No fid folder found in: %1").arg(path);
        return false;
    }
    
    QDir fidDir(expDir.absoluteFilePath("fid"));
    if (!fidDir.exists()) {
        errorMessage = QString("fid folder exists but is not accessible: %1").arg(fidDir.absolutePath());
        return false;
    }
    
    return true;
}

void BCExpOverlayWidget::updateExperimentStatus()
{
    if (d_experimentValid) {
        p_experimentStatusLabel->setText("✓ Valid experiment found");
        p_experimentStatusLabel->setStyleSheet("QLabel { color: green; }");
    } else {
        QString errorMessage;
        validateExperimentPath(getExperimentPath(), errorMessage);
        p_experimentStatusLabel->setText(QString("✗ %1").arg(errorMessage));
        p_experimentStatusLabel->setStyleSheet("QLabel { color: red; }");
    }
}

void BCExpOverlayWidget::updateFtStatus()
{
    if (d_hasFtData) {
        p_configureFtButton->setText("FT Configured ✓");
        p_configureFtButton->setStyleSheet("QPushButton { color: green; }");
        p_ftStatusLabel->setText("FT data configured and ready.");
        p_ftStatusLabel->setStyleSheet("QLabel { color: green; }");
    } else {
        p_configureFtButton->setText("Configure FT...");
        p_configureFtButton->setStyleSheet("");
        p_ftStatusLabel->setText("Click to configure FT processing settings for this overlay.");
        p_ftStatusLabel->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    }
}