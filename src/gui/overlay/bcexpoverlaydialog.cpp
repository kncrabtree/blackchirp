#include "bcexpoverlaydialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QStandardPaths>
#include <QButtonGroup>

#include <gui/widget/ftmwviewwidget.h>
#include <gui/widget/experimentviewwidget.h>
#include <data/storage/settingsstorage.h>
#include <data/storage/fidstoragebase.h>
#include <data/experiment/experiment.h>
#include <data/storage/blackchirpcsv.h>

using namespace BC::Store;

BCExpOverlayDialog::BCExpOverlayDialog(const QStringList &plotNames, FtmwViewWidget *parent) :
    QDialog(parent),
    p_ftmwViewWidget(parent),
    d_experimentValid(false),
    d_hasExperimentSettings(false),
    d_hasManualSettings(false),
    d_plotNames(plotNames),
    p_msw(nullptr)
{
    setupUI();
    setupConnections();
    initializeDefaults();
    getCurrentSettingsFromParent();
}

BCExpOverlayDialog::~BCExpOverlayDialog()
{
    if (p_msw) {
        p_msw->deleteLater();
    }
}

std::shared_ptr<OverlayBase> BCExpOverlayDialog::createOverlay() const
{
    if (!d_experimentValid) {
        return nullptr;
    }

    // Create the overlay with the experiment number/path and frame
    int experimentNumber = p_usePathCheckBox->isChecked() ? 0 : p_experimentNumberSpinBox->value();
    QString experimentPath = p_usePathCheckBox->isChecked() ? p_pathLineEdit->text() : QString("");
    int frame = p_frameSpinBox->value();
    
    auto overlay = std::make_shared<BCExpOverlay>(experimentNumber, experimentPath, frame);
    
    // Apply overlay base options (label, plot ID, scaling, offsets)
    if (p_overlayOptionsWidget) {
        p_overlayOptionsWidget->applyToOverlay(overlay);
    }
    
    // Set processing settings based on user selection
    if (p_useExperimentSettingsRadio->isChecked()) {
        // Use automatic processing (default)
        overlay->setAutomaticProcessing(true);
    } else if (p_useCurrentSettingsRadio->isChecked()) {
        // Use current settings from parent
        overlay->setProcessingSettings(d_currentSettings);
    } else if (p_useManualSettingsRadio->isChecked() && d_hasManualSettings) {
        // Use manually configured settings
        overlay->setProcessingSettings(d_manualSettings);
    } else {
        // Fallback to current settings
        overlay->setProcessingSettings(d_currentSettings);
    }
    
    return overlay;
}

void BCExpOverlayDialog::setupUI()
{
    setWindowTitle("Add BCExperiment Overlay");
    setModal(true);
    resize(450, 400);
    
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    setupOverlayBaseOptions();
    setupExperimentSelection();
    setupFrameSelection();
    setupProcessingSettings();
    
    // Add validation label
    p_validationLabel = new QLabel(this);
    p_validationLabel->setStyleSheet("QLabel { color: red; }");
    p_validationLabel->setWordWrap(true);
    p_validationLabel->hide();
    mainLayout->addWidget(p_validationLabel);
    
    // Add button box
    p_buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    p_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
    mainLayout->addWidget(p_buttonBox);
    
    setLayout(mainLayout);
}

void BCExpOverlayDialog::setupExperimentSelection()
{
    QGroupBox *experimentGroup = new QGroupBox("Experiment Selection", this);
    QFormLayout *formLayout = new QFormLayout(experimentGroup);
    
    // Experiment number
    p_experimentNumberSpinBox = new QSpinBox(this);
    p_experimentNumberSpinBox->setMinimum(1);
    p_experimentNumberSpinBox->setMaximum(999999);
    p_experimentNumberSpinBox->setValue(1);
    formLayout->addRow("Experiment Number:", p_experimentNumberSpinBox);
    
    // Custom path option
    p_usePathCheckBox = new QCheckBox("Use custom path", this);
    formLayout->addRow(p_usePathCheckBox);
    
    // Path selection
    QHBoxLayout *pathLayout = new QHBoxLayout();
    p_pathLineEdit = new QLineEdit(this);
    p_pathLineEdit->setEnabled(false);
    p_browseButton = new QToolButton(this);
    p_browseButton->setText("...");
    p_browseButton->setEnabled(false);
    pathLayout->addWidget(p_pathLineEdit);
    pathLayout->addWidget(p_browseButton);
    formLayout->addRow("Path:", pathLayout);
    
    layout()->addWidget(experimentGroup);
}

void BCExpOverlayDialog::setupFrameSelection()
{
    QGroupBox *frameGroup = new QGroupBox("Frame Selection", this);
    QFormLayout *formLayout = new QFormLayout(frameGroup);
    
    p_frameSpinBox = new QSpinBox(this);
    p_frameSpinBox->setMinimum(-1);
    p_frameSpinBox->setMaximum(999999);
    p_frameSpinBox->setValue(-1);
    p_frameSpinBox->setSpecialValueText("Averaged Data");
    formLayout->addRow("Frame:", p_frameSpinBox);
    
    layout()->addWidget(frameGroup);
}

void BCExpOverlayDialog::setupOverlayBaseOptions()
{
    QGroupBox *optionsGroup = new QGroupBox("Overlay Options", this);
    QVBoxLayout *optionsLayout = new QVBoxLayout(optionsGroup);
    
    // Create the options widget with the plot names
    p_overlayOptionsWidget = new OverlayBaseOptionsWidget(d_plotNames, this);
    optionsLayout->addWidget(p_overlayOptionsWidget);
    
    layout()->addWidget(optionsGroup);
}

void BCExpOverlayDialog::setupProcessingSettings()
{
    QGroupBox *settingsGroup = new QGroupBox("Processing Settings", this);
    QVBoxLayout *settingsLayout = new QVBoxLayout(settingsGroup);
    
    // Create radio buttons for the three options
    p_useExperimentSettingsRadio = new QRadioButton("Use experiment settings (if available)", this);
    p_useCurrentSettingsRadio = new QRadioButton("Use current view settings", this);
    p_useManualSettingsRadio = new QRadioButton("Configure manually", this);
    
    // Group the radio buttons
    QButtonGroup *settingsButtonGroup = new QButtonGroup(this);
    settingsButtonGroup->addButton(p_useExperimentSettingsRadio);
    settingsButtonGroup->addButton(p_useCurrentSettingsRadio);
    settingsButtonGroup->addButton(p_useManualSettingsRadio);
    
    // Manual settings button
    p_manualSettingsButton = new QPushButton("Configure...", this);
    p_manualSettingsButton->setEnabled(false);
    
    // Settings status label
    p_settingsStatusLabel = new QLabel(this);
    p_settingsStatusLabel->setWordWrap(true);
    p_settingsStatusLabel->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    
    // Layout
    settingsLayout->addWidget(p_useExperimentSettingsRadio);
    settingsLayout->addWidget(p_useCurrentSettingsRadio);
    
    QHBoxLayout *manualLayout = new QHBoxLayout();
    manualLayout->addWidget(p_useManualSettingsRadio);
    manualLayout->addWidget(p_manualSettingsButton);
    manualLayout->addStretch();
    settingsLayout->addLayout(manualLayout);
    
    settingsLayout->addWidget(p_settingsStatusLabel);
    
    // Set default selection
    p_useExperimentSettingsRadio->setChecked(true);
    
    layout()->addWidget(settingsGroup);
}

void BCExpOverlayDialog::setupConnections()
{
    // Experiment selection
    connect(p_experimentNumberSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &BCExpOverlayDialog::onExperimentNumberChanged);
    connect(p_usePathCheckBox, &QCheckBox::toggled,
            this, &BCExpOverlayDialog::onUsePathToggled);
    connect(p_browseButton, &QToolButton::clicked,
            this, &BCExpOverlayDialog::onBrowseButtonClicked);
    connect(p_pathLineEdit, &QLineEdit::textChanged,
            this, &BCExpOverlayDialog::onPathChanged);
    
    // Processing settings
    connect(p_useExperimentSettingsRadio, &QRadioButton::toggled,
            this, &BCExpOverlayDialog::onProcessingSettingsChanged);
    connect(p_useCurrentSettingsRadio, &QRadioButton::toggled,
            this, &BCExpOverlayDialog::onProcessingSettingsChanged);
    connect(p_useManualSettingsRadio, &QRadioButton::toggled,
            this, &BCExpOverlayDialog::onProcessingSettingsChanged);
    connect(p_manualSettingsButton, &QPushButton::clicked,
            this, &BCExpOverlayDialog::onManualSettingsClicked);
    
    // Dialog buttons
    connect(p_buttonBox, &QDialogButtonBox::accepted,
            this, &BCExpOverlayDialog::onDialogAccepted);
    connect(p_buttonBox, &QDialogButtonBox::rejected,
            this, &QDialog::reject);
}

void BCExpOverlayDialog::initializeDefaults()
{
    // Load the last used experiment number
    SettingsStorage s;
    int lastExperiment = s.get(BC::Key::exptNum, 1);
    p_experimentNumberSpinBox->setRange(1,lastExperiment);
    p_experimentNumberSpinBox->setValue(lastExperiment);
    
    // Set initial label based on experiment number
    if (p_overlayOptionsWidget) {
        p_overlayOptionsWidget->setLabel(QString("Exp%1").arg(lastExperiment));
    }
    
    // Validate the initial experiment
    validateExperiment();
}

void BCExpOverlayDialog::onExperimentNumberChanged(int number)
{
    if (!p_usePathCheckBox->isChecked()) {
        validateExperiment();
        // Update label to experiment number
        if (p_overlayOptionsWidget) {
            p_overlayOptionsWidget->setLabel(QString("Exp%1").arg(number));
        }
    }
}

void BCExpOverlayDialog::onUsePathToggled(bool enabled)
{
    p_pathLineEdit->setEnabled(enabled);
    p_browseButton->setEnabled(enabled);
    p_experimentNumberSpinBox->setEnabled(!enabled);
    
    validateExperiment();
    
    // Update label based on current selection
    if (p_overlayOptionsWidget) {
        if (enabled) {
            // Using custom path - update label from path if available
            QString path = p_pathLineEdit->text();
            if (!path.isEmpty()) {
                QDir dir(path);
                QString folderName = dir.dirName();
                if (!folderName.isEmpty()) {
                    p_overlayOptionsWidget->setLabel(folderName);
                }
            }
        } else {
            // Using experiment number - update label from number
            int number = p_experimentNumberSpinBox->value();
            p_overlayOptionsWidget->setLabel(QString("Exp%1").arg(number));
        }
    }
}

void BCExpOverlayDialog::onBrowseButtonClicked()
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

void BCExpOverlayDialog::onPathChanged()
{
    validateExperiment();
    // Update label to folder name if using custom path
    if (p_usePathCheckBox->isChecked() && p_overlayOptionsWidget) {
        QString path = p_pathLineEdit->text();
        if (!path.isEmpty()) {
            QDir dir(path);
            QString folderName = dir.dirName();
            if (!folderName.isEmpty()) {
                p_overlayOptionsWidget->setLabel(folderName);
            }
        }
    }
}

void BCExpOverlayDialog::validateExperiment()
{
    QString errorMessage;
    QString experimentPath = getExperimentPath();
    
    bool valid = validateExperimentPath(experimentPath, errorMessage);
    
    if (valid) {
        // Check if processing.csv exists for the experiment settings option
        QDir fidDir(experimentPath + "/fid");
        d_hasExperimentSettings = fidDir.exists("processing.csv");
        updateProcessingSettingsOptions();
    } else {
        d_hasExperimentSettings = false;
    }
    
    updateValidationStatus(valid, errorMessage);
    updateSettingsStatus();
    
    d_experimentValid = valid;
    p_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(valid);
}

void BCExpOverlayDialog::onProcessingSettingsChanged()
{
    p_manualSettingsButton->setEnabled(p_useManualSettingsRadio->isChecked());
    updateSettingsStatus();
}

void BCExpOverlayDialog::onManualSettingsClicked()
{
    if (p_msw) {
        p_msw->raise();
        p_msw->activateWindow();
        return;
    }
    
    QString experimentPath = getExperimentPath();
    if (experimentPath.isEmpty()) {
        QMessageBox::warning(this, "No Experiment Selected", 
                           "Please select a valid experiment before configuring manual settings.");
        return;
    }
    
    // Create a new ExperimentViewWidget

    // Load the experiment
    if (p_usePathCheckBox->isChecked()) {
        p_msw = new ExperimentViewWidget(0,p_pathLineEdit->text());
    } else {
        p_msw = new ExperimentViewWidget(p_experimentNumberSpinBox->value());
    }

    p_msw->setAttribute(Qt::WA_DeleteOnClose);
    p_msw->setWindowTitle("Configure Processing Settings");
    p_msw->setWindowModality(Qt::ApplicationModal);
    
    // Connect to handle when the widget is closed
    connect(p_msw, &QWidget::destroyed, this, [this]() {
        if (p_msw) {
            // Extract the processing settings before the widget is destroyed
            d_manualSettings = p_msw->getFtmwProcessingSettings();
            d_hasManualSettings = true;
            p_msw = nullptr;
            updateSettingsStatus();
        }
    });
       
    p_msw->show();
}

void BCExpOverlayDialog::onDialogAccepted()
{
    // Additional validation could go here
    accept();
}

void BCExpOverlayDialog::updateValidationStatus(bool valid, const QString &message)
{
    if (valid) {
        p_validationLabel->hide();
    } else {
        p_validationLabel->setText(message);
        p_validationLabel->show();
    }
}

QString BCExpOverlayDialog::getExperimentPath() const
{
    if (p_usePathCheckBox->isChecked()) {
        return p_pathLineEdit->text();
    } else {
        // Construct path from experiment number
        return BlackchirpCSV::exptDir(p_experimentNumberSpinBox->value()).absolutePath();
    }
}

bool BCExpOverlayDialog::validateExperimentPath(const QString &path, QString &errorMessage)
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


void BCExpOverlayDialog::updateProcessingSettingsOptions()
{
    p_useExperimentSettingsRadio->setEnabled(d_hasExperimentSettings);
    
    if (!d_hasExperimentSettings && p_useExperimentSettingsRadio->isChecked()) {
        p_useCurrentSettingsRadio->setChecked(true);
    }
}

void BCExpOverlayDialog::updateSettingsStatus()
{
    QString statusText;
    
    if (p_useExperimentSettingsRadio->isChecked()) {
        if (d_hasExperimentSettings) {
            statusText = "Will use processing settings from experiment's processing.csv file, or defaults if none found.";
        } else {
            statusText = "No processing.csv file found. Will use default processing settings.";
        }
    } else if (p_useCurrentSettingsRadio->isChecked()) {
        statusText = "Will use current processing settings from the active view.";
    } else if (p_useManualSettingsRadio->isChecked()) {
        if (d_hasManualSettings) {
            statusText = "Will use manually configured processing settings.";
        } else {
            statusText = "Click 'Configure...' to set up processing parameters.";
        }
    }
    
    p_settingsStatusLabel->setText(statusText);
}

FtWorker::FidProcessingSettings BCExpOverlayDialog::getSelectedProcessingSettings() const
{
    if (p_useExperimentSettingsRadio->isChecked() && d_hasExperimentSettings) {
        return d_experimentSettings;
    } else if (p_useCurrentSettingsRadio->isChecked()) {
        return d_currentSettings;
    } else if (p_useManualSettingsRadio->isChecked() && d_hasManualSettings) {
        return d_manualSettings;
    }
    
    // Fallback to current settings
    return d_currentSettings;
}

void BCExpOverlayDialog::getCurrentSettingsFromParent()
{
    if (p_ftmwViewWidget) {
         d_currentSettings = p_ftmwViewWidget->getProcessingSettings();
    } else {
        // Initialize with defaults if no parent
        d_currentSettings.startUs = 5.0;
        d_currentSettings.endUs = 10.0;
        d_currentSettings.expFilter = 0.0;
        d_currentSettings.zeroPadFactor = 0;
        d_currentSettings.removeDC = true;
        d_currentSettings.units = FtWorker::FtuV;
        d_currentSettings.autoScaleIgnoreMHz = 250.0;
        d_currentSettings.windowFunction = FtWorker::None;
    }
}

void BCExpOverlayDialog::accept()
{
    QStringList validationErrors;
    
    // Validate experiment
    if (!d_experimentValid) {
        validationErrors << "Please select a valid experiment";
    }
    
    // Validate overlay base options
    if (p_overlayOptionsWidget) {
        QString overlayError;
        QVector<std::shared_ptr<OverlayBase>> existingOverlays;
        
        // Get existing overlays from parent if available
        if (p_ftmwViewWidget) {
            existingOverlays = p_ftmwViewWidget->getOverlays();
        }
        
        if (!p_overlayOptionsWidget->validateSettings(overlayError, existingOverlays)) {
            validationErrors << overlayError;
        }
    }
    
    // Validate manual settings if selected
    if (p_useManualSettingsRadio->isChecked() && !d_hasManualSettings) {
        validationErrors << "Please configure manual settings or select a different processing option";
    }
    
    // Show validation errors if any
    if (!validationErrors.isEmpty()) {
        QString errorMessage = "Please fix the following issues:\n\n";
        errorMessage += validationErrors.join("\n");
        
        QMessageBox::warning(this, "Validation Error", errorMessage);
        return; // Don't close dialog
    }
    
    // All validation passed, call base class accept
    QDialog::accept();
}
