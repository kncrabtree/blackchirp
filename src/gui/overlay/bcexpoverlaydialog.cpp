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
    d_plotNames(plotNames),
    d_hasFtData(false),
    p_msw(nullptr)
{
    setupUI();
    setupConnections();
    initializeDefaults();
}

BCExpOverlayDialog::~BCExpOverlayDialog()
{
    if (p_msw) {
        p_msw->deleteLater();
    }
}

std::shared_ptr<OverlayBase> BCExpOverlayDialog::createOverlay() const
{
    if (!d_experimentValid || !d_hasFtData) {
        return nullptr;
    }

    // Create the overlay
    auto overlay = std::make_shared<BCExpOverlay>();
    
    // Set the source file path from the experiment
    QString experimentPath = getExperimentPath();
    overlay->setSourceFile(experimentPath);
    
    // Apply overlay base options (label, plot ID, scaling, offsets)
    if (p_overlayOptionsWidget) {
        p_overlayOptionsWidget->applyToOverlay(overlay);
    }
    
    // Set the configured FT data
    overlay->setFtData(d_configuredFt);
    
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
    setupFtConfiguration();
    
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

void BCExpOverlayDialog::setupFtConfiguration()
{
    QGroupBox *ftGroup = new QGroupBox("FT Configuration", this);
    QVBoxLayout *ftLayout = new QVBoxLayout(ftGroup);
    
    // Configure FT button
    p_configureFtButton = new QPushButton("Configure FT...", this);
    p_configureFtButton->setMinimumHeight(30);
    
    QLabel *infoLabel = new QLabel("Click to configure FT processing settings for this overlay.", this);
    infoLabel->setWordWrap(true);
    infoLabel->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    
    ftLayout->addWidget(p_configureFtButton);
    ftLayout->addWidget(infoLabel);
    
    layout()->addWidget(ftGroup);
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
    
    // FT Configuration
    connect(p_configureFtButton, &QPushButton::clicked,
            this, &BCExpOverlayDialog::onConfigureFtClicked);
    
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
        resetFtConfiguration(); // Reset FT config when experiment changes
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
    
    resetFtConfiguration(); // Reset FT config when switching modes
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
    resetFtConfiguration(); // Reset FT config when path changes
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
    
    // No additional processing needed for FT configuration mode
    
    updateValidationStatus(valid, errorMessage);
    
    d_experimentValid = valid;
    updateOkButtonState();
}


void BCExpOverlayDialog::onConfigureFtClicked()
{
    QString experimentPath = getExperimentPath();
    if (experimentPath.isEmpty()) {
        QMessageBox::warning(this, "No Experiment Selected", 
                           "Please select a valid experiment before configuring FT.");
        return;
    }
    
    // Create ExperimentViewWidget
    ExperimentViewWidget *experimentWidget;
    if (p_usePathCheckBox->isChecked()) {
        experimentWidget = new ExperimentViewWidget(0, p_pathLineEdit->text());
    } else {
        experimentWidget = new ExperimentViewWidget(p_experimentNumberSpinBox->value());
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
    
    // Show dialog and handle result
    if (ftDialog->exec() == QDialog::Accepted) {
        // Extract Ft object from the main plot
        d_configuredFt = experimentWidget->getMainPlotFt();
        d_hasFtData = !d_configuredFt.isEmpty();
        
        if (d_hasFtData) {
            p_configureFtButton->setText("FT Configured ✓");
            p_configureFtButton->setStyleSheet("QPushButton { color: green; }");
        } else {
            QMessageBox::warning(this, "No FT Data", 
                               "No valid FT data was found in the main plot. Please ensure the experiment data has been processed and is displayed in the main FT plot before configuring the overlay.");
            p_configureFtButton->setText("Configure FT...");
            p_configureFtButton->setStyleSheet("");
        }
        updateOkButtonState();
    }
    
    ftDialog->deleteLater();
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

void BCExpOverlayDialog::updateOkButtonState()
{
    bool canAccept = d_experimentValid && d_hasFtData;
    p_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(canAccept);
}

void BCExpOverlayDialog::resetFtConfiguration()
{
    d_configuredFt = Ft(); // Reset to empty FT
    d_hasFtData = false;
    
    // Reset button appearance
    p_configureFtButton->setText("Configure FT...");
    p_configureFtButton->setStyleSheet("");
    
    // Update OK button state
    updateOkButtonState();
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
            existingOverlays = p_ftmwViewWidget->getAllOverlays();
        }
        
        if (!p_overlayOptionsWidget->validateSettings(overlayError, existingOverlays)) {
            validationErrors << overlayError;
        }
    }
    
    // Validate FT configuration
    if (!d_hasFtData) {
        validationErrors << "Please configure FT processing by clicking 'Configure FT...'";
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
