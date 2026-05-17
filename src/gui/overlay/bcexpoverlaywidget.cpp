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
#include <QTimer>

#include <gui/widget/ftmwviewwidget.h>
#include <gui/style/themecolors.h>
#include <gui/widget/settingstable.h>
#include <gui/widget/experimentviewwidget.h>
#include <data/storage/settingsstorage.h>
#include <data/storage/fidstoragebase.h>
#include <data/experiment/experiment.h>
#include <data/storage/blackchirpcsv.h>

namespace {
// Apply a themed foreground color to a label. A widget-scoped
// stylesheet (themed via ThemeColors::getCSSColor) is used rather than
// a palette: a QLabel hosted in a QTableWidget cell does not reliably
// pick up a palette WindowText override, but a stylesheet color always
// wins. The color is still theme-derived, not a hard-coded string.
void styleStatusLabel(QLabel *label, ThemeColors::ColorRole role,
                      bool italic, bool bold = false)
{
    label->setStyleSheet(QString("color:%1;")
        .arg(ThemeColors::getCSSColor(role, label)));
    QFont f = label->font();
    f.setItalic(italic);
    f.setBold(bold);
    label->setFont(f);
}
}

using namespace BC::Store;

BCExpOverlayWidget::BCExpOverlayWidget(const Ft &currentFt, QWidget *parent)
    : OverlayTypeSpecificWidget(currentFt, parent),
      SettingsStorage(BC::Key::BCExpWidget::key),
      d_experimentValid(false),
      d_hasFtData(false)
{
    // Base class handles setupUI() and setupConnections()
}

BCExpOverlayWidget::~BCExpOverlayWidget() = default;

void BCExpOverlayWidget::setupForCreation()
{
    d_context = Context::Creation;
    d_overlay.reset();
    
    // Load default settings for creation context
    loadSettings();
    
    // Validate and update label after everything is set up (deferred to ensure UI is ready)
    QTimer::singleShot(0, this, [this]() {
        updateAutomaticLabel();
        validateExperiment();
    });
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

std::shared_ptr<OverlayBase> BCExpOverlayWidget::createOverlay()
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
    
    // Store the created overlay for operations in creation context
    d_overlay = overlay;
    
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

bool BCExpOverlayWidget::validateSettingsImpl()
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
        setSettingsErrorMessage(errors.join("\n"));
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
        updatePathDisplayAndTooltip(p_pathLineEdit, path);
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

bool BCExpOverlayWidget::validateSourceFileImpl()
{
    QString path = getExperimentPath();
    QString errorMessage;
    bool valid = validateExperimentPath(path, errorMessage);
    
    if (!valid) {
        setSourceFileErrorMessage(errorMessage);
    }
    
    d_experimentValid = valid;
    updateExperimentStatus();
    return valid;
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


std::shared_ptr<OverlayOperation> BCExpOverlayWidget::createOperation(OperationCapability::Type type,
                                                                     std::shared_ptr<OverlayBase> overlay) const
{
    Q_UNUSED(overlay);
    
    // BCExperiment overlays don't currently use background operations
    // Return nullptr to indicate synchronous processing should be used
    switch (type) {
    case OperationCapability::Creation:
    case OperationCapability::Validation:
    case OperationCapability::PreviewUpdate:
    case OperationCapability::Convolution:
        break;
    }
    
    return nullptr;
}


void BCExpOverlayWidget::onExperimentNumberChanged(int number)
{
    Q_UNUSED(number);
    if (!p_usePathCheckBox->isChecked()) {
        resetFtConfiguration(); // Reset FT config when experiment changes
        updateAutomaticLabel(); // Update label to new experiment number
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
    updateAutomaticLabel(); // Update label based on current selection
    validateExperiment();
    emit settingsChanged();
}

void BCExpOverlayWidget::onBrowseButtonClicked()
{
    QString startPath = p_pathLineEdit->text();
    if (startPath.isEmpty()) {
        startPath = get(BC::Key::BCExpWidget::lastBrowseDir,
                        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    }

    QString path = QFileDialog::getExistingDirectory(this, "Select Experiment Directory", startPath);
    if (!path.isEmpty()) {
        // Persist immediately so the directory is remembered even if the
        // dialog is cancelled before the overlay is created.
        set(BC::Key::BCExpWidget::lastBrowseDir, path, true);
        updatePathDisplayAndTooltip(p_pathLineEdit, path);
    }
}

void BCExpOverlayWidget::onPathChanged()
{
    resetFtConfiguration(); // Reset FT config when path changes
    updateAutomaticLabel(); // Update label to directory name if using custom path
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
    bool valid = validateSourceFile();
    
    d_experimentValid = valid;
    updateExperimentStatus();
    
    emit dataValidityChanged(isDataValid());
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
    // Load default experiment number from settings
    SettingsStorage s;
    int lastExperiment = s.get(BC::Key::exptNum, 1);
    p_experimentNumberSpinBox->setRange(1, lastExperiment);
    p_experimentNumberSpinBox->setValue(lastExperiment);
    
    // Use experiment number mode by default
    p_usePathCheckBox->setChecked(false);
    p_pathLineEdit->clear();
    
    // Reset FT configuration
    resetFtConfiguration();
    
    // Future: Could load last used paths, FT settings, etc.
}

void BCExpOverlayWidget::saveSettings()
{
    // Settings saving is minimal for now
    // Future: Could save last used paths, FT settings, etc.
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
        return getStoredFullSourceFilePath(); // Use stored full path instead of potentially abbreviated display text
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
        p_experimentStatusLabel->setText("Valid experiment found");
        styleStatusLabel(p_experimentStatusLabel, ThemeColors::StatusSuccess, false);
    } else {
        QString errorMessage;
        validateExperimentPath(getExperimentPath(), errorMessage);
        p_experimentStatusLabel->setText(errorMessage);
        styleStatusLabel(p_experimentStatusLabel, ThemeColors::StatusError, false);
    }
    p_experimentStatusLabel->setToolTip(p_experimentStatusLabel->text());
}

void BCExpOverlayWidget::updateFtStatus()
{
    if (d_hasFtData) {
        p_configureFtButton->setText("FT Configured");
        QPalette bpal = p_configureFtButton->palette();
        bpal.setColor(QPalette::ButtonText,
                      ThemeColors::getThemeAwareColor(ThemeColors::StatusSuccess, this));
        p_configureFtButton->setPalette(bpal);
        p_ftStatusLabel->setText("FT data configured and ready.");
        styleStatusLabel(p_ftStatusLabel, ThemeColors::StatusSuccess, false);
    } else {
        p_configureFtButton->setText("Configure FT...");
        p_configureFtButton->setPalette(QPalette());
        p_ftStatusLabel->setText("Click to configure FT processing settings for this overlay.");
        styleStatusLabel(p_ftStatusLabel, ThemeColors::SubtleText, true);
    }
    p_ftStatusLabel->setToolTip(p_ftStatusLabel->text());
}

void BCExpOverlayWidget::updateAutomaticLabel()
{
    if (p_usePathCheckBox->isChecked()) {
        // Using custom path - update label from directory name if available
        QString path = p_pathLineEdit->text();
        if (!path.isEmpty()) {
            QDir dir(path);
            QString dirName = dir.dirName();
            if (!dirName.isEmpty()) {
                emit labelUpdateRequested(dirName);
            }
        }
    } else {
        // Using experiment number - update label from number
        int number = p_experimentNumberSpinBox->value();
        emit labelUpdateRequested(QString("Exp%1").arg(number));
    }
}

void BCExpOverlayWidget::configureForCreationContext()
{
    // Creation context: Emphasize experiment selection and FT configuration
    if (p_experimentNumberSpinBox) {
        // Set focus on latest experiment for quick access
        SettingsStorage s;
        int lastExperiment = s.get(BC::Key::exptNum, 1);
        p_experimentNumberSpinBox->setValue(lastExperiment);
    }
    
    // Show helpful status message
    if (p_experimentStatusLabel) {
        p_experimentStatusLabel->setText("Select an experiment to create overlay");
        styleStatusLabel(p_experimentStatusLabel, ThemeColors::SubtleText, true);
    }

    // Emphasize FT configuration requirement
    if (p_configureFtButton) {
        QFont bf = p_configureFtButton->font();
        bf.setBold(true);
        p_configureFtButton->setFont(bf);
    }

    if (p_ftStatusLabel) {
        p_ftStatusLabel->setText("FT configuration required - click 'Configure FT...' to begin");
        styleStatusLabel(p_ftStatusLabel, ThemeColors::StatusWarning, true);
    }
}

void BCExpOverlayWidget::configureForSettingsContext()
{
    // Settings context: Show existing overlay information and focus on modifications
    if (d_overlay) {
        auto bcexpOverlay = std::dynamic_pointer_cast<BCExpOverlay>(d_overlay);
        if (bcexpOverlay) {
            // Show current overlay information
            QString sourceFile = bcexpOverlay->getSourceFile();
            QDir sourceDir(sourceFile);
            QString info = QString("Editing: %1").arg(sourceDir.dirName());
            
            if (p_experimentStatusLabel) {
                p_experimentStatusLabel->setText(info);
                styleStatusLabel(p_experimentStatusLabel, ThemeColors::StatusInfo, false);
            }

            // Show FT configuration status
            if (!bcexpOverlay->getFtData().isEmpty() && p_ftStatusLabel) {
                const auto& ft = bcexpOverlay->getFtData();
                QString ftInfo = QString("FT configured: %1 MHz to %2 MHz (%3 points)")
                    .arg(ft.minFreqMHz(), 0, 'f', 1)
                    .arg(ft.maxFreqMHz(), 0, 'f', 1)
                    .arg(ft.size());
                p_ftStatusLabel->setText(ftInfo);
                styleStatusLabel(p_ftStatusLabel, ThemeColors::StatusSuccess, false);
            }
        }
    }

    // Reduce emphasis on configuration button in settings mode
    if (p_configureFtButton) {
        QFont bf = p_configureFtButton->font();
        bf.setBold(false);
        p_configureFtButton->setFont(bf);
    }
}

void BCExpOverlayWidget::populateSourceFileConfigRows(SettingsTable *table)
{
    // Fixed source-file selector: experiment number / custom path,
    // plus a single status line. The base has already added the
    // checkable section row above.

    // Experiment number selection (spinbox + "OR" + custom-path toggle
    // in a single value cell).
    SettingsStorage s;
    int lastExperiment = s.get(BC::Key::exptNum, 1);
    p_experimentNumberSpinBox = new QSpinBox(table);
    p_experimentNumberSpinBox->setMinimum(1);
    p_experimentNumberSpinBox->setMaximum(lastExperiment);
    p_experimentNumberSpinBox->setValue(1);
    p_experimentNumberSpinBox->setMinimumWidth(80);
    p_experimentNumberSpinBox->setSizePolicy(QSizePolicy::Expanding,
                                             QSizePolicy::Fixed);

    p_usePathCheckBox = new QCheckBox("Use custom path", table);

    auto experimentCell = new QWidget(table);
    auto experimentRow = new QHBoxLayout(experimentCell);
    experimentRow->setContentsMargins(0, 0, 0, 0);
    experimentRow->setSpacing(6);
    experimentRow->addWidget(p_experimentNumberSpinBox, 1);
    experimentRow->addWidget(new QLabel("OR", experimentCell));
    experimentRow->addWidget(p_usePathCheckBox);
    table->addSettingRow("Experiment", experimentCell);

    // Path selection
    p_pathLineEdit = new QLineEdit(table);
    p_pathLineEdit->setEnabled(false);
    p_pathLineEdit->setPlaceholderText("Select experiment directory...");

    p_browseButton = new QToolButton(table);
    p_browseButton->setIcon(ThemeColors::createThemedIcon(
        ":/icons/folder-open.svg", ThemeColors::IconSecondary, this));
    p_browseButton->setToolTip("Browse for experiment directory");
    p_browseButton->setEnabled(false);

    table->addSettingRow("Path", p_pathLineEdit, p_browseButton);

    // Status display (compact, single line with icon)
    p_experimentStatusLabel = new QLabel(table);
    p_experimentStatusLabel->setWordWrap(false);
    p_experimentStatusLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    p_experimentStatusLabel->setMinimumHeight(20);
    p_experimentStatusLabel->setMaximumHeight(22);
    p_experimentStatusLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    table->addSettingRow("Status", p_experimentStatusLabel);
}

void BCExpOverlayWidget::populateSourceFileSettingsRows(SettingsTable *table)
{
    // No "FT Configuration" heading: the base already adds the
    // "Source File Settings" tier heading above these rows.
    p_configureFtButton = new QPushButton("Configure FT...", table);
    p_configureFtButton->setMinimumHeight(30);
    table->addSettingRow("Processing", p_configureFtButton);

    p_ftStatusLabel = new QLabel("Click to configure FT processing settings for this overlay.", table);
    p_ftStatusLabel->setWordWrap(false);
    p_ftStatusLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    p_ftStatusLabel->setToolTip(p_ftStatusLabel->text());
    styleStatusLabel(p_ftStatusLabel, ThemeColors::SubtleText, true);
    table->addSettingRow("Status", p_ftStatusLabel);
}

void BCExpOverlayWidget::populateTypeSpecificRows(SettingsTable *)
{
    // BC experiments have no overlay-specific settings
    // (hasTypeSpecificSettings() returns false, so the base never adds
    // the section and this is not called; defined to satisfy the
    // pure-virtual contract).
}
