#include "genericxyoverlaywidget.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QHeaderView>
#include <QFormLayout>
#include <QGridLayout>
#include <QSplitter>
#include <QApplication>
#include <QStandardPaths>

#include <data/experiment/overlaytypes.h>
#include <data/storage/settingsstorage.h>
#include <data/processing/parsers/fileparserregistry.h>
#include <data/processing/parsers/genericxyparser.h>

namespace BC::Key::GenericXYWidget {
static const QString key{"GenericXYOverlayWidget"};
static const QString lastFilePath{"lastFilePath"};
static const QString delimiter{"delimiter"};
static const QString headerLines{"headerLines"};
static const QString xColumn{"xColumn"};
static const QString yColumn{"yColumn"};
static const QString enableFiltering{"enableFiltering"};
static const QString xMin{"xMin"};
static const QString xMax{"xMax"};

// Additional keys for settings hash (not persisted to settings storage)
static const QString filePath{"filePath"};
static const QString fileValid{"fileValid"};
static const QString dataPoints{"dataPoints"};
static const QString columnNames{"columnNames"};
}

GenericXYOverlayWidget::GenericXYOverlayWidget(const Ft &currentFt, QWidget *parent)
    : OverlayTypeSpecificWidget(currentFt, parent)
    , SettingsStorage(BC::Key::GenericXYWidget::key)
    , d_dataValid(false)
    , d_settingsLoaded(false)
    , p_sourceFileConfigWidget(nullptr)
    , p_sourceFileSettingsWidget(nullptr)
    , p_typeSpecificWidget(nullptr)
    , d_fileAnalyzed(false)
{
    setupUI();
    setupConnections();
}

GenericXYOverlayWidget::~GenericXYOverlayWidget() = default;

void GenericXYOverlayWidget::setupForCreation()
{
    d_context = Context::Creation;
    d_overlay.reset();
    loadSettings();
    resetToDefaults();
}

void GenericXYOverlayWidget::setupForSettings(std::shared_ptr<OverlayBase> overlay)
{
    d_context = Context::Settings;
    d_overlay = overlay;
    
    auto genericXYOverlay = std::dynamic_pointer_cast<GenericXYOverlay>(overlay);
    if (!genericXYOverlay) {
        qWarning() << "GenericXYOverlayWidget::setupForSettings: Invalid overlay type";
        return;
    }
    
    // Load settings from overlay using the actual GenericXYOverlay interface
    d_sourceFilePath = genericXYOverlay->getSourceFile();
    p_filePathEdit->setText(d_sourceFilePath);
    
    // Update UI controls with overlay settings
    QString delimiter = genericXYOverlay->delimiter();
    int headerLines = genericXYOverlay->headerLines();
    int xColumn = genericXYOverlay->xColumn();
    int yColumn = genericXYOverlay->yColumn();
    
    // Update delimiter combo
    for (int i = 0; i < p_delimiterCombo->count(); ++i) {
        if (p_delimiterCombo->itemData(i).toString() == delimiter) {
            p_delimiterCombo->setCurrentIndex(i);
            break;
        }
    }
    
    p_headerLinesSpinBox->setValue(headerLines);
    p_xColumnCombo->setCurrentIndex(xColumn);
    p_yColumnCombo->setCurrentIndex(yColumn);
    
    // Load filtering settings from overlay
    double filterMinX = genericXYOverlay->filterMinX();
    double filterMaxX = genericXYOverlay->filterMaxX();
    bool hasFiltering = (filterMinX != 0.0 || filterMaxX != 1000.0); // Check if non-default values
    
    p_enableFilteringCheckBox->setChecked(hasFiltering);
    if (hasFiltering) {
        p_xMinEdit->setText(QString::number(filterMinX, 'g', 6));
        p_xMaxEdit->setText(QString::number(filterMaxX, 'g', 6));
    }
    
    // Reload and analyze file if valid
    if (!d_sourceFilePath.isEmpty() && validateFileExists()) {
        loadAndAnalyzeFile();
    }
}

std::shared_ptr<OverlayBase> GenericXYOverlayWidget::createOverlay()
{
    if (!isDataValid()) {
        return nullptr;
    }
    
    auto overlay = std::make_shared<GenericXYOverlay>();
    
    // Set source file and parser settings
    overlay->setSourceFile(d_sourceFilePath);
    overlay->setDelimiter(p_delimiterCombo->currentData().toString());
    overlay->setHeaderLines(p_headerLinesSpinBox->value());
    overlay->setDataColumns(p_xColumnCombo->currentData().toInt(), p_yColumnCombo->currentData().toInt());
    
    // Set data
    overlay->setRawData(d_parsedData.data());
    
    // Set column names if detected
    if (!d_detectedColumnNames.isEmpty()) {
        overlay->setColumnNames(d_detectedColumnNames);
    }
    
    // Data range is calculated automatically by the overlay
    
    return overlay;
}

void GenericXYOverlayWidget::applyToOverlay(std::shared_ptr<OverlayBase> overlay) const
{
    auto genericXYOverlay = std::dynamic_pointer_cast<GenericXYOverlay>(overlay);
    if (!genericXYOverlay) {
        return;
    }
    
    // Apply all current settings to the overlay
    genericXYOverlay->setSourceFile(d_sourceFilePath);
    genericXYOverlay->setDelimiter(p_delimiterCombo->currentData().toString());
    genericXYOverlay->setHeaderLines(p_headerLinesSpinBox->value());
    genericXYOverlay->setDataColumns(p_xColumnCombo->currentData().toInt(), p_yColumnCombo->currentData().toInt());
    
    // Apply data if valid
    if (d_dataValid) {
        genericXYOverlay->setRawData(d_parsedData.data());
        
        // Set column names
        if (!d_detectedColumnNames.isEmpty()) {
            genericXYOverlay->setColumnNames(d_detectedColumnNames);
        }
    }
    
    // Apply filtering settings
    if (p_enableFilteringCheckBox->isChecked()) {
        bool xMinOk, xMaxOk;
        double xMin = p_xMinEdit->text().toDouble(&xMinOk);
        double xMax = p_xMaxEdit->text().toDouble(&xMaxOk);
        
        if (xMinOk && xMaxOk && xMin < xMax) {
            genericXYOverlay->setFilterRange(xMin, xMax);
        }
    }
}

bool GenericXYOverlayWidget::validateSettings(QString &errorMessage) const
{
    if (!validateFileExists()) {
        errorMessage = "Source file does not exist or is not readable";
        return false;
    }
    
    if (!validateColumns()) {
        errorMessage = "Invalid column selection";
        return false;
    }
    
    if (!d_dataValid) {
        errorMessage = "Data could not be parsed or is invalid";
        return false;
    }
    
    if (p_enableFilteringCheckBox->isChecked()) {
        if (!validateDataRange()) {
            errorMessage = "Invalid filtering range";
            return false;
        }
    }
    
    return true;
}

bool GenericXYOverlayWidget::isDataValid() const
{
    return d_dataValid && !d_parsedData.data().isEmpty();
}

bool GenericXYOverlayWidget::hasValidSourceFile() const
{
    return !d_sourceFilePath.isEmpty() && validateFileExists();
}

QString GenericXYOverlayWidget::getSourceFilePath() const
{
    return d_sourceFilePath;
}

void GenericXYOverlayWidget::setSourceFilePath(const QString &path)
{
    if (d_sourceFilePath != path) {
        d_sourceFilePath = path;
        p_filePathEdit->setText(path);
        
        if (!path.isEmpty()) {
            loadAndAnalyzeFile();
        } else {
            d_dataValid = false;
            d_fileAnalyzed = false;
            emit fileChanged();
            emit sourceFileChanged();
        }
    }
}

bool GenericXYOverlayWidget::validateSourceFile(QString &errorMessage)
{
    if (!validateFileExists()) {
        errorMessage = "File does not exist or is not readable";
        return false;
    }
    
    if (!d_fileAnalyzed) {
        loadAndAnalyzeFile();
    }
    
    if (!d_dataValid) {
        errorMessage = "File format could not be detected or data is invalid";
        return false;
    }
    
    return true;
}

void GenericXYOverlayWidget::resetToDefaults()
{
    // Reset UI to default settings
    p_delimiterCombo->setCurrentIndex(0); // Comma
    p_headerLinesSpinBox->setValue(0);
    
    // Reset column selectors to defaults
    if (p_xColumnCombo->count() > 0) {
        p_xColumnCombo->setCurrentIndex(0);
    }
    if (p_yColumnCombo->count() > 1) {
        p_yColumnCombo->setCurrentIndex(1);
    }
    
    // Reset filtering
    p_enableFilteringCheckBox->setChecked(false);
    p_xMinEdit->clear();
    p_xMaxEdit->clear();
    
    // Clear file selection in creation context
    if (d_context == Context::Creation) {
        d_sourceFilePath.clear();
        p_filePathEdit->clear();
        d_dataValid = false;
        d_fileAnalyzed = false;
    }
    
    // Update preview
    updatePreview();
}

QHash<QString, QVariant> GenericXYOverlayWidget::getSettingsHash() const
{
    QHash<QString, QVariant> hash;
    
    // File and parsing settings
    hash[BC::Key::GenericXYWidget::filePath] = d_sourceFilePath;
    hash[BC::Key::GenericXYWidget::fileValid] = d_dataValid;
    hash[BC::Key::GenericXYWidget::delimiter] = p_delimiterCombo->currentData().toString();
    hash[BC::Key::GenericXYWidget::headerLines] = p_headerLinesSpinBox->value();
    hash[BC::Key::GenericXYWidget::xColumn] = p_xColumnCombo->currentData().toInt();
    hash[BC::Key::GenericXYWidget::yColumn] = p_yColumnCombo->currentData().toInt();
    
    // Filtering settings
    hash[BC::Key::GenericXYWidget::enableFiltering] = p_enableFilteringCheckBox->isChecked();
    hash[BC::Key::GenericXYWidget::xMin] = p_xMinEdit->text();
    hash[BC::Key::GenericXYWidget::xMax] = p_xMaxEdit->text();
    
    // Data information
    hash[BC::Key::GenericXYWidget::dataPoints] = d_parsedData.data().size();
    hash[BC::Key::GenericXYWidget::columnNames] = d_detectedColumnNames;
    
    return hash;
}

QVector<OperationCapability> GenericXYOverlayWidget::getSupportedOperations() const
{
    return {}; // Simplified for now
}

bool GenericXYOverlayWidget::supportsBackgroundOperation(OperationCapability::Type type) const
{
    return false; // Simplified for now
}

std::shared_ptr<OverlayOperation> GenericXYOverlayWidget::createOperation(OperationCapability::Type type,
                                                                         std::shared_ptr<OverlayBase> overlay) const
{
    return nullptr; // Simplified for now
}

QWidget* GenericXYOverlayWidget::getSourceFileConfigWidget()
{
    return p_sourceFileConfigWidget;
}

QWidget* GenericXYOverlayWidget::getSourceFileSettingsWidget()
{
    return p_sourceFileSettingsWidget;
}

QWidget* GenericXYOverlayWidget::getOverlaySettingsWidget()
{
    return p_typeSpecificWidget;
}

void GenericXYOverlayWidget::onFileSelected()
{
    QString selectedPath = p_filePathEdit->text().trimmed();
    if (selectedPath != d_sourceFilePath) {
        setSourceFilePath(selectedPath);
    }
}

void GenericXYOverlayWidget::onFormatDetectionRequested()
{
    if (!d_sourceFilePath.isEmpty()) {
        detectFileFormat();
        parseAndPreview();
    }
}

void GenericXYOverlayWidget::onParseSettingsChanged()
{
    parseAndPreview();
    saveSettings();
    emit settingsChanged();
}

void GenericXYOverlayWidget::updatePreview()
{
    if (!p_previewTable || !d_dataValid) {
        return;
    }
    
    // Clear and setup table
    p_previewTable->clear();
    p_previewTable->setRowCount(0);
    p_previewTable->setColumnCount(0);
    
    if (d_parsedData.data().isEmpty()) {
        return;
    }
    
    // Setup table with data preview (limit to first 100 rows)
    const int maxPreviewRows = 100;
    const int numRows = qMin(d_parsedData.data().size(), maxPreviewRows);
    
    p_previewTable->setColumnCount(2);
    p_previewTable->setRowCount(numRows);
    
    // Set column headers
    QStringList headers;
    int xColumn = p_xColumnCombo->currentData().toInt();
    int yColumn = p_yColumnCombo->currentData().toInt();
    
    if (xColumn < d_detectedColumnNames.size()) {
        headers << d_detectedColumnNames[xColumn];
    } else {
        headers << QString("X (Col %1)").arg(xColumn + 1);
    }
    
    if (yColumn < d_detectedColumnNames.size()) {
        headers << d_detectedColumnNames[yColumn];
    } else {
        headers << QString("Y (Col %1)").arg(yColumn + 1);
    }
    
    p_previewTable->setHorizontalHeaderLabels(headers);
    
    // Fill with data
    for (int i = 0; i < numRows; ++i) {
        const QPointF &point = d_parsedData.data()[i];
        p_previewTable->setItem(i, 0, new QTableWidgetItem(QString::number(point.x(), 'g', 6)));
        p_previewTable->setItem(i, 1, new QTableWidgetItem(QString::number(point.y(), 'g', 6)));
    }
    
    // Auto-resize columns
    p_previewTable->resizeColumnsToContents();
    
    // Update statistics
    updateDataStatistics();
}

void GenericXYOverlayWidget::onFileDialogRequested()
{
    QString startDir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    if (!d_sourceFilePath.isEmpty()) {
        startDir = QFileInfo(d_sourceFilePath).absolutePath();
    }
    
    QString selectedFile = QFileDialog::getOpenFileName(
        this,
        "Select Generic XY Data File",
        startDir,
        "Data Files (*.csv *.tsv *.txt *.dat);;All Files (*)"
    );
    
    if (!selectedFile.isEmpty()) {
        setSourceFilePath(selectedFile);
    }
}

void GenericXYOverlayWidget::onDelimiterChanged()
{
    onParseSettingsChanged();
}

void GenericXYOverlayWidget::onHeaderLinesChanged()
{
    onParseSettingsChanged();
}

void GenericXYOverlayWidget::onColumnSelectionChanged()
{
    onParseSettingsChanged();
}

void GenericXYOverlayWidget::onAutoDetectClicked()
{
    onFormatDetectionRequested();
}

void GenericXYOverlayWidget::setupUI()
{
    // Main layout will be handled by UnifiedOverlayWidget
    // We just create the individual widget components
    
    createSourceFileConfigUI();
    createSourceFileSettingsUI();
    createTypeSpecificSettingsUI();
}

void GenericXYOverlayWidget::setupConnections()
{
    // Source file config connections
    connect(p_filePathEdit, &QLineEdit::editingFinished, this, &GenericXYOverlayWidget::onFileSelected);
    connect(p_browseButton, &QPushButton::clicked, this, &GenericXYOverlayWidget::onFileDialogRequested);
    connect(p_autoDetectButton, &QPushButton::clicked, this, &GenericXYOverlayWidget::onAutoDetectClicked);
    
    // Source file settings connections
    connect(p_delimiterCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GenericXYOverlayWidget::onDelimiterChanged);
    connect(p_headerLinesSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &GenericXYOverlayWidget::onHeaderLinesChanged);
    connect(p_xColumnCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GenericXYOverlayWidget::onColumnSelectionChanged);
    connect(p_yColumnCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GenericXYOverlayWidget::onColumnSelectionChanged);
    
    // Filtering connections
    connect(p_enableFilteringCheckBox, &QCheckBox::toggled, this, &GenericXYOverlayWidget::onParseSettingsChanged);
    connect(p_xMinEdit, &QLineEdit::editingFinished, this, &GenericXYOverlayWidget::onParseSettingsChanged);
    connect(p_xMaxEdit, &QLineEdit::editingFinished, this, &GenericXYOverlayWidget::onParseSettingsChanged);
}

void GenericXYOverlayWidget::createSourceFileConfigUI()
{
    p_sourceFileConfigWidget = new QWidget(this);
    auto layout = new QFormLayout(p_sourceFileConfigWidget);
    
    // File path selection
    auto fileLayout = new QHBoxLayout();
    p_filePathEdit = new QLineEdit(p_sourceFileConfigWidget);
    p_filePathEdit->setPlaceholderText("Select a data file...");
    p_browseButton = new QPushButton("Browse...", p_sourceFileConfigWidget);
    p_autoDetectButton = new QPushButton("Auto-Detect", p_sourceFileConfigWidget);
    
    fileLayout->addWidget(p_filePathEdit, 1);
    fileLayout->addWidget(p_browseButton);
    fileLayout->addWidget(p_autoDetectButton);
    
    // File status label
    p_fileStatusLabel = new QLabel(p_sourceFileConfigWidget);
    p_fileStatusLabel->setWordWrap(true);
    
    layout->addRow("Data File:", fileLayout);
    layout->addRow("Status:", p_fileStatusLabel);
}

void GenericXYOverlayWidget::createSourceFileSettingsUI()
{
    p_sourceFileSettingsWidget = new QWidget(this);
    auto layout = new QFormLayout(p_sourceFileSettingsWidget);
    
    // Delimiter selection
    p_delimiterCombo = new QComboBox(p_sourceFileSettingsWidget);
    populateDelimiterComboBox();
    
    // Header lines
    p_headerLinesSpinBox = new QSpinBox(p_sourceFileSettingsWidget);
    p_headerLinesSpinBox->setRange(0, 100);
    p_headerLinesSpinBox->setValue(0);
    
    // Column selection
    p_xColumnCombo = new QComboBox(p_sourceFileSettingsWidget);
    p_yColumnCombo = new QComboBox(p_sourceFileSettingsWidget);
    
    // Filtering controls
    p_enableFilteringCheckBox = new QCheckBox("Enable X-range filtering", p_sourceFileSettingsWidget);
    
    auto filterLayout = new QHBoxLayout();
    p_xMinEdit = new QLineEdit(p_sourceFileSettingsWidget);
    p_xMinEdit->setPlaceholderText("Min X");
    p_xMaxEdit = new QLineEdit(p_sourceFileSettingsWidget);
    p_xMaxEdit->setPlaceholderText("Max X");
    
    filterLayout->addWidget(p_xMinEdit);
    filterLayout->addWidget(new QLabel("to"));
    filterLayout->addWidget(p_xMaxEdit);
    filterLayout->addStretch();
    
    // Data statistics
    p_dataStatsLabel = new QLabel(p_sourceFileSettingsWidget);
    p_dataStatsLabel->setWordWrap(true);
    
    layout->addRow("Delimiter:", p_delimiterCombo);
    layout->addRow("Header Lines:", p_headerLinesSpinBox);
    layout->addRow("X Column:", p_xColumnCombo);
    layout->addRow("Y Column:", p_yColumnCombo);
    layout->addRow(p_enableFilteringCheckBox);
    layout->addRow("Filter Range:", filterLayout);
    layout->addRow("Data Info:", p_dataStatsLabel);
    
    // Initially disable filtering controls
    p_xMinEdit->setEnabled(false);
    p_xMaxEdit->setEnabled(false);
    
    // Connect filtering checkbox to enable/disable controls
    connect(p_enableFilteringCheckBox, &QCheckBox::toggled, [this](bool enabled) {
        p_xMinEdit->setEnabled(enabled);
        p_xMaxEdit->setEnabled(enabled);
    });
}

void GenericXYOverlayWidget::createTypeSpecificSettingsUI()
{
    p_typeSpecificWidget = new QWidget(this);
    auto layout = new QVBoxLayout(p_typeSpecificWidget);
    
    // Preview table
    p_previewTable = new QTableWidget(p_typeSpecificWidget);
    p_previewTable->setAlternatingRowColors(true);
    p_previewTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    p_previewTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    p_previewTable->verticalHeader()->setVisible(false);
    p_previewTable->setMaximumHeight(200);
    
    layout->addWidget(new QLabel("Data Preview:"));
    layout->addWidget(p_previewTable);
    layout->addStretch();
}

void GenericXYOverlayWidget::loadAndAnalyzeFile()
{
    if (d_sourceFilePath.isEmpty() || !validateFileExists()) {
        d_dataValid = false;
        d_fileAnalyzed = false;
        p_fileStatusLabel->setText("No file selected or file not found");
        return;
    }
    
    // Get parser from registry
    auto parser = getParser();
    if (!parser) {
        d_dataValid = false;
        d_fileAnalyzed = true;
        p_fileStatusLabel->setText("No GenericXY parser available");
        return;
    }
    
    // Create parse settings from UI
    GenericXYParser::ParseSettings settings;
    settings.delimiter = p_delimiterCombo->currentData().toString();
    settings.headerLines = p_headerLinesSpinBox->value();
    settings.xColumn = p_xColumnCombo->currentData().toInt();
    settings.yColumn = p_yColumnCombo->currentData().toInt();
    
    // Parse the file
    GenericXYData result = parser->parseWithSettings(d_sourceFilePath, settings);
    
    if (result.isValid()) {
        d_parsedData = result;
        d_dataValid = true;
        d_fileAnalyzed = true;
        
        updateColumnSelectors();
        updatePreview();
        
        p_fileStatusLabel->setText(QString("Loaded %1 data points").arg(d_parsedData.data().size()));
    } else {
        d_dataValid = false;
        d_fileAnalyzed = true;
        p_fileStatusLabel->setText("Failed to parse file");
    }
    
    emit fileChanged();
    emit sourceFileChanged();
    emit dataValidityChanged(d_dataValid);
}

void GenericXYOverlayWidget::detectFileFormat()
{
    if (d_sourceFilePath.isEmpty()) {
        return;
    }
    
    // Auto-detect format using parser
    auto parser = getParser();
    if (!parser) {
        return;
    }
    
    GenericXYParser::ParseSettings settings = parser->autoDetectSettings(d_sourceFilePath);
    
    // Update UI with detected settings
    for (int i = 0; i < p_delimiterCombo->count(); ++i) {
        if (p_delimiterCombo->itemData(i).toString() == settings.delimiter) {
            p_delimiterCombo->setCurrentIndex(i);
            break;
        }
    }
    
    p_headerLinesSpinBox->setValue(settings.headerLines);
    d_detectedColumnNames = settings.columnNames;
    
    updateColumnSelectors();
}

void GenericXYOverlayWidget::updateColumnSelectors()
{
    // Update column combo boxes
    p_xColumnCombo->clear();
    p_yColumnCombo->clear();
    
    if (!d_detectedColumnNames.isEmpty()) {
        // Use detected column names
        for (int i = 0; i < d_detectedColumnNames.size(); ++i) {
            QString columnLabel = QString("%1 (%2)").arg(d_detectedColumnNames[i]).arg(i + 1);
            p_xColumnCombo->addItem(columnLabel, i);
            p_yColumnCombo->addItem(columnLabel, i);
        }
    } else {
        // Use generic column numbers
        for (int i = 0; i < 10; ++i) { // Assume max 10 columns for now
            QString columnLabel = QString("Column %1").arg(i + 1);
            p_xColumnCombo->addItem(columnLabel, i);
            p_yColumnCombo->addItem(columnLabel, i);
        }
    }
    
    // Set current selections to defaults
    if (p_xColumnCombo->count() > 0) {
        p_xColumnCombo->setCurrentIndex(0);
    }
    if (p_yColumnCombo->count() > 1) {
        p_yColumnCombo->setCurrentIndex(1);
    }
}

void GenericXYOverlayWidget::parseAndPreview()
{
    if (!d_sourceFilePath.isEmpty() && validateFileExists()) {
        loadAndAnalyzeFile();
    }
}

void GenericXYOverlayWidget::loadSettings()
{
    // Settings are loaded in the UI setup methods and resetToDefaults()
    // This is mainly a placeholder for consistency with the interface
}

void GenericXYOverlayWidget::saveSettings()
{
    // Save current file path for next dialog use
    set(BC::Key::GenericXYWidget::lastFilePath, d_sourceFilePath);
    
    // Save UI preferences (following CatalogOverlayWidget pattern)
    set(BC::Key::GenericXYWidget::delimiter, p_delimiterCombo->currentData().toString());
    set(BC::Key::GenericXYWidget::headerLines, p_headerLinesSpinBox->value());
    set(BC::Key::GenericXYWidget::xColumn, p_xColumnCombo->currentData().toInt());
    set(BC::Key::GenericXYWidget::yColumn, p_yColumnCombo->currentData().toInt());
    set(BC::Key::GenericXYWidget::enableFiltering, p_enableFilteringCheckBox->isChecked());
    set(BC::Key::GenericXYWidget::xMin, p_xMinEdit->text());
    set(BC::Key::GenericXYWidget::xMax, p_xMaxEdit->text());
}

// These methods are no longer needed since we don't maintain parser state

bool GenericXYOverlayWidget::validateFileExists() const
{
    return !d_sourceFilePath.isEmpty() && QFileInfo::exists(d_sourceFilePath);
}

bool GenericXYOverlayWidget::validateColumns() const
{
    int xCol = p_xColumnCombo->currentData().toInt();
    int yCol = p_yColumnCombo->currentData().toInt();
    return xCol >= 0 && yCol >= 0 && xCol != yCol;
}

bool GenericXYOverlayWidget::validateDataRange() const
{
    if (!p_enableFilteringCheckBox->isChecked()) {
        return true;
    }
    
    bool xMinOk, xMaxOk;
    double xMin = p_xMinEdit->text().toDouble(&xMinOk);
    double xMax = p_xMaxEdit->text().toDouble(&xMaxOk);
    
    return xMinOk && xMaxOk && xMin < xMax;
}

QString GenericXYOverlayWidget::getDelimiterDisplayName(const QString &delimiter) const
{
    if (delimiter == ",") return "Comma (,)";
    if (delimiter == "\t") return "Tab";
    if (delimiter == " ") return "Space";
    if (delimiter == ";") return "Semicolon (;)";
    return delimiter;
}

void GenericXYOverlayWidget::populateDelimiterComboBox()
{
    p_delimiterCombo->addItem("Comma (,)", ",");
    p_delimiterCombo->addItem("Tab", "\t");
    p_delimiterCombo->addItem("Space", " ");
    p_delimiterCombo->addItem("Semicolon (;)", ";");
}

void GenericXYOverlayWidget::updateDataStatistics()
{
    if (!d_dataValid || d_parsedData.data().isEmpty()) {
        p_dataStatsLabel->setText("No valid data");
        return;
    }
    
    QString stats = QString("%1 points | X: %2 to %3 | Y: %4 to %5")
                        .arg(d_parsedData.data().size())
                        .arg(d_parsedData.xMin(), 0, 'g', 4)
                        .arg(d_parsedData.xMax(), 0, 'g', 4)
                        .arg(d_parsedData.yMin(), 0, 'g', 4)
                        .arg(d_parsedData.yMax(), 0, 'g', 4);
    
    p_dataStatsLabel->setText(stats);
}

GenericXYParser* GenericXYOverlayWidget::getParser() const
{
    auto registry = FileParserRegistry::instance();
    return registry->findParserOfType<GenericXYParser>(d_sourceFilePath);
}