#include "genericxyoverlaywidget.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QHeaderView>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QSplitter>
#include <QApplication>
#include <QStandardPaths>

#include <data/experiment/overlaytypes.h>
#include <data/storage/settingsstorage.h>
#include <data/processing/parsers/fileparserregistry.h>
#include <data/processing/parsers/genericxyparser.h>
#include <gui/style/themecolors.h>

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
    , d_settingsLoaded(false)
    , d_fileAnalyzed(false)
{
    // Base class handles setupUI() and setupConnections()
}

GenericXYOverlayWidget::~GenericXYOverlayWidget() = default;

void GenericXYOverlayWidget::setupForCreation()
{
    d_context = Context::Creation;
    d_overlay.reset();
    loadSettings();
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
    updatePathDisplayAndTooltip(p_filePathEdit, genericXYOverlay->getSourceFile());
    
    // Get overlay settings first
    QString delimiter = genericXYOverlay->delimiter();
    int headerLines = genericXYOverlay->headerLines();
    int xColumn = genericXYOverlay->xColumn();
    int yColumn = genericXYOverlay->yColumn();
    
    // Load data directly from overlay (don't re-parse source file in settings context)
    d_parsedData.setData(genericXYOverlay->rawData());
    d_parsedData.setXColumn(xColumn);
    d_parsedData.setYColumn(yColumn);
    d_parsedData.setDelimiter(delimiter);
    d_parsedData.setHeaderLines(headerLines);
    
    // Set source file path for isValid() check - overlay is self-contained for viewing
    // Source file is only needed if user wants to reparse/reload data
    QString sourceFile = genericXYOverlay->getSourceFile();
    d_parsedData.setFilePath(sourceFile); // Set even if empty or non-existent
    // Data validity is automatically tracked by d_parsedData
    d_fileAnalyzed = true;
    
    // Set column names from overlay, with fallback if empty
    QStringList columnNames = genericXYOverlay->columnNames();
    if (columnNames.isEmpty()) {
        // Create fallback column names - we know we have at least X and Y columns
        int maxColumn = qMax(xColumn, yColumn);
        for (int i = 0; i <= maxColumn; ++i) {
            columnNames << QString("Column %1").arg(i + 1);
        }
    }
    
    // Set the column names in parsed data
    d_parsedData.setColumnNames(columnNames);
    
    // Update delimiter combo
    for (int i = 0; i < p_delimiterCombo->count(); ++i) {
        if (p_delimiterCombo->itemData(i).toString() == delimiter) {
            p_delimiterCombo->setCurrentIndex(i);
            break;
        }
    }
    
    p_headerLinesSpinBox->setValue(headerLines);
    
    // Update column selectors with current data (don't reset to defaults in settings context)
    updateColumnSelectors(false);
    
    // Now set the correct column selections from overlay
    // Find the correct index for each column value
    for (int i = 0; i < p_xColumnCombo->count(); ++i) {
        if (p_xColumnCombo->itemData(i).toInt() == xColumn) {
            p_xColumnCombo->setCurrentIndex(i);
            break;
        }
    }
    
    for (int i = 0; i < p_yColumnCombo->count(); ++i) {
        if (p_yColumnCombo->itemData(i).toInt() == yColumn) {
            p_yColumnCombo->setCurrentIndex(i);
            break;
        }
    }
    
    // Load filtering settings from overlay
    double filterMinX = genericXYOverlay->filterMinX();
    double filterMaxX = genericXYOverlay->filterMaxX();
    bool hasFiltering = (filterMinX != 0.0 || filterMaxX != 1000.0); // Check if non-default values
    
    p_enableFilteringCheckBox->setChecked(hasFiltering);
    if (hasFiltering) {
        p_xMinEdit->setText(QString::number(filterMinX, 'g', 6));
        p_xMaxEdit->setText(QString::number(filterMaxX, 'g', 6));
    }
    
    // Update UI status and preview with loaded data
    if (d_parsedData.isValid()) {
        p_fileStatusLabel->setText(QString("Loaded %1 data points from overlay").arg(d_parsedData.data().size()));
        p_fileStatusLabel->setStyleSheet(QString("QLabel { color: %1; }").arg(ThemeColors::getCSSColor(ThemeColors::StatusInfo, this)));
        updatePreview();
    } else {
        p_fileStatusLabel->setText("No data available in overlay");
        p_fileStatusLabel->setStyleSheet(QString("QLabel { color: %1; }").arg(ThemeColors::getCSSColor(ThemeColors::StatusError, this)));
    }
}

std::shared_ptr<OverlayBase> GenericXYOverlayWidget::createOverlay()
{
    if (!isDataValid()) {
        return nullptr;
    }
    
    auto overlay = std::make_shared<GenericXYOverlay>();
    
    // Set source file and parser settings - read from d_parsedData (authoritative source)
    overlay->setSourceFile(getStoredFullSourceFilePath());
    overlay->setDelimiter(d_parsedData.delimiter());
    overlay->setHeaderLines(d_parsedData.headerLines());
    overlay->setDataColumns(d_parsedData.xColumn(), d_parsedData.yColumn());
    
    // Set data
    overlay->setRawData(d_parsedData.data());
    
    // Set column names from parsed data (authoritative source)
    overlay->setColumnNames(d_parsedData.columnNames());
    
    // Data range is calculated automatically by the overlay
    
    return overlay;
}

void GenericXYOverlayWidget::applyToOverlay(std::shared_ptr<OverlayBase> overlay) const
{
    auto genericXYOverlay = std::dynamic_pointer_cast<GenericXYOverlay>(overlay);
    if (!genericXYOverlay) {
        return;
    }
    
    // Apply all current settings to the overlay - read from d_parsedData (authoritative source)
    genericXYOverlay->setSourceFile(getStoredFullSourceFilePath());
    genericXYOverlay->setDelimiter(d_parsedData.delimiter());
    genericXYOverlay->setHeaderLines(d_parsedData.headerLines());
    genericXYOverlay->setDataColumns(d_parsedData.xColumn(), d_parsedData.yColumn());
    
    // Apply data if valid
    if (d_parsedData.isValid()) {
        genericXYOverlay->setRawData(d_parsedData.data());
        
        // Set column names from parsed data (authoritative source)
        genericXYOverlay->setColumnNames(d_parsedData.columnNames());
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

bool GenericXYOverlayWidget::validateSettingsImpl()
{
    if (!validateFileExists()) {
        setSettingsErrorMessage("Source file does not exist or is not readable");
        return false;
    }
    
    if (!validateColumns()) {
        setSettingsErrorMessage("Invalid column selection");
        return false;
    }
    
    if (!d_parsedData.isValid()) {
        setSettingsErrorMessage("Data could not be parsed or is invalid");
        return false;
    }
    
    if (p_enableFilteringCheckBox->isChecked()) {
        if (!validateDataRange()) {
            setSettingsErrorMessage("Invalid filtering range");
            return false;
        }
    }
    
    return true;
}

bool GenericXYOverlayWidget::isDataValid() const
{
    return d_parsedData.isValid() && !d_parsedData.data().isEmpty();
}

bool GenericXYOverlayWidget::hasValidSourceFile() const
{
    QString path = getStoredFullSourceFilePath();
    return !path.isEmpty() && QFileInfo::exists(path);
}

QString GenericXYOverlayWidget::getSourceFilePath() const
{
    return getStoredFullSourceFilePath();
}

void GenericXYOverlayWidget::setSourceFilePath(const QString &path)
{
    updatePathDisplayAndTooltip(p_filePathEdit, path);
    
    // Update UI state based on file validity
    if (hasValidSourceFile()) {
        p_parseButton->setEnabled(true);
        
        // Generate a label from the filename for overlaybaseoptionswidget
        QFileInfo fileInfo(path);
        QString baseName = fileInfo.completeBaseName(); // Gets filename without extension
        if (!baseName.isEmpty()) {
            emit labelUpdateRequested(baseName);
        }
        
        // Trigger auto-detection when file changes (first call occasion)
        analyzeAndParseFile(true);
    } else {
        p_parseButton->setEnabled(false);
        d_parsedData.clear(); // Clears data, making isValid() return false
        d_fileAnalyzed = false;
    }
}

bool GenericXYOverlayWidget::validateSourceFileImpl()
{
    if (!validateFileExists()) {
        setSourceFileErrorMessage("File does not exist or is not readable");
        return false;
    }
    
    if (!d_fileAnalyzed) {
        analyzeAndParseFile(false); // Use current UI settings
    }
    
    if (!d_parsedData.isValid()) {
        setSourceFileErrorMessage("File format could not be detected or data is invalid");
        return false;
    }
    
    return true;
}


QHash<QString, QVariant> GenericXYOverlayWidget::getSettingsHash() const
{
    QHash<QString, QVariant> hash;
    
    // File and parsing settings
    hash[BC::Key::GenericXYWidget::fileValid] = d_parsedData.isValid();
    hash[BC::Key::GenericXYWidget::delimiter] = p_delimiterCombo->currentData().toString();
    hash[BC::Key::GenericXYWidget::headerLines] = p_headerLinesSpinBox->value();
    hash[BC::Key::GenericXYWidget::xColumn] = p_xColumnCombo->currentIndex();
    hash[BC::Key::GenericXYWidget::yColumn] = p_yColumnCombo->currentIndex();
    
    // Filtering settings
    hash[BC::Key::GenericXYWidget::enableFiltering] = p_enableFilteringCheckBox->isChecked();
    hash[BC::Key::GenericXYWidget::xMin] = p_xMinEdit->text();
    hash[BC::Key::GenericXYWidget::xMax] = p_xMaxEdit->text();
    
    // File and data information
    hash[BC::Key::GenericXYWidget::filePath] = getStoredFullSourceFilePath();
    hash[BC::Key::GenericXYWidget::dataPoints] = d_parsedData.data().size();
    hash[BC::Key::GenericXYWidget::columnNames] = d_parsedData.columnNames();
    
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




void GenericXYOverlayWidget::updatePreview()
{
    if (!p_previewTable || !d_parsedData.isValid()) {
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
    int xColumn = p_xColumnCombo->currentIndex();
    int yColumn = p_yColumnCombo->currentIndex();
    
    // Use column names from parsed data (with fallbacks if needed)
    QStringList columnNames = d_parsedData.columnNames();
    if (xColumn < columnNames.size()) {
        headers << columnNames[xColumn];
    } else {
        headers << QString("X (Col %1)").arg(xColumn + 1);
    }
    
    if (yColumn < columnNames.size()) {
        headers << columnNames[yColumn];
    } else {
        headers << QString("Y (Col %1)").arg(yColumn + 1);
    }
    
    p_previewTable->setHorizontalHeaderLabels(headers);
    
    // Fill with data
    auto d = d_parsedData.data();
    for (int i = 0; i < numRows; ++i) {
        const QPointF &point = d[i];
        p_previewTable->setItem(i, 0, new QTableWidgetItem(QString::number(point.x(), 'g', 6)));
        p_previewTable->setItem(i, 1, new QTableWidgetItem(QString::number(point.y(), 'g', 6)));
    }
    
    // Columns will automatically stretch to fill available space
    // due to QHeaderView::Stretch resize mode set during initialization
    
    // Update statistics
    updateDataStatistics();
}

void GenericXYOverlayWidget::onFileDialogRequested()
{
    QString startDir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    auto p = getStoredFullSourceFilePath();
    if (!p.isEmpty()) {
        startDir = QFileInfo(p).absolutePath();
    }
    else
        startDir = get(BC::Key::GenericXYWidget::lastFilePath,startDir);
    
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

void GenericXYOverlayWidget::onAutoDetectClicked()
{
    if (hasValidSourceFile()) {
        analyzeAndParseFile(true); // Autodetect and parse
    }
}

void GenericXYOverlayWidget::onFilteringChanged()
{
    // Save filtering settings and emit signals without re-parsing
    saveSettings();
    emit settingsChanged();
}


void GenericXYOverlayWidget::setupConnections()
{
    // Source file config connections
    connect(p_filePathEdit, &QLineEdit::editingFinished, [this]() {
        // File path changed, use the setter which handles validation and auto-detection
        setSourceFilePath(p_filePathEdit->text());
    });
    connect(p_browseButton, &QPushButton::clicked, this, &GenericXYOverlayWidget::onFileDialogRequested);
    connect(p_autoDetectButton, &QPushButton::clicked, this, &GenericXYOverlayWidget::onAutoDetectClicked);
    
    // Parse button triggers parsing with current UI settings
    connect(p_parseButton, &QPushButton::clicked, [this]() {
        analyzeAndParseFile(false); // Use current UI settings, don't autodetect
    });
    
    // Filtering connections - apply in real time since they don't require re-parsing
    connect(p_enableFilteringCheckBox, &QCheckBox::toggled, this, &GenericXYOverlayWidget::onFilteringChanged);
    connect(p_xMinEdit, &QLineEdit::editingFinished, this, &GenericXYOverlayWidget::onFilteringChanged);
    connect(p_xMaxEdit, &QLineEdit::editingFinished, this, &GenericXYOverlayWidget::onFilteringChanged);
}

void GenericXYOverlayWidget::configureForCreationContext()
{
    // Creation context: Enable auto-detection and emphasize preview
    if (p_autoDetectButton) {
        p_autoDetectButton->setVisible(true);
        p_autoDetectButton->setText("Auto-Detect Format");
    }
    
    // Enable all source file controls
    if (p_filePathEdit) p_filePathEdit->setEnabled(true);
    if (p_browseButton) p_browseButton->setEnabled(true);
    
    // Make preview more prominent in creation context
    if (p_previewTable) {
        p_previewTable->setMinimumHeight(150);
    }
    
    // Show helpful status messages
    if (p_fileStatusLabel) {
        p_fileStatusLabel->setText("Select a file and configure parsing to create overlay");
        p_fileStatusLabel->setStyleSheet(QString("QLabel { color: %1; font-style: italic; }").arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
    }
}

void GenericXYOverlayWidget::configureForSettingsContext()
{
    // Settings context: Focus on existing data, less emphasis on discovery
    if (p_autoDetectButton) {
        p_autoDetectButton->setText("Re-detect Format");
    }
    
    // Show current overlay information
    if (d_overlay && p_fileStatusLabel) {
        auto genericOverlay = std::dynamic_pointer_cast<GenericXYOverlay>(d_overlay);
        if (genericOverlay) {
            QString info = QString("Editing overlay: %1 data points").arg(genericOverlay->rawData().size());
            p_fileStatusLabel->setText(info);
            p_fileStatusLabel->setStyleSheet(QString("QLabel { color: %1; }").arg(ThemeColors::getCSSColor(ThemeColors::StatusInfo, this)));
        }
    }
    
    // Compact preview in settings context
    if (p_previewTable) {
        p_previewTable->setMaximumHeight(100);
    }
}

void GenericXYOverlayWidget::createSourceFileConfigUI(QGroupBox *parent)
{
    // Use compact horizontal layout similar to other overlay types
    auto mainLayout = new QVBoxLayout(parent);
    mainLayout->setSpacing(6);
    mainLayout->setContentsMargins(6, 6, 6, 6);
    
    // Primary file selection row
    auto fileRow = new QHBoxLayout();
    fileRow->setSpacing(6);
    
    // File path input (give adequate space for visibility)
    p_filePathEdit = new QLineEdit(parent);
    p_filePathEdit->setPlaceholderText("Select a data file...");
    p_filePathEdit->setMinimumWidth(250); // Ensure adequate width
    fileRow->addWidget(p_filePathEdit, 1); // Give it stretch priority
    
    // Browse button (compact with icon)
    p_browseButton = new QPushButton("📁", parent);
    p_browseButton->setToolTip("Browse for data file");
    p_browseButton->setMaximumSize(30, 30);
    fileRow->addWidget(p_browseButton);
    
    mainLayout->addLayout(fileRow);
    
    // Status display (compact, single line)
    auto statusRow = new QHBoxLayout();
    statusRow->setSpacing(6);
    
    p_fileStatusLabel = new QLabel(parent);
    p_fileStatusLabel->setWordWrap(false);
    p_fileStatusLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    p_fileStatusLabel->setMinimumHeight(20);
    p_fileStatusLabel->setMaximumHeight(22);
    p_fileStatusLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    p_fileStatusLabel->setStyleSheet("QLabel { font-size: 11px; padding: 2px; }");
    
    statusRow->addWidget(p_fileStatusLabel, 1);
    mainLayout->addLayout(statusRow);
}

void GenericXYOverlayWidget::createSourceFileSettingsUI(QGroupBox *parent)
{
    auto mainLayout = new QVBoxLayout(parent);
    mainLayout->setSpacing(4); // Reduced spacing between flat GroupBoxes
    
    // Group 1: Format Detection
    auto formatGroup = new QGroupBox("Format Detection", parent);
    formatGroup->setFlat(true); // Flat appearance for nested GroupBox
    auto formatLayout = new QGridLayout(formatGroup);
    formatLayout->setContentsMargins(3, 2, 3, 3); // Reduced margins for flat GroupBox
    formatLayout->setSpacing(3); // Reduced spacing for compact appearance
    
    p_delimiterCombo = new QComboBox(formatGroup);
    populateDelimiterComboBox();
    
    p_headerLinesSpinBox = new QSpinBox(formatGroup);
    p_headerLinesSpinBox->setRange(0, 100);
    p_headerLinesSpinBox->setValue(0);
    
    p_autoDetectButton = new QPushButton("Auto-Detect", formatGroup);
    p_autoDetectButton->setToolTip("Automatically detect delimiter, headers, and column structure");
    
    // Compact 2x2 grid layout
    auto delimiterLabel = new QLabel("Delimiter:", formatGroup);
    delimiterLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    formatLayout->addWidget(delimiterLabel, 0, 0);
    formatLayout->addWidget(p_delimiterCombo, 0, 1);
    auto headerLabel = new QLabel("Header Lines:", formatGroup);
    headerLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    formatLayout->addWidget(headerLabel, 1, 0);  
    formatLayout->addWidget(p_headerLinesSpinBox, 1, 1);
    formatLayout->addWidget(p_autoDetectButton, 0, 2, 2, 1); // Spans 2 rows
    
    // Group 2: Column Mapping  
    auto columnGroup = new QGroupBox("Column Mapping", parent);
    columnGroup->setFlat(true); // Flat appearance for nested GroupBox
    auto columnLayout = new QGridLayout(columnGroup);
    columnLayout->setContentsMargins(3, 2, 3, 3); // Reduced margins for flat GroupBox
    columnLayout->setSpacing(3); // Reduced spacing for compact appearance
    
    p_xColumnCombo = new QComboBox(columnGroup);
    p_yColumnCombo = new QComboBox(columnGroup);
    
    p_parseButton = new QPushButton("Parse File", columnGroup);
    p_parseButton->setEnabled(false);
    
    auto xColumnLabel = new QLabel("X Column:", columnGroup);
    xColumnLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    columnLayout->addWidget(xColumnLabel, 0, 0);
    columnLayout->addWidget(p_xColumnCombo, 0, 1);
    auto yColumnLabel = new QLabel("Y Column:", columnGroup);
    yColumnLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    columnLayout->addWidget(yColumnLabel, 1, 0);
    columnLayout->addWidget(p_yColumnCombo, 1, 1);
    columnLayout->addWidget(p_parseButton, 0, 2, 2, 1); // Spans 2 rows
    
    // Group 3: Data Filtering (Collapsible)
    auto filterGroup = new QGroupBox("Data Filtering", parent);
    filterGroup->setFlat(true); // Flat appearance for nested GroupBox
    filterGroup->setCheckable(true);
    filterGroup->setChecked(false);
    auto filterGroupLayout = new QVBoxLayout(filterGroup);
    filterGroupLayout->setContentsMargins(3, 2, 3, 3); // Reduced margins for flat GroupBox
    
    // Collapsible content widget
    auto filterContentWidget = new QWidget(parent);
    auto filterLayout = new QGridLayout(filterContentWidget);
    filterLayout->setContentsMargins(3, 2, 3, 3); // Reduced margins for flat GroupBox
    filterLayout->setSpacing(3); // Reduced spacing for compact appearance
    
    p_enableFilteringCheckBox = new QCheckBox("Enable X-range filtering", filterContentWidget);
    
    p_xMinEdit = new QLineEdit(filterContentWidget);
    p_xMinEdit->setPlaceholderText("Min X");
    p_xMaxEdit = new QLineEdit(filterContentWidget);
    p_xMaxEdit->setPlaceholderText("Max X");
    
    filterLayout->addWidget(p_enableFilteringCheckBox, 0, 0, 1, 4);
    auto rangeFilterLabel = new QLabel("Range:", filterContentWidget);
    rangeFilterLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    filterLayout->addWidget(rangeFilterLabel, 1, 0);
    filterLayout->addWidget(p_xMinEdit, 1, 1);
    auto toFilterLabel = new QLabel("to", filterContentWidget);
    toFilterLabel->setAlignment(Qt::AlignCenter | Qt::AlignVCenter);
    filterLayout->addWidget(toFilterLabel, 1, 2);
    filterLayout->addWidget(p_xMaxEdit, 1, 3);
    
    filterGroupLayout->addWidget(filterContentWidget);
    
    // Data statistics (compact info display)
    p_dataStatsLabel = new QLabel(parent);
    p_dataStatsLabel->setWordWrap(true);
    p_dataStatsLabel->setStyleSheet(QString("QLabel { color: %1; font-size: 11px; padding: 4px; }").arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
    
    // Add groups to main layout
    mainLayout->addWidget(formatGroup);
    mainLayout->addWidget(columnGroup);
    mainLayout->addWidget(filterGroup);
    mainLayout->addWidget(p_dataStatsLabel);
    mainLayout->addStretch();
    
    // Initially disable filtering controls
    p_xMinEdit->setEnabled(false);
    p_xMaxEdit->setEnabled(false);
    
    // Connect filtering controls
    connect(p_enableFilteringCheckBox, &QCheckBox::toggled, [this](bool enabled) {
        p_xMinEdit->setEnabled(enabled);
        p_xMaxEdit->setEnabled(enabled);
    });
    
    // Connect filter group checkable to enable filtering AND implement collapsible behavior
    connect(filterGroup, &QGroupBox::toggled, [this, filterContentWidget](bool enabled) {
        p_enableFilteringCheckBox->setChecked(enabled);
        filterContentWidget->setVisible(enabled);
    });
    
    // Initially hide filter content since group starts unchecked
    filterContentWidget->setVisible(false);
}

void GenericXYOverlayWidget::createTypeSpecificSettingsUI(QGroupBox *parent)
{
    auto layout = new QVBoxLayout(parent);
    
    // Preview table
    p_previewTable = new QTableWidget(parent);
    p_previewTable->setAlternatingRowColors(true);
    p_previewTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    p_previewTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    p_previewTable->verticalHeader()->setVisible(false);
    p_previewTable->setMaximumHeight(200);
    
    // Configure columns to stretch and fill evenly
    p_previewTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    
    layout->addWidget(new QLabel("Data Preview:"));
    layout->addWidget(p_previewTable);
    layout->addStretch();
}

void GenericXYOverlayWidget::analyzeAndParseFile(bool autodetect)
{
    QString filePath = getStoredFullSourceFilePath();
    if (filePath.isEmpty() || !QFileInfo::exists(filePath)) {
        d_parsedData.clear(); // Clears data, making isValid() return false
        d_fileAnalyzed = false;
        p_fileStatusLabel->setText("No file selected or file not found");
        p_parseButton->setEnabled(false);
        emit dataValidityChanged(false);
        return;
    }
    
    // Get parser from registry
    auto parser = getParser();
    if (!parser) {
        d_parsedData.clear(); // Clears data, making isValid() return false
        d_fileAnalyzed = true;
        p_fileStatusLabel->setText("No GenericXY parser available");
        p_parseButton->setEnabled(false);
        emit dataValidityChanged(false);
        return;
    }
    
    // Enable parse button since we have a valid file and parser
    p_parseButton->setEnabled(true);
    
    GenericXYParser::ParseSettings settings;
    
    // Handle autodetection if requested
    if (autodetect) {
        settings = parser->autoDetectSettings(filePath);
        
        // Update UI with detected settings
        for (int i = 0; i < p_delimiterCombo->count(); ++i) {
            if (p_delimiterCombo->itemData(i).toString() == settings.delimiter) {
                p_delimiterCombo->setCurrentIndex(i);
                break;
            }
        }
        
        p_headerLinesSpinBox->setValue(settings.headerLines);
        // Column names are now stored directly in d_parsedData
        
        updateColumnSelectors();
        
        // Apply user's saved column preferences if available and valid
        int savedXColumn = get(BC::Key::GenericXYWidget::xColumn, settings.xColumn);
        int savedYColumn = get(BC::Key::GenericXYWidget::yColumn, settings.yColumn);
        
        // Validate and set saved column selections (combo index = column index)
        if (savedXColumn >= 0 && savedXColumn < p_xColumnCombo->count()) {
            p_xColumnCombo->setCurrentIndex(savedXColumn);
        }
        
        if (savedYColumn >= 0 && savedYColumn < p_yColumnCombo->count()) {
            p_yColumnCombo->setCurrentIndex(savedYColumn);
        }
    } else {
        // Create parse settings from current UI state
        settings.delimiter = p_delimiterCombo->currentData().toString();
        settings.headerLines = p_headerLinesSpinBox->value();
        settings.xColumn = p_xColumnCombo->currentIndex();
        settings.yColumn = p_yColumnCombo->currentIndex();
    }
    
    // Parse the file with either detected or UI settings
    GenericXYData result = parser->parseWithSettings(filePath, settings);
    
    if (result.isValid()) {
        d_parsedData = result;
        // Data validity is now automatically tracked by d_parsedData
        d_fileAnalyzed = true;
        
        updateColumnSelectors();
        updatePreview();
        
        p_fileStatusLabel->setText(QString("Loaded %1 data points").arg(d_parsedData.data().size()));
        p_fileStatusLabel->setStyleSheet(QString("QLabel { color: %1; }").arg(ThemeColors::getCSSColor(ThemeColors::StatusSuccess, this)));
    } else {
        d_parsedData.clear(); // Clears data, making isValid() return false
        d_fileAnalyzed = true;
        p_fileStatusLabel->setText("Failed to parse file");
        p_fileStatusLabel->setStyleSheet(QString("QLabel { color: %1; }").arg(ThemeColors::getCSSColor(ThemeColors::StatusError, this)));
    }
    
    emit dataValidityChanged(d_parsedData.isValid());
    emit settingsChanged();
}


void GenericXYOverlayWidget::updateColumnSelectors(bool setDefaults)
{
    // Block signals to prevent infinite loops during UI updates
    const bool xColumnBlocked = p_xColumnCombo->blockSignals(true);
    const bool yColumnBlocked = p_yColumnCombo->blockSignals(true);
    
    // Update column combo boxes
    p_xColumnCombo->clear();
    p_yColumnCombo->clear();
    
    QStringList columnNames = d_parsedData.columnNames();
    if (!columnNames.isEmpty()) {
        // Use column names from parsed data
        for (int i = 0; i < columnNames.size(); ++i) {
            QString columnLabel = QString("%1 (%2)").arg(columnNames[i]).arg(i + 1);
            p_xColumnCombo->addItem(columnLabel);
            p_yColumnCombo->addItem(columnLabel);
        }
    } else {
        // Use generic column numbers
        for (int i = 0; i < 10; ++i) { // Assume max 10 columns for now
            QString columnLabel = QString("Column %1").arg(i + 1);
            p_xColumnCombo->addItem(columnLabel);
            p_yColumnCombo->addItem(columnLabel);
        }
    }
    
    // Set current selections to defaults only if requested
    if (setDefaults) {
        if (p_xColumnCombo->count() > 0) {
            p_xColumnCombo->setCurrentIndex(0);
        }
        if (p_yColumnCombo->count() > 1) {
            p_yColumnCombo->setCurrentIndex(1);
        }
    }
    
    // Restore signal state
    p_xColumnCombo->blockSignals(xColumnBlocked);
    p_yColumnCombo->blockSignals(yColumnBlocked);
}


void GenericXYOverlayWidget::loadSettings()
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
        p_filePathEdit->clear();
        updatePathDisplayAndTooltip(p_filePathEdit, QString()); // Clear stored path
        d_parsedData.clear(); // Clears data, making isValid() return false
        d_fileAnalyzed = false;
    }
    
    // Update preview
    updatePreview();
}

void GenericXYOverlayWidget::saveSettings()
{
    // Save current file path for next dialog use
    set(BC::Key::GenericXYWidget::lastFilePath, getStoredFullSourceFilePath());
    
    // Save UI preferences (following CatalogOverlayWidget pattern)
    set(BC::Key::GenericXYWidget::delimiter, p_delimiterCombo->currentData().toString());
    set(BC::Key::GenericXYWidget::headerLines, p_headerLinesSpinBox->value());
    set(BC::Key::GenericXYWidget::xColumn, p_xColumnCombo->currentIndex());
    set(BC::Key::GenericXYWidget::yColumn, p_yColumnCombo->currentIndex());
    set(BC::Key::GenericXYWidget::enableFiltering, p_enableFilteringCheckBox->isChecked());
    set(BC::Key::GenericXYWidget::xMin, p_xMinEdit->text());
    set(BC::Key::GenericXYWidget::xMax, p_xMaxEdit->text());
}

bool GenericXYOverlayWidget::validateFileExists() const
{
    QString path = getStoredFullSourceFilePath();
    return !path.isEmpty() && QFileInfo::exists(path);
}

bool GenericXYOverlayWidget::validateColumns() const
{
    int xCol = p_xColumnCombo->currentIndex();
    int yCol = p_yColumnCombo->currentIndex();
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
    if (!d_parsedData.isValid() || d_parsedData.data().isEmpty()) {
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
    return registry->findParserOfType<GenericXYParser>(getStoredFullSourceFilePath());
}
