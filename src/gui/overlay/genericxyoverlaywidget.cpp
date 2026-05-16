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
#include <gui/widget/settingstable.h>

namespace {
// Apply a themed foreground color to a label without a raw stylesheet
// color string.
void styleStatusLabel(QLabel *label, ThemeColors::ColorRole role,
                      bool italic = false)
{
    QPalette pal = label->palette();
    pal.setColor(QPalette::WindowText,
                 ThemeColors::getThemeAwareColor(role, label));
    label->setPalette(pal);
    QFont f = label->font();
    f.setItalic(italic);
    label->setFont(f);
}
}

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
    // The combo box index directly corresponds to the column index
    if (xColumn >= 0 && xColumn < p_xColumnCombo->count()) {
        p_xColumnCombo->setCurrentIndex(xColumn);
    }
    
    if (yColumn >= 0 && yColumn < p_yColumnCombo->count()) {
        p_yColumnCombo->setCurrentIndex(yColumn);
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
        styleStatusLabel(p_fileStatusLabel, ThemeColors::StatusInfo);
        updatePreview();
    } else {
        p_fileStatusLabel->setText("No data available in overlay");
        styleStatusLabel(p_fileStatusLabel, ThemeColors::StatusError);
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
    if (!hasValidSourceFile()) {
        QString filePath = getStoredFullSourceFilePath();
        if (filePath.isEmpty()) {
            setSettingsErrorMessage("No source file selected");
        } else {
            setSettingsErrorMessage(QString("Source file does not exist: %1").arg(filePath));
        }
        return false;
    }
    
    if (!validateColumns()) {
        if (!d_parsedData.isValid()) {
            setSettingsErrorMessage("Data not parsed - click Parse File button or Auto-Detect to analyze file");
        } else {
            int xCol = d_parsedData.xColumn();
            int yCol = d_parsedData.yColumn();
            if (xCol < 0 || yCol < 0) {
                setSettingsErrorMessage("Invalid column selection - X and Y columns must be valid");
            } else if (xCol == yCol) {
                setSettingsErrorMessage(QString("X and Y columns cannot be the same (both set to column %1)").arg(xCol + 1));
            } else {
                setSettingsErrorMessage("Invalid column configuration");
            }
        }
        return false;
    }
    
    if (!d_parsedData.isValid()) {
        setSettingsErrorMessage("Data could not be parsed - check file format and parsing settings");
        return false;
    }
    
    if (d_parsedData.data().isEmpty()) {
        setSettingsErrorMessage("No valid data points found in file");
        return false;
    }
    
    if (p_enableFilteringCheckBox->isChecked()) {
        if (!validateDataRange()) {
            bool xMinOk, xMaxOk;
            double xMin = p_xMinEdit->text().toDouble(&xMinOk);
            double xMax = p_xMaxEdit->text().toDouble(&xMaxOk);
            
            if (!xMinOk || !xMaxOk) {
                setSettingsErrorMessage("Invalid filtering range - min and max values must be numbers");
            } else if (xMin >= xMax) {
                setSettingsErrorMessage(QString("Invalid filtering range - min (%1) must be less than max (%2)").arg(xMin).arg(xMax));
            } else {
                setSettingsErrorMessage("Invalid filtering range configuration");
            }
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
        // Generate a label from the filename for overlaybaseoptionswidget
        QFileInfo fileInfo(path);
        QString baseName = fileInfo.completeBaseName(); // Gets filename without extension
        if (!baseName.isEmpty()) {
            emit labelUpdateRequested(baseName);
        }
        
        // Trigger auto-detection when file changes (first call occasion)
        analyzeAndParseFile(true);
    } else {
        d_parsedData.clear(); // Clears data, making isValid() return false
        d_fileAnalyzed = false;
    }
}

bool GenericXYOverlayWidget::validateSourceFileImpl()
{
    if (!hasValidSourceFile()) {
        QString filePath = getStoredFullSourceFilePath();
        if (filePath.isEmpty()) {
            setSourceFileErrorMessage("No source file selected");
        } else {
            setSourceFileErrorMessage(QString("Source file does not exist or is not readable: %1").arg(filePath));
        }
        return false;
    }
    
    if (!d_fileAnalyzed) {
        analyzeAndParseFile(false); // Use current UI settings
    }
    
    if (!d_parsedData.isValid()) {
        setSourceFileErrorMessage("File format could not be detected - check file format and parsing settings, or try Auto-Detect");
        return false;
    }
    
    if (d_parsedData.data().isEmpty()) {
        setSourceFileErrorMessage("No valid data points found in file");
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
    Q_UNUSED(type)
    return false; // Simplified for now
}

std::shared_ptr<OverlayOperation> GenericXYOverlayWidget::createOperation(OperationCapability::Type type,
                                                                         std::shared_ptr<OverlayBase> overlay) const
{
    Q_UNUSED(type)
    Q_UNUSED(overlay)
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
    // Emit signals without re-parsing (don't save settings here - only save on dialog accept)
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
    
    // Column selection changes should update parsed data and trigger revalidation
    connect(p_xColumnCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this]() {
        // Update parsed data column settings immediately
        d_parsedData.setXColumn(p_xColumnCombo->currentIndex());
        updatePreview();
        emit settingsChanged();
    });
    
    connect(p_yColumnCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this]() {
        // Update parsed data column settings immediately
        d_parsedData.setYColumn(p_yColumnCombo->currentIndex());
        updatePreview();
        emit settingsChanged();
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
        styleStatusLabel(p_fileStatusLabel, ThemeColors::SubtleText, true);
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
            styleStatusLabel(p_fileStatusLabel, ThemeColors::StatusInfo);
        }
    }
    
    // Compact preview in settings context
    if (p_previewTable) {
        p_previewTable->setMaximumHeight(100);
    }
}

void GenericXYOverlayWidget::createSourceFileConfigUI(QGroupBox *parent)
{
    // Fixed source-file row: path + browse, plus a single status line.
    auto mainLayout = new QVBoxLayout(parent);
    mainLayout->setSpacing(6);
    mainLayout->setContentsMargins(0, 0, 0, 0);

    auto fileRow = new QHBoxLayout();
    fileRow->setSpacing(6);

    p_filePathEdit = new QLineEdit(parent);
    p_filePathEdit->setPlaceholderText("Select a data file...");
    p_filePathEdit->setMinimumWidth(250); // Ensure adequate width
    fileRow->addWidget(p_filePathEdit, 1); // Give it stretch priority

    p_browseButton = new QPushButton("📁", parent);
    p_browseButton->setToolTip("Browse for data file");
    p_browseButton->setMaximumSize(30, 30);
    fileRow->addWidget(p_browseButton);

    mainLayout->addLayout(fileRow);

    p_fileStatusLabel = new QLabel(parent);
    p_fileStatusLabel->setWordWrap(false);
    p_fileStatusLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    p_fileStatusLabel->setMinimumHeight(20);
    p_fileStatusLabel->setMaximumHeight(22);
    p_fileStatusLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    mainLayout->addWidget(p_fileStatusLabel);
}

void GenericXYOverlayWidget::createSourceFileSettingsUI(QGroupBox *parent)
{
    auto mainLayout = new QVBoxLayout(parent);
    mainLayout->setSpacing(4);
    mainLayout->setContentsMargins(0, 0, 0, 0);

    auto table = new SettingsTable(parent);

    // --- Format Detection ---
    table->addSectionRow("Format Detection");

    p_delimiterCombo = new QComboBox(parent);
    populateDelimiterComboBox();

    p_headerLinesSpinBox = new QSpinBox(parent);
    p_headerLinesSpinBox->setRange(0, 100);
    p_headerLinesSpinBox->setValue(0);

    p_autoDetectButton = new QPushButton("Auto-Detect", parent);
    p_autoDetectButton->setIcon(ThemeColors::createThemedIcon(":/icons/cpu-chip.svg", ThemeColors::IconPrimary, this));
    p_autoDetectButton->setToolTip("Automatically detect delimiter, headers, and column structure");

    table->addSettingRow("Delimiter", p_delimiterCombo);
    table->addSettingRow("Header Lines", p_headerLinesSpinBox);
    table->addSettingRow("Detection", p_autoDetectButton);

    // --- Column Mapping ---
    table->addSectionRow("Column Mapping");

    p_xColumnCombo = new QComboBox(parent);
    p_yColumnCombo = new QComboBox(parent);

    p_parseButton = new QPushButton("Parse File", parent);
    p_parseButton->setIcon(ThemeColors::createThemedIcon(":/icons/document-magnifying-glass.svg", ThemeColors::IconPrimary, this));

    table->addSettingRow("X Column", p_xColumnCombo);
    table->addSettingRow("Y Column", p_yColumnCombo);
    table->addSettingRow("Parse", p_parseButton);

    // --- Data Filtering (collapsible section) ---
    QCheckBox *filterSectionBox = nullptr;
    int filterSection = table->addCheckableSectionRow("Data Filtering", false,
                                                      &filterSectionBox);

    p_enableFilteringCheckBox = new QCheckBox("Enable X-range filtering", parent);

    p_xMinEdit = new QLineEdit(parent);
    p_xMinEdit->setPlaceholderText("Min X");
    p_xMaxEdit = new QLineEdit(parent);
    p_xMaxEdit->setPlaceholderText("Max X");

    int enableRow = table->addSettingRow("Filtering", p_enableFilteringCheckBox);

    auto rangeCell = new QWidget(parent);
    auto rangeRow = new QHBoxLayout(rangeCell);
    rangeRow->setContentsMargins(0, 0, 0, 0);
    rangeRow->addWidget(p_xMinEdit);
    rangeRow->addWidget(new QLabel("to", rangeCell));
    rangeRow->addWidget(p_xMaxEdit);
    rangeRow->addStretch();
    int rangeRowIdx = table->addSettingRow("Range", rangeCell);

    // Collapse the filtering rows when the section box is unchecked,
    // matching the old checkable-QGroupBox behavior.
    table->bindSectionRows(filterSection, {enableRow, rangeRowIdx});

    // Data statistics (compact info display)
    p_dataStatsLabel = new QLabel(parent);
    p_dataStatsLabel->setWordWrap(true);
    styleStatusLabel(p_dataStatsLabel, ThemeColors::SubtleText);

    mainLayout->addWidget(table);
    mainLayout->addWidget(p_dataStatsLabel);
    mainLayout->addStretch();

    // Initially disable filtering controls
    p_xMinEdit->setEnabled(false);
    p_xMaxEdit->setEnabled(false);

    // Inner checkbox gates the range edits.
    connect(p_enableFilteringCheckBox, &QCheckBox::toggled, [this](bool enabled) {
        p_xMinEdit->setEnabled(enabled);
        p_xMaxEdit->setEnabled(enabled);
    });

    // Section checkbox mirrors the enable-filtering checkbox, matching
    // the old filter-group checkable behavior. bindSectionRows() above
    // already collapses the bound rows on toggle.
    connect(filterSectionBox, &QCheckBox::toggled, [this](bool enabled) {
        p_enableFilteringCheckBox->setChecked(enabled);
    });
}

void GenericXYOverlayWidget::createTypeSpecificSettingsUI(QGroupBox *parent)
{
    auto layout = new QVBoxLayout(parent);
    layout->setContentsMargins(0, 0, 0, 0);

    // Preview table — the one widget that keeps stretch 1 and grows.
    p_previewTable = new QTableWidget(parent);
    p_previewTable->setAlternatingRowColors(true);
    p_previewTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    p_previewTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    p_previewTable->verticalHeader()->setVisible(false);
    p_previewTable->setMaximumHeight(200);

    // Configure columns to stretch and fill evenly
    p_previewTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    layout->addWidget(new QLabel("Data Preview:"));
    layout->addWidget(p_previewTable, 1);
}

void GenericXYOverlayWidget::analyzeAndParseFile(bool autodetect)
{
    QString filePath = getStoredFullSourceFilePath();
    if (filePath.isEmpty() || !QFileInfo::exists(filePath)) {
        d_parsedData.clear(); // Clears data, making isValid() return false
        d_fileAnalyzed = false;
        p_fileStatusLabel->setText("No file selected or file not found");
        emit dataValidityChanged(false);
        return;
    }
    
    // Get parser from registry
    auto parser = getParser();
    if (!parser) {
        d_parsedData.clear(); // Clears data, making isValid() return false
        d_fileAnalyzed = true;
        p_fileStatusLabel->setText("No GenericXY parser available");
        emit dataValidityChanged(false);
        return;
    }
    
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
        styleStatusLabel(p_fileStatusLabel, ThemeColors::StatusSuccess);
    } else {
        d_parsedData.clear(); // Clears data, making isValid() return false
        d_fileAnalyzed = true;
        p_fileStatusLabel->setText("Failed to parse file");
        styleStatusLabel(p_fileStatusLabel, ThemeColors::StatusError);
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


bool GenericXYOverlayWidget::validateColumns() const
{
    // Validate based on parsed data state, not UI state
    if (!d_parsedData.isValid()) {
        return false;
    }
    
    int xCol = d_parsedData.xColumn();
    int yCol = d_parsedData.yColumn();
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
