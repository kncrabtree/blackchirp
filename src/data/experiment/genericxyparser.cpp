#include "genericxyparser.h"

#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QtMath>
#include <QDebug>
#include <QPointF>
#include <QFileInfo>

GenericXYParser::GenericXYParser()
{
}

bool GenericXYParser::analyzeFile(const QString &filePath, const QVariantMap &hints) const
{
    if (!isFileReadable(filePath))
        return false;
    
    // Check file modification time for cache validation
    QFileInfo fileInfo(filePath);
    QDateTime currentModified = fileInfo.lastModified();
    
    // Check if we have valid cached analysis for this file
    if (d_cachedAnalysis.isValid && 
        d_cachedAnalysis.filePath == filePath && 
        d_cachedAnalysis.lastModified == currentModified) {
        return true; // Analysis already done and cached
    }
    
    // Reset cache
    d_cachedAnalysis = FileAnalysis();
    d_cachedAnalysis.filePath = filePath;
    d_cachedAnalysis.lastModified = currentModified;
    
    // Read entire file to handle different line endings
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly))
        return false;
    
    QByteArray data = file.readAll();
    QString content = QString::fromUtf8(data);
    
    // Handle different line ending types
    if (content.contains("\r\n")) {
        d_cachedAnalysis.allLines = content.split("\r\n", Qt::KeepEmptyParts);
    } else if (content.contains("\r")) {
        d_cachedAnalysis.allLines = content.split("\r", Qt::KeepEmptyParts);
    } else {
        d_cachedAnalysis.allLines = content.split("\n", Qt::KeepEmptyParts);
    }
    
    if (d_cachedAnalysis.allLines.isEmpty())
        return false;
    
    // Get data lines from the tail (more reliable for detection)
    for (int i = d_cachedAnalysis.allLines.size() - 1; i >= 0 && d_cachedAnalysis.dataLines.size() < 50; --i) {
        QString line = d_cachedAnalysis.allLines[i].trimmed();
        
        // Stop if we hit a blank line and already have some data lines (separator between header and data)
        if (line.isEmpty()) {
            if (!d_cachedAnalysis.dataLines.isEmpty()) {
                break;
            }
            continue; // Skip blank lines if we haven't found data yet
        }
        
        // Stop if we hit a comment line and already have data lines (end of data section)
        if (isCommentLine(line)) {
            if (!d_cachedAnalysis.dataLines.isEmpty()) {
                break;
            }
            continue; // Skip comment lines if we haven't found data yet
        }
        
        // Check if line might be a single-number header by testing if entire line is just one number
        bool isSingleNumber = false;
        line.toDouble(&isSingleNumber);
        
        if (isSingleNumber) {
            continue; // Skip single-number headers (like atom counts)
        }
        
        // Collect any line that has multiple parts (could be data or column headers)
        // We'll sort out column headers later in delimiter detection
        
        d_cachedAnalysis.dataLines.prepend(line); // Keep order for consistency
    }
    if (d_cachedAnalysis.dataLines.isEmpty())
        return false;
    
    // Apply hints if provided
    if (hints.contains("delimiter")) {
        d_cachedAnalysis.settings.delimiter = hints["delimiter"].toString();
    } else {
        // Auto-detect delimiter from tail data
        d_cachedAnalysis.settings.delimiter = detectDelimiter(d_cachedAnalysis.dataLines);
    }
    
    // Calculate expected numeric columns using the determined delimiter
    calculateExpectedNumericColumns();
    
    if (d_cachedAnalysis.settings.delimiter.isEmpty())
        return false;
    
    // Improved header detection using cached expected column count
    d_cachedAnalysis.settings.headerLines = detectHeaderLinesUsingDelimiter();
    
    // Check for column headers and generate column names
    // Find the line immediately after header lines for column header detection
    QString candidateHeaderLine;
    if (d_cachedAnalysis.settings.headerLines < d_cachedAnalysis.allLines.size()) {
        candidateHeaderLine = d_cachedAnalysis.allLines[d_cachedAnalysis.settings.headerLines].trimmed();
    }
    
    if (!candidateHeaderLine.isEmpty()) {
        d_cachedAnalysis.settings.hasColumnHeaders = detectColumnHeaders(candidateHeaderLine, d_cachedAnalysis.settings.delimiter);
        
        if (d_cachedAnalysis.settings.hasColumnHeaders) {
            d_cachedAnalysis.settings.columnNames = parseColumnHeaders(candidateHeaderLine, d_cachedAnalysis.settings.delimiter);
        } else {
            // Count columns from candidate line (which is first data line)
            QStringList parts;
            if (d_cachedAnalysis.settings.delimiter == "\\s+") {
                parts = candidateHeaderLine.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
            } else {
                parts = candidateHeaderLine.split(d_cachedAnalysis.settings.delimiter, Qt::KeepEmptyParts);
            }
            d_cachedAnalysis.settings.columnNames = generateColumnNames(parts.size());
        }
    } else if (!d_cachedAnalysis.dataLines.isEmpty()) {
        // Fallback to using tail sample if we can't find the post-header line
        QString fallbackLine = d_cachedAnalysis.dataLines.first();
        QStringList parts;
        if (d_cachedAnalysis.settings.delimiter == "\\s+") {
            parts = fallbackLine.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        } else {
            parts = fallbackLine.split(d_cachedAnalysis.settings.delimiter, Qt::KeepEmptyParts);
        }
        d_cachedAnalysis.settings.columnNames = generateColumnNames(parts.size());
        d_cachedAnalysis.settings.hasColumnHeaders = false;
    }
    
    // Smart column assignments - find first two numeric columns
    d_cachedAnalysis.settings.xColumn = 0;
    d_cachedAnalysis.settings.yColumn = 1;
    
    // If we have sample data, find the first two numeric columns
    if (!d_cachedAnalysis.dataLines.isEmpty()) {
        QString sampleLine = d_cachedAnalysis.dataLines.first();
        QStringList parts;
        if (d_cachedAnalysis.settings.delimiter == "\\s+") {
            parts = sampleLine.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        } else {
            parts = sampleLine.split(d_cachedAnalysis.settings.delimiter, Qt::KeepEmptyParts);
        }
        
        QList<int> numericColumns;
        for (int i = 0; i < parts.size(); ++i) {
            QString trimmed = parts[i].trimmed();
            if (!trimmed.isEmpty()) {
                bool ok;
                trimmed.toDouble(&ok);
                if (ok) {
                    numericColumns.append(i);
                    if (numericColumns.size() >= 2) break; // Found first two numeric columns
                }
            }
        }
        
        // Use first two numeric columns if found
        if (numericColumns.size() >= 2) {
            d_cachedAnalysis.settings.xColumn = numericColumns[0];
            d_cachedAnalysis.settings.yColumn = numericColumns[1];
        } else if (numericColumns.size() == 1) {
            // Only one numeric column found, use it as Y and hope X is convertible
            d_cachedAnalysis.settings.yColumn = numericColumns[0];
        }
        // else: fallback to default 0,1
    }
    
    // Validate by counting valid numerical data points
    int validDataPoints = 0;
    int minRequired = 2;
    
    for (const QString &line : d_cachedAnalysis.dataLines) {
        QStringList parts;
        
        // Handle greedy whitespace delimiter specially
        if (d_cachedAnalysis.settings.delimiter == "\\s+") {
            parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        } else {
            parts = line.split(d_cachedAnalysis.settings.delimiter, Qt::KeepEmptyParts);
        }
        
        if (parts.size() >= 2) {
            // Count how many columns can be parsed as numbers
            int numericCols = 0;
            for (const QString &part : parts) {
                QString trimmed = part.trimmed();
                if (!trimmed.isEmpty()) {  // Don't treat empty strings as numbers
                    bool ok;
                    trimmed.toDouble(&ok);
                    if (ok) numericCols++;
                }
            }
            
            // Require at least 2 numeric columns (for X,Y data)
            if (numericCols >= 2) {
                validDataPoints++;
                if (validDataPoints >= minRequired) {
                    d_cachedAnalysis.isValid = true;
                    return true; // Found enough valid data points
                }
            }
        }
    }
    
    return false;
}

bool GenericXYParser::canParse(const QString &filePath, const QVariantMap &hints) const
{
    return analyzeFile(filePath, hints);
}

GenericXYData GenericXYParser::parse(const QString &filePath, const QVariantMap &hints) const
{
    if (!analyzeFile(filePath, hints)) {
        GenericXYData errorData;
        errorData.setErrorMessage(QString("Cannot parse file: %1").arg(filePath));
        return errorData;
    }
    
    return parseWithSettings(filePath, d_cachedAnalysis.settings);
}

QString GenericXYParser::formatName() const
{
    return "GenericXY";
}

QString GenericXYParser::formatDescription() const
{
    return "Generic XY data files (CSV, TSV, space-delimited)";
}

QStringList GenericXYParser::fileExtensions() const
{
    return {".csv", ".tsv", ".txt", ".dat", ".data", ".xy", ".tab"};
}

GenericXYParser::ParseSettings GenericXYParser::autoDetectSettings(const QString &filePath) const
{
    if (analyzeFile(filePath)) {
        return d_cachedAnalysis.settings;
    }
    return ParseSettings(); // Return default settings if analysis fails
}

GenericXYParser::ParsePreview GenericXYParser::generatePreview(const QString &filePath, const ParseSettings &settings) const
{
    ParsePreview preview;
    
    // Use provided settings if delimiter is specified, otherwise auto-detect
    if (settings.delimiter.isEmpty()) {
        if (!analyzeFile(filePath)) {
            preview.errorMessage = "Cannot analyze file: " + filePath;
            return preview;
        }
        preview.detectedSettings = d_cachedAnalysis.settings;
    } else {
        preview.detectedSettings = settings;
    }
    
    if (!isFileReadable(filePath)) {
        preview.errorMessage = "Cannot read file: " + filePath;
        return preview;
    }
    
    // Use cached file data if available
    QStringList allFileLines;
    if (d_cachedAnalysis.isValid && d_cachedAnalysis.filePath == filePath) {
        allFileLines = d_cachedAnalysis.allLines;
    } else {
        // Read file for preview
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            preview.errorMessage = "Cannot open file: " + filePath;
            return preview;
        }
        
        QByteArray data = file.readAll();
        QString content = QString::fromUtf8(data);
        
        // Handle different line ending types
        if (content.contains("\r\n")) {
            allFileLines = content.split("\r\n", Qt::KeepEmptyParts);
        } else if (content.contains("\r")) {
            allFileLines = content.split("\r", Qt::KeepEmptyParts);
        } else {
            allFileLines = content.split("\n", Qt::KeepEmptyParts);
        }
    }
    
    // Take first 50 lines for preview
    const int maxPreviewLines = 50;
    QStringList allLines = allFileLines.mid(0, qMin(maxPreviewLines, allFileLines.size()));
    int lineCount = allFileLines.size();
    
    preview.sampleLines = allLines;
    
    // Parse preview data
    QVector<QPointF> previewData;
    const int maxPreviewPoints = 100;
    
    // Calculate start line: skip headers and column headers
    int startLine = preview.detectedSettings.headerLines;
    if (preview.detectedSettings.hasColumnHeaders) {
        startLine++; // Skip the column header line too
    }
    
    for (int i = startLine; i < allLines.size() && previewData.size() < maxPreviewPoints; ++i) {
        QString line = allLines[i].trimmed();
        if (line.isEmpty())
            continue;
        
        QPointF point = parseDataLine(line, preview.detectedSettings.delimiter,
                                     preview.detectedSettings.xColumn,
                                     preview.detectedSettings.yColumn);
        if (!point.isNull()) {
            previewData.append(point);
        }
    }
    
    preview.previewData = previewData;
    preview.totalDataLines = lineCount - preview.detectedSettings.headerLines;
    if (preview.detectedSettings.hasColumnHeaders) {
        preview.totalDataLines--;
    }
    
    preview.success = !previewData.isEmpty();
    if (!preview.success && preview.errorMessage.isEmpty()) {
        preview.errorMessage = "No valid numerical data found in file";
    }
    
    return preview;
}

GenericXYParser::ParsePreview GenericXYParser::generatePreview(const QString &filePath) const
{
    ParseSettings defaultSettings;
    defaultSettings.delimiter = ""; // Force auto-detection
    return generatePreview(filePath, defaultSettings);
}

GenericXYData GenericXYParser::parseWithSettings(const QString &filePath, const ParseSettings &settings) const
{
    GenericXYData data;
    
    if (!isFileReadable(filePath)) {
        data.setErrorMessage(QString("Cannot read file: %1").arg(filePath));
        return data;
    }
    
    // Use cached lines if available and up-to-date
    QStringList allLines;
    QFileInfo fileInfo(filePath);
    QDateTime currentModified = fileInfo.lastModified();
    
    if (d_cachedAnalysis.isValid && 
        d_cachedAnalysis.filePath == filePath && 
        d_cachedAnalysis.lastModified == currentModified) {
        // Use cached file contents
        allLines = d_cachedAnalysis.allLines;
    } else {
        // Read file contents
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            data.setErrorMessage(QString("Cannot open file: %1").arg(filePath));
            return data;
        }
        
        QByteArray fileData = file.readAll();
        QString content = QString::fromUtf8(fileData);
        
        // Handle different line ending types
        if (content.contains("\r\n")) {
            allLines = content.split("\r\n", Qt::KeepEmptyParts);
        } else if (content.contains("\r")) {
            allLines = content.split("\r", Qt::KeepEmptyParts);
        } else {
            allLines = content.split("\n", Qt::KeepEmptyParts);
        }
    }
    
    // Calculate start line: skip headers and column headers
    int startLine = settings.headerLines;
    if (settings.hasColumnHeaders) {
        startLine++; // Skip the column header line too
    }
    
    // Parse data lines
    for (int i = startLine; i < allLines.size(); ++i) {
        QString line = allLines[i].trimmed();
        if (line.isEmpty())
            continue;
            
        QPointF point = parseDataLine(line, settings.delimiter, settings.xColumn, settings.yColumn);
        if (!point.isNull()) {
            data.addDataPoint(point);
        }
    }
    
    if (data.isEmpty()) {
        data.setErrorMessage("No valid data points found in file");
        return data;
    }
    
    // Set metadata
    data.setFileName(fileInfo.fileName());
    data.setFilePath(filePath);
    data.setDelimiter(settings.delimiter);
    data.setHeaderLines(settings.headerLines);
    data.setHasColumnHeaders(settings.hasColumnHeaders);
    data.setColumnNames(settings.columnNames);
    data.setXColumn(settings.xColumn);
    data.setYColumn(settings.yColumn);
    
    return data;
}

QString GenericXYParser::detectDelimiter(const QStringList &lines) const
{
    QStringList candidates = {",", "\t", " ", ";", "\\s+"};  // Added greedy whitespace
    QMap<QString, int> scores;
    
    // Get lines for analysis, skipping first line which might be column headers
    QStringList analysisLines;
    int startIdx = (lines.size() > 1) ? 1 : 0; // Skip first line if we have more than one
    
    for (int i = startIdx; i < lines.size(); ++i) {
        QString line = lines[i].trimmed();
        if (!line.isEmpty() && !isCommentLine(line)) {
            analysisLines.append(line);
        }
    }
    
    // If we don't have enough lines after skipping first, include all lines
    if (analysisLines.size() < 2) {
        analysisLines.clear();
        for (const QString &line : lines) {
            if (!line.trimmed().isEmpty() && !isCommentLine(line)) {
                analysisLines.append(line);
            }
        }
    }
    
    for (const QString &delimiter : candidates) {
        int totalColumns = 0;
        int validLines = 0;
        int numericColumns = 0;
        
        for (const QString &line : analysisLines) {
            QStringList parts;
            
            // Handle greedy whitespace delimiter specially
            if (delimiter == "\\s+") {
                parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
            } else {
                parts = line.split(delimiter, Qt::KeepEmptyParts);
            }
            
            if (parts.size() >= 2) {
                totalColumns += parts.size();
                validLines++;
                
                // Count how many columns can be parsed as numbers
                for (const QString &part : parts) {
                    QString trimmed = part.trimmed();
                    if (!trimmed.isEmpty()) {  // Don't treat empty strings as numbers
                        bool ok;
                        trimmed.toDouble(&ok);
                        if (ok) numericColumns++;
                    }
                }
            }
        }
        
        if (validLines > 0) {
            // Prefer delimiters that give highest percentage of numeric columns (quality over quantity)
            double avgColumns = double(totalColumns) / validLines;
            double numericRatio = double(numericColumns) / totalColumns;
            
            // Weight numeric percentage heavily to prefer quality parsing
            int score = int(numericRatio * 1000) + validLines * 10 + int(avgColumns);
            scores[delimiter] = score;
        }
    }
    
    if (scores.isEmpty())
        return ","; // Default fallback
    
    // Return delimiter with highest score
    auto maxIt = std::max_element(scores.begin(), scores.end());
    return maxIt.key();
}

void GenericXYParser::calculateExpectedNumericColumns() const
{
    d_cachedAnalysis.expectedNumericColumns = 0;
    d_cachedAnalysis.expectedTotalColumns = 0;
    
    if (d_cachedAnalysis.dataLines.isEmpty())
        return;
    
    QString testLine = d_cachedAnalysis.dataLines.first();
    QStringList parts;
    if (d_cachedAnalysis.settings.delimiter == "\\s+") {
        parts = testLine.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
    } else {
        parts = testLine.split(d_cachedAnalysis.settings.delimiter, Qt::KeepEmptyParts);
    }
    d_cachedAnalysis.expectedTotalColumns = parts.size();
    
    for (const QString &part : parts) {
        QString trimmed = part.trimmed();
        if (!trimmed.isEmpty()) {  // Don't treat empty strings as numbers
            bool ok;
            trimmed.toDouble(&ok);
            if (ok) d_cachedAnalysis.expectedNumericColumns++;
        }
    }
    
    if (d_cachedAnalysis.expectedNumericColumns < 2)
        d_cachedAnalysis.expectedNumericColumns = 2; // Minimum for XY data
}

int GenericXYParser::detectHeaderLinesUsingDelimiter() const
{
    if (d_cachedAnalysis.allLines.isEmpty())
        return 0;
    
    // Use cached expected columns (already calculated)
    int expectedNumericColumns = d_cachedAnalysis.expectedNumericColumns;
    int expectedTotalColumns = d_cachedAnalysis.expectedTotalColumns;
    
    if (expectedNumericColumns < 2)
        expectedNumericColumns = 2; // Fallback minimum
    if (expectedTotalColumns < 2)
        expectedTotalColumns = expectedNumericColumns; // Fallback to numeric count
    
    
    // Iterate from beginning, counting headers until we find lines with expected numeric columns
    int headerCount = 0;
    bool prevLineMaybeColHeaders = false;
    
    for (const QString &line : d_cachedAnalysis.allLines) {
        QString trimmed = line.trimmed();
        
        // Count blank lines as headers
        if (trimmed.isEmpty()) {
            headerCount++;
            continue;
        }
        
        // Count comment lines as headers
        if (isCommentLine(line)) {
            headerCount++;
            continue;
        }
        
        // Split line to check column structure
        QStringList parts;
        if (d_cachedAnalysis.settings.delimiter == "\\s+") {
            parts = trimmed.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        } else {
            parts = trimmed.split(d_cachedAnalysis.settings.delimiter, Qt::KeepEmptyParts);
        }
        
        // If line has only 1 part, it can't be XY data, so treat as header
        if (parts.size() < 2) {
            headerCount++;
            continue;
        }
        
        // Check if this line has the expected number of numeric columns
        int numericCols = 0;
        for (const QString &part : parts) {
            QString trimmedPart = part.trimmed();
            if (!trimmedPart.isEmpty()) {  // Don't treat empty strings as numbers
                bool ok;
                trimmedPart.toDouble(&ok);
                if (ok) numericCols++;
            }
        }
        
        // If we found a line with expected numeric columns, it's data
        if (numericCols >= expectedNumericColumns) {
            // If previous line was maybe column headers, don't count it as header
            if (prevLineMaybeColHeaders) {
                headerCount--;
            }
            break;
        }
        
        // Check if this might be column headers (right total number of columns, but non-numeric)
        if (parts.size() == expectedTotalColumns && numericCols == 0) {
            headerCount++; // Tentatively count as header
            prevLineMaybeColHeaders = true;
        } else {
            // Line has multiple parts but wrong count or some numeric - treat as header
            headerCount++;
            prevLineMaybeColHeaders = false;
        }
    }
    return headerCount;
}

int GenericXYParser::detectHeaderLines(const QStringList &lines) const
{
    int headerCount = 0;
    
    for (const QString &line : lines) {
        QString trimmed = line.trimmed();
        
        // Count blank lines as potential headers
        if (trimmed.isEmpty()) {
            headerCount++;
            continue;
        }
            
        if (isCommentLine(line)) {
            headerCount++;
        } else {
            // Check if line looks like a single-value header (e.g., atom count, line count)
            QStringList parts = trimmed.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
            if (parts.size() == 1) {
                bool isNumber;
                parts[0].toDouble(&isNumber);
                if (isNumber) {
                    headerCount++; // Single number likely a header/count
                    continue;
                }
            }
            
            break; // First multi-column data line found
        }
    }
    
    return headerCount;
}

bool GenericXYParser::detectColumnHeaders(const QString &line, const QString &delimiter) const
{
    if (line.trimmed().isEmpty())
        return false;
    
    QStringList parts;
    if (delimiter == "\\s+") {
        parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
    } else {
        parts = line.split(delimiter, Qt::KeepEmptyParts);
    }
    
    if (parts.size() < 2)
        return false;
    
    // Check if first few columns look like text headers rather than numbers
    int textColumns = 0;
    int numericColumns = 0;
    
    for (int i = 0; i < qMin(4, parts.size()); ++i) {
        QString part = parts[i].trimmed();
        if (part.isEmpty())
            continue;
            
        bool isNumber;
        part.toDouble(&isNumber);
        
        if (isNumber) {
            numericColumns++;
        } else {
            // Check if it looks like a text header (any alphabetic characters)
            if (part.contains(QRegularExpression("[a-zA-Z]"))) {
                textColumns++;
            }
        }
    }
    
    // Consider it headers if ANY text columns found (more lenient)
    return textColumns > 0;
}

QStringList GenericXYParser::generateColumnNames(int numColumns) const
{
    QStringList names;
    for (int i = 0; i < numColumns; ++i) {
        names.append(QString("Col%1").arg(i + 1));
    }
    return names;
}

QStringList GenericXYParser::parseColumnHeaders(const QString &line, const QString &delimiter) const
{
    QStringList parts;
    if (delimiter == "\\s+") {
        parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
    } else {
        parts = line.split(delimiter, Qt::KeepEmptyParts);
    }
    
    QStringList cleanNames;
    
    for (int i = 0; i < parts.size(); ++i) {
        QString name = cleanSemicolons(parts[i].trimmed());
        if (name.isEmpty()) {
            name = QString("Col%1").arg(i + 1);
        }
        cleanNames.append(name);
    }
    
    return cleanNames;
}

QPointF GenericXYParser::parseDataLine(const QString &line, const QString &delimiter, int xCol, int yCol) const
{
    if (line.trimmed().isEmpty() || isCommentLine(line))
        return QPointF();
    
    QStringList parts;
    if (delimiter == "\\s+") {
        parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
    } else {
        parts = line.split(delimiter, Qt::KeepEmptyParts);
    }
    
    if (xCol >= parts.size() || yCol >= parts.size())
        return QPointF();
    
    bool xOk, yOk;
    double x = parts[xCol].trimmed().toDouble(&xOk);
    double y = parts[yCol].trimmed().toDouble(&yOk);
    
    if (xOk && yOk) {
        return QPointF(x, y);
    }
    
    return QPointF();
}

bool GenericXYParser::isCommentLine(const QString &line) const
{
    QString trimmed = line.trimmed();
    return trimmed.startsWith('#') || trimmed.startsWith('!') || trimmed.startsWith('%');
}

QString GenericXYParser::cleanSemicolons(const QString &input) const
{
    QString cleaned = input;
    cleaned.replace(';', '_'); // Replace semicolons with underscores for BC CSV compatibility
    return cleaned;
}

QStringList GenericXYParser::readSampleLines(const QString &filePath, int maxLines) const
{
    QStringList lines;
    
    if (!isFileReadable(filePath))
        return lines;
    
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return lines;
    
    // Read entire file to handle both CR and LF line endings
    QByteArray data = file.readAll();
    QString content = QString::fromUtf8(data);
    
    // Handle different line ending types
    if (content.contains("\r\n")) {
        // Windows CRLF
        lines = content.split("\r\n", Qt::KeepEmptyParts);
    } else if (content.contains("\r")) {
        // Mac CR (like Od_230602 file)
        lines = content.split("\r", Qt::KeepEmptyParts);
    } else {
        // Unix LF
        lines = content.split("\n", Qt::KeepEmptyParts);
    }
    
    // Limit to requested number of lines
    if (lines.size() > maxLines) {
        lines = lines.mid(0, maxLines);
    }
    
    return lines;
}