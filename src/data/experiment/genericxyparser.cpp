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

bool GenericXYParser::canParse(const QString &filePath) const
{
    if (!isFileReadable(filePath))
        return false;
    
    // Read entire file to handle different line endings
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;
    
    QByteArray data = file.readAll();
    QString content = QString::fromUtf8(data);
    
    // Handle different line ending types
    QStringList allLines;
    if (content.contains("\r\n")) {
        allLines = content.split("\r\n", Qt::KeepEmptyParts);
    } else if (content.contains("\r")) {
        allLines = content.split("\r", Qt::KeepEmptyParts);
    } else {
        allLines = content.split("\n", Qt::KeepEmptyParts);
    }
    
    if (allLines.isEmpty())
        return false;
    
    // Get data lines from the tail (more reliable for detection)
    QStringList dataLines;
    for (int i = allLines.size() - 1; i >= 0 && dataLines.size() < 50; --i) {
        QString line = allLines[i].trimmed();
        if (!line.isEmpty() && !isCommentLine(line)) {
            dataLines.prepend(line); // Keep order for consistency
        }
    }
    
    if (dataLines.isEmpty())
        return false;
    
    // Auto-detect delimiter from tail data
    QString testDelimiter = detectDelimiter(dataLines);
    if (testDelimiter.isEmpty())
        return false;
    
    // Count valid numerical data points (need at least 2 for basic validation)
    int validDataPoints = 0;
    int minRequired = 2;
    
    for (const QString &line : dataLines) {
        QStringList parts;
        
        // Handle greedy whitespace delimiter specially
        if (testDelimiter == "\\s+") {
            parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        } else {
            parts = line.split(testDelimiter, Qt::KeepEmptyParts);
        }
        
        if (parts.size() >= 2) {
            // Try to parse at least first two columns as numbers
            bool xOk, yOk;
            parts[0].trimmed().toDouble(&xOk);
            parts[1].trimmed().toDouble(&yOk);
            if (xOk && yOk) {
                validDataPoints++;
                if (validDataPoints >= minRequired) {
                    return true; // Found enough valid data points
                }
            }
        }
    }
    
    return false;
}

CatalogData GenericXYParser::parse(const QString &filePath) const
{
    ParseSettings settings = autoDetectSettings(filePath);
    return parseWithSettings(filePath, settings);
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
    ParseSettings settings;
    
    // Use same robust file reading as canParse method
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return settings;
    
    QByteArray data = file.readAll();
    QString content = QString::fromUtf8(data);
    
    // Handle different line ending types
    QStringList allLines;
    if (content.contains("\r\n")) {
        allLines = content.split("\r\n", Qt::KeepEmptyParts);
    } else if (content.contains("\r")) {
        allLines = content.split("\r", Qt::KeepEmptyParts);
    } else {
        allLines = content.split("\n", Qt::KeepEmptyParts);
    }
    
    if (allLines.isEmpty())
        return settings;
    
    // Use first portion for header detection
    QStringList headerSampleLines = allLines.mid(0, qMin(50, allLines.size()));
    settings.headerLines = detectHeaderLines(headerSampleLines);
    
    // Get data lines from the tail (more reliable, same as canParse)
    QStringList dataLines;
    for (int i = allLines.size() - 1; i >= 0 && dataLines.size() < 50; --i) {
        QString line = allLines[i].trimmed();
        if (!line.isEmpty() && !isCommentLine(line)) {
            dataLines.prepend(line); // Keep order for consistency
        }
    }
    
    if (dataLines.isEmpty())
        return settings;
    
    // Detect delimiter
    settings.delimiter = detectDelimiter(dataLines);
    
    // Check for column headers in first non-header line
    if (!dataLines.isEmpty()) {
        QString firstDataLine = dataLines.first();
        settings.hasColumnHeaders = detectColumnHeaders(firstDataLine, settings.delimiter);
        
        if (settings.hasColumnHeaders) {
            settings.columnNames = parseColumnHeaders(firstDataLine, settings.delimiter);
        } else {
            // Count columns from first data line
            QStringList parts;
            if (settings.delimiter == "\\s+") {
                parts = firstDataLine.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
            } else {
                parts = firstDataLine.split(settings.delimiter, Qt::KeepEmptyParts);
            }
            settings.columnNames = generateColumnNames(parts.size());
        }
    }
    
    // Default to first two columns
    settings.xColumn = 0;
    settings.yColumn = qMin(1, settings.columnNames.size() - 1);
    
    return settings;
}

GenericXYParser::ParsePreview GenericXYParser::generatePreview(const QString &filePath, const ParseSettings &settings) const
{
    ParsePreview preview;
    
    // Use provided settings if delimiter is specified, otherwise auto-detect
    if (settings.delimiter.isEmpty()) {
        preview.detectedSettings = autoDetectSettings(filePath);
    } else {
        preview.detectedSettings = settings;
    }
    
    if (!isFileReadable(filePath)) {
        preview.errorMessage = "Cannot read file: " + filePath;
        return preview;
    }
    
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        preview.errorMessage = "Cannot open file: " + filePath;
        return preview;
    }
    
    // Read entire file to handle different line endings (like canParse method)
    QByteArray data = file.readAll();
    QString content = QString::fromUtf8(data);
    
    // Handle different line ending types
    QStringList allFileLines;
    if (content.contains("\r\n")) {
        allFileLines = content.split("\r\n", Qt::KeepEmptyParts);
    } else if (content.contains("\r")) {
        allFileLines = content.split("\r", Qt::KeepEmptyParts);
    } else {
        allFileLines = content.split("\n", Qt::KeepEmptyParts);
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

CatalogData GenericXYParser::parseWithSettings(const QString &filePath, const ParseSettings &settings) const
{
    CatalogData data;
    
    if (!isFileReadable(filePath)) {
        throw std::runtime_error("Cannot read file: " + filePath.toStdString());
    }
    
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        throw std::runtime_error("Cannot open file: " + filePath.toStdString());
    }
    
    QTextStream stream(&file);
    QVector<TransitionData> transitions;
    
    // Skip header lines
    for (int i = 0; i < settings.headerLines && !stream.atEnd(); ++i) {
        stream.readLine();
    }
    
    // Skip column header line if present
    if (settings.hasColumnHeaders && !stream.atEnd()) {
        stream.readLine();
    }
    
    // Parse data lines
    while (!stream.atEnd()) {
        QString line = stream.readLine().trimmed();
        if (line.isEmpty())
            continue;
            
        QPointF point = parseDataLine(line, settings.delimiter, settings.xColumn, settings.yColumn);
        if (!point.isNull()) {
            TransitionData trans;
            trans.frequency = point.x();
            trans.intensity = point.y();
            trans.quantumNumbers = QString("X=%1 Y=%2").arg(point.x()).arg(point.y());
            transitions.append(trans);
        }
    }
    
    if (transitions.isEmpty()) {
        throw std::runtime_error("No valid data points found in file");
    }
    
    data.setTransitions(transitions);
    data.setSourceProgram("GenericXY");
    
    // Try to extract meaningful name from filename
    QFileInfo fileInfo(filePath);
    QString baseName = fileInfo.baseName();
    if (!baseName.isEmpty()) {
        data.setMoleculeName(baseName);
    } else {
        data.setMoleculeName("GenericXY Data");
    }
    
    return data;
}

QString GenericXYParser::detectDelimiter(const QStringList &lines) const
{
    QStringList candidates = {",", "\t", " ", ";", "\\s+"};  // Added greedy whitespace
    QMap<QString, int> scores;
    
    // Get last 20 lines for tail-first analysis (more reliable)
    QStringList analysisLines;
    int startIdx = qMax(0, lines.size() - 20);
    for (int i = startIdx; i < lines.size(); ++i) {
        QString line = lines[i].trimmed();
        if (!line.isEmpty() && !isCommentLine(line)) {
            analysisLines.append(line);
        }
    }
    
    // Fall back to all lines if tail doesn't have enough data
    if (analysisLines.size() < 3) {
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
                    bool ok;
                    part.trimmed().toDouble(&ok);
                    if (ok) numericColumns++;
                }
            }
        }
        
        if (validLines > 0) {
            // Prefer delimiters that give consistent column counts AND more numeric data
            double avgColumns = double(totalColumns) / validLines;
            double numericRatio = double(numericColumns) / totalColumns;
            scores[delimiter] = validLines * 100 + int(avgColumns * 10) + int(numericRatio * 50);
        }
    }
    
    if (scores.isEmpty())
        return ","; // Default fallback
    
    // Return delimiter with highest score
    auto maxIt = std::max_element(scores.begin(), scores.end());
    return maxIt.key();
}

int GenericXYParser::detectHeaderLines(const QStringList &lines) const
{
    int headerCount = 0;
    
    for (const QString &line : lines) {
        // Count blank lines as potential headers
        if (line.trimmed().isEmpty()) {
            headerCount++;
            continue;
        }
            
        if (isCommentLine(line)) {
            headerCount++;
        } else {
            break; // First non-comment, non-blank line found
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