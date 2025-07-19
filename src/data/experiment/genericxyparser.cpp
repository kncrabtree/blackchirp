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
    
    // Check file extension first
    if (!hasMatchingExtension(filePath, fileExtensions()))
        return false;
    
    // Try to read sample lines and detect format
    QStringList sampleLines = readSampleLines(filePath, 10);
    if (sampleLines.isEmpty())
        return false;
    
    // Skip comment lines
    QStringList dataLines;
    for (const QString &line : sampleLines) {
        if (!isCommentLine(line) && !line.trimmed().isEmpty()) {
            dataLines.append(line);
        }
    }
    
    if (dataLines.isEmpty())
        return false;
    
    // Try to detect delimiter and parse at least one valid data point
    QString delimiter = detectDelimiter(dataLines);
    if (delimiter.isEmpty())
        return false;
    
    // Try to parse first data line
    for (const QString &line : dataLines) {
        QStringList parts = line.split(delimiter, Qt::KeepEmptyParts);
        if (parts.size() >= 2) {
            // Try to parse as numbers
            bool xOk, yOk;
            parts[0].toDouble(&xOk);
            parts[1].toDouble(&yOk);
            if (xOk && yOk) {
                return true; // Found at least one valid data point
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
    
    QStringList sampleLines = readSampleLines(filePath, 30);
    if (sampleLines.isEmpty())
        return settings;
    
    // Detect header lines
    settings.headerLines = detectHeaderLines(sampleLines);
    
    // Get data lines (skip headers)
    QStringList dataLines;
    for (int i = settings.headerLines; i < sampleLines.size(); ++i) {
        if (!sampleLines[i].trimmed().isEmpty()) {
            dataLines.append(sampleLines[i]);
        }
    }
    
    if (dataLines.isEmpty())
        return settings;
    
    // Detect delimiter
    settings.delimiter = detectDelimiter(dataLines);
    
    // Check for column headers
    if (!dataLines.isEmpty()) {
        settings.hasColumnHeaders = detectColumnHeaders(dataLines.first(), settings.delimiter);
        
        if (settings.hasColumnHeaders) {
            settings.columnNames = parseColumnHeaders(dataLines.first(), settings.delimiter);
        } else {
            // Count columns from first data line
            QStringList parts = dataLines.first().split(settings.delimiter, Qt::KeepEmptyParts);
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
    
    QTextStream stream(&file);
    QStringList allLines;
    int lineCount = 0;
    const int maxPreviewLines = 50;
    
    // Read file and collect sample lines
    while (!stream.atEnd() && lineCount < 200) {
        QString line = stream.readLine();
        if (lineCount < maxPreviewLines) {
            allLines.append(line);
        }
        lineCount++;
    }
    
    preview.sampleLines = allLines;
    
    // Parse preview data
    QVector<QPointF> previewData;
    int dataLineCount = 0;
    const int maxPreviewPoints = 100;
    
    for (int i = preview.detectedSettings.headerLines; i < allLines.size() && dataLineCount < maxPreviewPoints; ++i) {
        QString line = allLines[i].trimmed();
        if (line.isEmpty())
            continue;
            
        // Skip column header line if present
        if (dataLineCount == 0 && preview.detectedSettings.hasColumnHeaders) {
            dataLineCount++;
            continue;
        }
        
        QPointF point = parseDataLine(line, preview.detectedSettings.delimiter,
                                     preview.detectedSettings.xColumn,
                                     preview.detectedSettings.yColumn);
        if (!point.isNull()) {
            previewData.append(point);
        }
        dataLineCount++;
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
    QStringList candidates = {",", "\t", " ", ";"};
    QMap<QString, int> scores;
    
    for (const QString &delimiter : candidates) {
        int totalColumns = 0;
        int validLines = 0;
        
        for (const QString &line : lines) {
            if (line.trimmed().isEmpty() || isCommentLine(line))
                continue;
                
            QStringList parts = line.split(delimiter, Qt::KeepEmptyParts);
            if (parts.size() >= 2) {
                totalColumns += parts.size();
                validLines++;
            }
        }
        
        if (validLines > 0) {
            // Prefer delimiters that give consistent column counts
            double avgColumns = double(totalColumns) / validLines;
            scores[delimiter] = validLines * 100 + int(avgColumns * 10);
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
        if (line.trimmed().isEmpty())
            continue;
            
        if (isCommentLine(line)) {
            headerCount++;
        } else {
            break; // First non-comment line found
        }
    }
    
    return headerCount;
}

bool GenericXYParser::detectColumnHeaders(const QString &line, const QString &delimiter) const
{
    if (line.trimmed().isEmpty())
        return false;
    
    QStringList parts = line.split(delimiter, Qt::KeepEmptyParts);
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
            // Check if it looks like a text header
            if (part.length() > 1 && part.contains(QRegularExpression("[a-zA-Z]"))) {
                textColumns++;
            }
        }
    }
    
    // Consider it headers if more text than numbers
    return textColumns > numericColumns;
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
    QStringList parts = line.split(delimiter, Qt::KeepEmptyParts);
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
    
    QStringList parts = line.split(delimiter, Qt::KeepEmptyParts);
    
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
    
    QTextStream stream(&file);
    int lineCount = 0;
    
    while (!stream.atEnd() && lineCount < maxLines) {
        lines.append(stream.readLine());
        lineCount++;
    }
    
    return lines;
}