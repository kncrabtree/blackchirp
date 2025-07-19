#ifndef GENERICXYPARSER_H
#define GENERICXYPARSER_H

#include "fileparser.h"
#include "genericxydata.h"
#include <QStringList>
#include <QPointF>
#include <QDateTime>

/**
 * @brief Parser for generic XY data files (CSV, TSV, space-delimited)
 * 
 * This parser handles various text-based XY data formats with:
 * - Auto-detection of delimiters (comma, tab, space, semicolon)
 * - Configurable header line skipping (with comment character detection)
 * - Column mapping for X and Y data selection
 * - Robust parsing with invalid data skipping
 * - Preview capabilities for UI integration
 */
class GenericXYParser : public FileParser
{
public:
    struct ParseSettings {
        QString delimiter = ",";
        int headerLines = 0;
        int xColumn = 0;
        int yColumn = 1;
        QStringList columnNames;
        bool hasColumnHeaders = false;
    };
    
    struct ParsePreview {
        QStringList sampleLines;
        ParseSettings detectedSettings;
        QVector<QPointF> previewData;
        int totalDataLines = 0;
        QString errorMessage;
        bool success = false;
    };
    
    GenericXYParser();
    
    // FileParser interface
    bool canParse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const override;
    QString formatName() const override;
    QString formatDescription() const override;
    QStringList fileExtensions() const override;
    
    // GenericXY-specific methods
    GenericXYData parse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const;
    ParseSettings autoDetectSettings(const QString &filePath) const;
    ParsePreview generatePreview(const QString &filePath, const ParseSettings &settings) const;
    ParsePreview generatePreview(const QString &filePath) const; // Uses auto-detected settings
    GenericXYData parseWithSettings(const QString &filePath, const ParseSettings &settings) const;
    
    // Public test access methods
    QString detectDelimiterPublic(const QStringList &lines) const { return detectDelimiter(lines); }
    int detectHeaderLinesPublic(const QStringList &lines) const { return detectHeaderLines(lines); }
    bool detectColumnHeadersPublic(const QString &line, const QString &delimiter) const { return detectColumnHeaders(line, delimiter); }
    QStringList readSampleLinesPublic(const QString &filePath, int maxLines = 20) const { return readSampleLines(filePath, maxLines); }

private:
    /**
     * @brief Detect the most likely delimiter character
     * @param lines Sample lines from the file
     * @return Detected delimiter character
     */
    QString detectDelimiter(const QStringList &lines) const;
    
    /**
     * @brief Count header lines to skip (comment lines with #, !, %)
     * @param lines Sample lines from the file
     * @return Number of header lines to skip
     */
    int detectHeaderLines(const QStringList &lines) const;
    
    /**
     * @brief Detect if first data line contains column headers
     * @param line First data line after header
     * @param delimiter Detected delimiter
     * @return true if line appears to contain text headers
     */
    bool detectColumnHeaders(const QString &line, const QString &delimiter) const;
    
    /**
     * @brief Generate automatic column names
     * @param numColumns Number of columns detected
     * @return List of auto-generated column names (Col1, Col2, etc.)
     */
    QStringList generateColumnNames(int numColumns) const;
    
    /**
     * @brief Parse column headers from a line
     * @param line Header line
     * @param delimiter Delimiter character
     * @return List of cleaned column names
     */
    QStringList parseColumnHeaders(const QString &line, const QString &delimiter) const;
    
    /**
     * @brief Parse a data line into numerical values
     * @param line Data line
     * @param delimiter Delimiter character
     * @param xCol X column index
     * @param yCol Y column index
     * @return QPointF with X,Y data, or invalid point if parsing fails
     */
    QPointF parseDataLine(const QString &line, const QString &delimiter, 
                          int xCol, int yCol) const;
    
    /**
     * @brief Check if a line is a comment line
     * @param line Line to check
     * @return true if line starts with comment characters (#, !, %)
     */
    bool isCommentLine(const QString &line) const;
    
    /**
     * @brief Clean semicolons from strings (BC CSV compatibility)
     * @param input Input string
     * @return String with semicolons removed/replaced
     */
    QString cleanSemicolons(const QString &input) const;
    
    /**
     * @brief Read sample lines from file for analysis
     * @param filePath Path to file
     * @param maxLines Maximum lines to read (default: 20)
     * @return Sample lines for format detection
     */
    QStringList readSampleLines(const QString &filePath, int maxLines = 20) const;
    
    /**
     * @brief Calculate expected numeric columns using current delimiter and data lines
     * Updates d_cachedAnalysis.expectedNumericColumns
     */
    void calculateExpectedNumericColumns() const;
    
    /**
     * @brief Detect header lines using cached analysis data
     * @return Number of header lines
     */
    int detectHeaderLinesUsingDelimiter() const;
    
    // Analysis caching for performance and consistency
    struct FileAnalysis {
        QString filePath;
        QDateTime lastModified;
        QStringList allLines;
        QStringList dataLines;
        ParseSettings settings;
        int expectedNumericColumns = 0;
        int expectedTotalColumns = 0;
        bool isValid = false;
    };
    
    /**
     * @brief Perform comprehensive file analysis with caching
     * @param filePath Path to file to analyze
     * @param hints Optional parsing hints from interface
     * @return true if file can be parsed, false otherwise
     */
    bool analyzeFile(const QString &filePath, const QVariantMap &hints = QVariantMap()) const;
    
    // Mutable cache for analysis results
    mutable FileAnalysis d_cachedAnalysis;
};

#endif // GENERICXYPARSER_H