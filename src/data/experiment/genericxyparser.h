#ifndef GENERICXYPARSER_H
#define GENERICXYPARSER_H

#include "catalogparser.h"
#include <QStringList>
#include <QPointF>

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
class GenericXYParser : public CatalogParser
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
    
    // CatalogParser interface
    bool canParse(const QString &filePath) const override;
    CatalogData parse(const QString &filePath) const override;
    QString formatName() const override;
    QString formatDescription() const override;
    QStringList fileExtensions() const override;
    
    // GenericXY-specific methods
    ParseSettings autoDetectSettings(const QString &filePath) const;
    ParsePreview generatePreview(const QString &filePath, const ParseSettings &settings) const;
    ParsePreview generatePreview(const QString &filePath) const; // Uses auto-detected settings
    CatalogData parseWithSettings(const QString &filePath, const ParseSettings &settings) const;
    
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
};

#endif // GENERICXYPARSER_H