#ifndef GENERICXYPARSER_H
#define GENERICXYPARSER_H

#include "fileparser.h"
#include "genericxydata.h"
#include <QStringList>
#include <QPointF>
#include <QDateTime>

/// \brief Parser for generic two-column XY text files.
class GenericXYParser : public FileParser
{
public:
    /// \brief Resolved parse parameters for a single file.
    ///
    /// The defaults match a comma-delimited file with no header rows and
    /// X/Y in the first two columns. ``autoDetectSettings()`` populates
    /// these fields from the file contents; the overlay dialog can then
    /// edit them and pass the result back to ``parseWithSettings()``.
    struct ParseSettings {
        QString delimiter = ",";       ///< Delimiter string used between fields.
        int headerLines = 0;           ///< Number of leading comment/header lines to skip.
        int xColumn = 0;               ///< Zero-based index of the X column.
        int yColumn = 1;               ///< Zero-based index of the Y column.
        QStringList columnNames;       ///< Resolved column names (auto-generated when no header row is present).
        bool hasColumnHeaders = false; ///< ``true`` when the first data line is a text header row.
    };

    /// \brief Result of a parse-preview pass.
    ///
    /// Combines the auto-detected settings, a small sample of raw lines,
    /// a short slice of parsed points, the total data-line count, and a
    /// success flag with optional error message. Used by
    /// ``GenericXYOverlayWidget`` to show the user what the importer
    /// will produce before they commit to the full parse.
    struct ParsePreview {
        QStringList sampleLines;        ///< First lines of the file, in source order.
        ParseSettings detectedSettings; ///< Settings produced by auto-detection.
        QVector<QPointF> previewData;   ///< Up to a handful of parsed points for display.
        int totalDataLines = 0;         ///< Total number of data lines in the file.
        QString errorMessage;           ///< Detection error, when ``success`` is ``false``.
        bool success = false;           ///< ``true`` when auto-detection produced a usable result.
    };

    GenericXYParser();

    /// \brief Recognize a file by suffix and a structural sniff of the
    /// first lines.
    bool canParse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const override;

    /// \brief Returns ``"GenericXY"``.
    QString formatName() const override;

    /// \brief Returns a one-line description of the format.
    QString formatDescription() const override;

    /// \brief Returns the list of recognized suffixes.
    QStringList fileExtensions() const override;

    /// \brief Parse a recognized file using auto-detected settings.
    ///
    /// Equivalent to calling :cpp:func:`parseWithSettings` with the
    /// result of :cpp:func:`autoDetectSettings`.
    GenericXYData parse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const;

    /// \brief Sniff ``filePath`` and return the inferred parse settings.
    ParseSettings autoDetectSettings(const QString &filePath) const;

    /// \brief Generate a preview using the supplied settings.
    ParsePreview generatePreview(const QString &filePath, const ParseSettings &settings) const;

    /// \brief Generate a preview using auto-detected settings.
    ParsePreview generatePreview(const QString &filePath) const;

    /// \brief Parse a recognized file with explicit settings.
    ///
    /// Overrides any auto-detection and uses the supplied
    /// :cpp:struct:`ParseSettings` verbatim.
    GenericXYData parseWithSettings(const QString &filePath, const ParseSettings &settings) const;

    /// \brief Test-only delimiter-detection accessor.
    QString detectDelimiterPublic(const QStringList &lines) const { return detectDelimiter(lines); }

    /// \brief Test-only header-line-count accessor.
    int detectHeaderLinesPublic(const QStringList &lines) const { return detectHeaderLines(lines); }

    /// \brief Test-only column-header detection accessor.
    bool detectColumnHeadersPublic(const QString &line, const QString &delimiter) const { return detectColumnHeaders(line, delimiter); }

    /// \brief Test-only sample-line reader.
    QStringList readSampleLinesPublic(const QString &filePath, int maxLines = 20) const { return readSampleLines(filePath, maxLines); }

private:
    /// \brief Score the candidate delimiters and return the winner.
    QString detectDelimiter(const QStringList &lines) const;

    /// \brief Count consecutive comment/header lines at the top of the file.
    int detectHeaderLines(const QStringList &lines) const;

    /// \brief Decide whether ``line`` looks like a textual column-header row.
    bool detectColumnHeaders(const QString &line, const QString &delimiter) const;

    /// \brief Generate placeholder column names ``Col1``, ``Col2``, ...
    QStringList generateColumnNames(int numColumns) const;

    /// \brief Split a header row into cleaned column names.
    QStringList parseColumnHeaders(const QString &line, const QString &delimiter) const;

    /// \brief Parse one data line into an ``(x, y)`` point.
    /// \return Valid ``QPointF`` on success; an invalid point on failure.
    QPointF parseDataLine(const QString &line, const QString &delimiter,
                          int xCol, int yCol) const;

    /// \brief Test whether ``line`` begins with one of the recognized
    /// comment characters (``#``, ``!``, ``%``).
    bool isCommentLine(const QString &line) const;

    /// \brief Strip embedded semicolons so the result can round-trip
    /// safely through Blackchirp's CSV storage.
    QString cleanSemicolons(const QString &input) const;

    /// \brief Read the first ``maxLines`` lines of ``filePath``.
    QStringList readSampleLines(const QString &filePath, int maxLines = 20) const;

    /// \brief Recompute and cache the expected numeric-column count.
    void calculateExpectedNumericColumns() const;

    /// \brief Header-line count derived from the cached delimiter.
    int detectHeaderLinesUsingDelimiter() const;

    /// \brief Cached file analysis used by ``analyzeFile()`` so the
    /// preview, ``canParse``, and full-parse paths share one detection
    /// result per ``(filePath, lastModified)`` pair.
    struct FileAnalysis {
        QString filePath;            ///< File the cached result describes.
        QDateTime lastModified;      ///< Mtime captured when the result was produced.
        QStringList allLines;        ///< Full file contents.
        QStringList dataLines;       ///< Subset of ``allLines`` past the header.
        ParseSettings settings;      ///< Auto-detected settings.
        int expectedNumericColumns = 0; ///< Number of columns that parse as numbers in the data section.
        int expectedTotalColumns = 0;   ///< Total column count, including non-numeric columns.
        bool isValid = false;        ///< ``true`` when the cached analysis is current and usable.
    };

    /// \brief Run a full file analysis, populating ``d_cachedAnalysis``.
    /// \return ``true`` when the file is recognized as a parseable XY file.
    bool analyzeFile(const QString &filePath, const QVariantMap &hints = QVariantMap()) const;

    mutable FileAnalysis d_cachedAnalysis; ///< Cached analysis result; invalidated when the file's mtime changes.
};

#endif // GENERICXYPARSER_H
