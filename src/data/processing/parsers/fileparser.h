#ifndef FILEPARSER_H
#define FILEPARSER_H

#include <QString>
#include <QStringList>
#include <QVariantMap>

/// \brief Abstract interface for file-format parsers.
///
/// \sa CatalogParser, GenericXYParser, FileParserRegistry
class FileParser
{
public:
    virtual ~FileParser() = default;

    /// \brief Test whether this parser recognizes the given file.
    ///
    /// Implementations typically combine an extension check
    /// (hasMatchingExtension()) with a structural sniff of the first few
    /// lines (readFileHeader()) to keep the test cheap.
    ///
    /// \param filePath Absolute or relative path to the candidate file.
    /// \param hints Optional format-specific hints (delimiter overrides, column indices, etc.). Each subclass documents which keys it consumes.
    /// \return ``true`` when the parser is willing to handle the file.
    virtual bool canParse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const = 0;

    /// \brief Short, human-readable format identifier.
    ///
    /// Used in file-dialog filters and in user-visible error messages.
    /// Examples: ``"SPCAT"``, ``"XIAM"``, ``"GenericXY"``.
    virtual QString formatName() const = 0;

    /// \brief Glob patterns this parser accepts for file-dialog filters.
    ///
    /// Patterns include the leading wildcard and dot (e.g. ``"*.cat"``).
    /// :cpp:func:`FileParserRegistry::fileDialogFilter` joins these into
    /// a single ``QFileDialog`` filter string.
    virtual QStringList fileExtensions() const = 0;

    /// \brief One-line description of the format suitable for tooltips.
    virtual QString formatDescription() const = 0;

protected:
    /// \brief Verify that ``filePath`` exists, is a regular file, and is readable.
    /// \return ``true`` when ``QFileInfo`` reports the file is readable.
    bool isFileReadable(const QString &filePath) const;

    /// \brief Test whether the file's suffix matches one of ``extensions``.
    ///
    /// Comparison is case-insensitive. ``extensions`` entries should be of
    /// the form ``".cat"`` (with the leading dot, no glob wildcard).
    bool hasMatchingExtension(const QString &filePath, const QStringList &extensions) const;

    /// \brief Read the first ``headerLines`` lines of ``filePath`` as text.
    ///
    /// Returns an empty list if the file is unreadable. Used by
    /// ``canParse()`` implementations that need to peek at the file body
    /// without reading the entire file.
    QStringList readFileHeader(const QString &filePath, int headerLines = 10) const;
};

#endif // FILEPARSER_H
