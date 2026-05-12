#ifndef CATALOGPARSERREGISTRY_H
#define CATALOGPARSERREGISTRY_H

#include <QObject>
#include <QStringList>
#include <memory>
#include <vector>
#include "fileparser.h"

/// \brief Singleton catalog of registered :cpp:class:`FileParser` instances.
class FileParserRegistry : public QObject
{
    Q_OBJECT

public:
    /// \brief Return the process-wide singleton, constructing it on
    /// first call.
    static FileParserRegistry* instance();

    /// \brief Destroy the singleton.
    ///
    /// Called from application shutdown so all owned parsers are torn
    /// down before Qt's global cleanup runs.
    static void cleanup();

    /// \brief Take ownership of ``parser`` and add it to the registry.
    ///
    /// Emits :cpp:func:`parserRegistered` with the new parser's format
    /// name. A null pointer is rejected with a ``qWarning`` and no
    /// signal is emitted.
    void registerParser(std::unique_ptr<FileParser> parser);

    /// \brief Find the first registered parser whose ``canParse`` returns ``true`` for ``filePath``.
    /// \return Borrowed pointer (registry retains ownership), or ``nullptr`` if no parser claims the file.
    FileParser* findParser(const QString &filePath) const;

    /// \brief Find the first parser of type ``T`` (or a subclass) that
    /// claims ``filePath``.
    ///
    /// Used by callers that need a specific parser family — e.g.,
    /// ``CatalogOverlayWidget`` only wants a :cpp:class:`CatalogParser`,
    /// even if a generic parser would also accept the file.
    ///
    /// \tparam T Parser class to filter on.
    /// \return Borrowed ``T*`` or ``nullptr``.
    template<typename T>
    T* findParserOfType(const QString &filePath) const {
        for (const auto &parser : d_parsers) {
            if (auto specificParser = dynamic_cast<T*>(parser.get())) {
                if (specificParser->canParse(filePath)) {
                    return specificParser;
                }
            }
        }
        return nullptr;
    }

    /// \brief Return borrowed pointers to every registered parser, in
    /// registration order.
    std::vector<FileParser*> getAllParsers() const;

    /// \brief Return the format names of every registered parser.
    QStringList supportedFormats() const;

    /// \brief Return the union of every parser's
    /// :cpp:func:`FileParser::fileExtensions` with duplicates removed.
    QStringList supportedExtensions() const;

    /// \brief Build a ``QFileDialog`` filter string covering every
    /// registered parser.
    ///
    /// The filter string includes one entry per format, an "All Catalog
    /// Files" entry combining every supported extension, and a final
    /// "All Files" wildcard.
    QString fileDialogFilter() const;

    /// \brief Convenience predicate equivalent to
    /// ``findParser(filePath) != nullptr``.
    bool canParseFile(const QString &filePath) const;

signals:
    /// \brief Emitted from :cpp:func:`registerParser` after the new
    /// parser is added.
    /// \param formatName The :cpp:func:`FileParser::formatName` of the
    ///                   registered parser.
    void parserRegistered(const QString &formatName);

private:
    explicit FileParserRegistry(QObject *parent = nullptr);
    ~FileParserRegistry();

    // Disable copy/move
    FileParserRegistry(const FileParserRegistry&) = delete;
    FileParserRegistry& operator=(const FileParserRegistry&) = delete;

    static FileParserRegistry* s_instance;
    std::vector<std::unique_ptr<FileParser>> d_parsers;
};

#endif // FILEPARSERREGISTRY_H
