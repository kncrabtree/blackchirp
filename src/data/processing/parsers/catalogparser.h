#ifndef CATALOGPARSER_H
#define CATALOGPARSER_H

#include <QString>
#include <QStringList>
#include <QVariantMap>
#include "fileparser.h"
#include <data/experiment/catalogdata.h>

/// \brief Abstract base for spectroscopic catalog parsers.
///
/// Adds a single hook on top of :cpp:class:`FileParser` — the
/// ``parse()`` method that converts a recognized catalog file into a
/// :cpp:class:`CatalogData` value. ``CatalogData`` contains the
/// ordered list of transitions (frequency, intensity, error, lower
/// energy, quantum numbers, ...) plus source-program metadata. The
/// concrete subclasses ship with Blackchirp are :cpp:class:`SPCATParser`
/// and :cpp:class:`XIAMParser`; a new format that exposes the same
/// transition shape (frequency + intensity + quantum numbers) should
/// derive from this class so that ``CatalogOverlay`` can consume it
/// without further code changes.
///
/// \sa SPCATParser, XIAMParser, CatalogData, FileParserRegistry
class CatalogParser : public FileParser
{
public:
    virtual ~CatalogParser() = default;

    /// \brief Parse a catalog file into a :cpp:class:`CatalogData` value.
    ///
    /// Returns a default-constructed (empty) ``CatalogData`` if the file
    /// cannot be opened or contains no recognizable transitions. Callers
    /// are expected to check ``CatalogData::transitions().isEmpty()``
    /// before consuming the result.
    ///
    /// \param filePath Absolute or relative path to the catalog file.
    /// \param hints Optional format-specific overrides. Subclasses
    ///              document which keys they consume.
    /// \return Parsed catalog data; empty on failure.
    virtual CatalogData parse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const = 0;
};

#endif // CATALOGPARSER_H
