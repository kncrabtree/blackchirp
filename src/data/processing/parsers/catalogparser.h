#ifndef CATALOGPARSER_H
#define CATALOGPARSER_H

#include <QString>
#include <QStringList>
#include <QVariantMap>
#include <memory>
#include "fileparser.h"
#include <data/experiment/catalogdata.h>

/**
 * @brief Abstract base class for spectroscopic catalog parsers
 * 
 * This class defines the interface for parsing different catalog formats.
 * Subclasses implement format-specific parsing logic for programs like
 * SPCAT, XIAM, BELGI, ASROT, etc.
 */
class CatalogParser : public FileParser
{
public:
    virtual ~CatalogParser() = default;
    
    /**
     * @brief Parse a catalog file and extract transition data
     * @param filePath Path to the catalog file to parse
     * @param hints Optional parsing hints/settings (default: empty)
     * @return CatalogData containing parsed transitions and metadata
     */
    virtual CatalogData parse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const = 0;
};

#endif // CATALOGPARSER_H