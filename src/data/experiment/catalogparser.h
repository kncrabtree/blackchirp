#ifndef CATALOGPARSER_H
#define CATALOGPARSER_H

#include <QString>
#include <QStringList>
#include <memory>
#include "catalogdata.h"

/**
 * @brief Abstract base class for spectroscopic catalog parsers
 * 
 * This class defines the interface for parsing different catalog formats.
 * Subclasses implement format-specific parsing logic for programs like
 * SPCAT, XIAM, BELGI, ASROT, etc.
 */
class CatalogParser
{
public:
    virtual ~CatalogParser() = default;
    
    /**
     * @brief Check if this parser can handle the given file
     * @param filePath Path to the catalog file to check
     * @return true if this parser can handle the file format
     */
    virtual bool canParse(const QString &filePath) const = 0;
    
    /**
     * @brief Parse a catalog file and extract transition data
     * @param filePath Path to the catalog file to parse
     * @return CatalogData containing parsed transitions and metadata
     * @throws std::runtime_error if parsing fails
     */
    virtual CatalogData parse(const QString &filePath) const = 0;
    
    /**
     * @brief Get the human-readable name of this catalog format
     * @return Format name (e.g., "SPCAT", "XIAM")
     */
    virtual QString formatName() const = 0;
    
    /**
     * @brief Get typical file extensions for this format
     * @return List of file extensions (e.g., [".cat", ".out"])
     */
    virtual QStringList fileExtensions() const = 0;
    
    /**
     * @brief Get a description of this catalog format
     * @return Human-readable description
     */
    virtual QString formatDescription() const = 0;
    
protected:
    /**
     * @brief Helper method to check if file exists and is readable
     * @param filePath Path to check
     * @return true if file can be read
     */
    bool isFileReadable(const QString &filePath) const;
    
    /**
     * @brief Helper method to detect format by file extension
     * @param filePath Path to check
     * @param extensions List of supported extensions
     * @return true if extension matches
     */
    bool hasMatchingExtension(const QString &filePath, const QStringList &extensions) const;
    
    /**
     * @brief Helper method to read file header for format detection
     * @param filePath Path to file
     * @param headerLines Number of lines to read (default: 10)
     * @return Header lines as string list
     */
    QStringList readFileHeader(const QString &filePath, int headerLines = 10) const;
};

#endif // CATALOGPARSER_H