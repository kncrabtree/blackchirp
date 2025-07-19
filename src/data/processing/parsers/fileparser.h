#ifndef FILEPARSER_H
#define FILEPARSER_H

#include <QString>
#include <QStringList>
#include <QVariantMap>

/**
 * @brief Abstract base interface for file parsers
 * 
 * This interface defines the common functionality for parsing different file formats.
 * It provides a flexible foundation that can be extended for specific use cases like
 * spectroscopic catalogs (CatalogParser) or generic data files (GenericXYParser).
 */
class FileParser
{
public:
    virtual ~FileParser() = default;
    
    /**
     * @brief Check if this parser can handle the given file
     * @param filePath Path to the file to check
     * @param hints Optional parsing hints/settings (default: empty)
     * @return true if this parser can handle the file format
     */
    virtual bool canParse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const = 0;
    
    /**
     * @brief Get the human-readable name of this file format
     * @return Format name (e.g., "SPCAT", "GenericXY")
     */
    virtual QString formatName() const = 0;
    
    /**
     * @brief Get typical file extensions for this format
     * @return List of file extensions (e.g., [".cat", ".csv"])
     */
    virtual QStringList fileExtensions() const = 0;
    
    /**
     * @brief Get a description of this file format
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

#endif // FILEPARSER_H