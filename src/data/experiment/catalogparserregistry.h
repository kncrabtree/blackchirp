#ifndef CATALOGPARSERREGISTRY_H
#define CATALOGPARSERREGISTRY_H

#include <QObject>
#include <QStringList>
#include <memory>
#include <vector>
#include "catalogparser.h"

/**
 * @brief Registry for managing catalog parser instances
 * 
 * This singleton class manages all available catalog parsers and provides
 * functionality to auto-detect file formats and find appropriate parsers.
 */
class CatalogParserRegistry : public QObject
{
    Q_OBJECT
    
public:
    /**
     * @brief Get the singleton instance
     * @return Pointer to the registry instance
     */
    static CatalogParserRegistry* instance();
    
    /**
     * @brief Clean up the singleton instance (call during application shutdown)
     */
    static void cleanup();
    
    /**
     * @brief Register a new catalog parser
     * @param parser Unique pointer to the parser (registry takes ownership)
     */
    void registerParser(std::unique_ptr<CatalogParser> parser);
    
    /**
     * @brief Find a parser that can handle the given file
     * @param filePath Path to the catalog file
     * @return Pointer to parser that can handle the file, or nullptr if none found
     */
    CatalogParser* findParser(const QString &filePath) const;
    
    /**
     * @brief Get all registered parsers
     * @return Vector of parser pointers (registry retains ownership)
     */
    std::vector<CatalogParser*> getAllParsers() const;
    
    /**
     * @brief Get list of supported format names
     * @return List of format names (e.g., ["SPCAT", "XIAM"])
     */
    QStringList supportedFormats() const;
    
    /**
     * @brief Get list of all supported file extensions
     * @return List of file extensions (e.g., [".cat", ".out"])
     */
    QStringList supportedExtensions() const;
    
    /**
     * @brief Get file filter string for file dialogs
     * @return Filter string suitable for QFileDialog
     */
    QString fileDialogFilter() const;
    
    /**
     * @brief Check if any parser can handle the given file
     * @param filePath Path to check
     * @return true if a compatible parser is found
     */
    bool canParseFile(const QString &filePath) const;

signals:
    /**
     * @brief Emitted when a new parser is registered
     * @param formatName Name of the newly registered format
     */
    void parserRegistered(const QString &formatName);

private:
    explicit CatalogParserRegistry(QObject *parent = nullptr);
    ~CatalogParserRegistry();
    
    // Disable copy/move
    CatalogParserRegistry(const CatalogParserRegistry&) = delete;
    CatalogParserRegistry& operator=(const CatalogParserRegistry&) = delete;
    
    static CatalogParserRegistry* s_instance;
    std::vector<std::unique_ptr<CatalogParser>> d_parsers;
};

#endif // CATALOGPARSERREGISTRY_H