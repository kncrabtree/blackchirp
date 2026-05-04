#ifndef CATALOGPARSERREGISTRY_H
#define CATALOGPARSERREGISTRY_H

#include <QObject>
#include <QStringList>
#include <memory>
#include <vector>
#include "fileparser.h"

/**
 * @brief Registry for managing file parser instances
 * 
 * This singleton class manages all available file parsers (catalog parsers,
 * generic XY parsers, etc.) and provides functionality to auto-detect file
 * formats and find appropriate parsers.
 */
class FileParserRegistry : public QObject
{
    Q_OBJECT
    
public:
    /**
     * @brief Get the singleton instance
     * @return Pointer to the registry instance
     */
    static FileParserRegistry* instance();
    
    /**
     * @brief Clean up the singleton instance (call during application shutdown)
     */
    static void cleanup();
    
    /**
     * @brief Register a new file parser
     * @param parser Unique pointer to the parser (registry takes ownership)
     */
    void registerParser(std::unique_ptr<FileParser> parser);
    
    /**
     * @brief Find a parser that can handle the given file
     * @param filePath Path to the file
     * @return Pointer to parser that can handle the file, or nullptr if none found
     */
    FileParser* findParser(const QString &filePath) const;
    
    /**
     * @brief Find a parser of specific type that can handle the given file
     * @param filePath Path to the file
     * @return Pointer to parser of specified type, or nullptr if none found
     */
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
    
    /**
     * @brief Get all registered parsers
     * @return Vector of parser pointers (registry retains ownership)
     */
    std::vector<FileParser*> getAllParsers() const;
    
    /**
     * @brief Get list of supported format names
     * @return List of format names (e.g., ["SPCAT", "XIAM", "GenericXY"])
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
    explicit FileParserRegistry(QObject *parent = nullptr);
    ~FileParserRegistry();
    
    // Disable copy/move
    FileParserRegistry(const FileParserRegistry&) = delete;
    FileParserRegistry& operator=(const FileParserRegistry&) = delete;
    
    static FileParserRegistry* s_instance;
    std::vector<std::unique_ptr<FileParser>> d_parsers;
};

#endif // FILEPARSERREGISTRY_H
