#include "catalogparserregistry.h"
#include <QDebug>

CatalogParserRegistry* CatalogParserRegistry::s_instance = nullptr;

CatalogParserRegistry::CatalogParserRegistry(QObject *parent)
    : QObject(parent)
{
}

CatalogParserRegistry::~CatalogParserRegistry() = default;

CatalogParserRegistry* CatalogParserRegistry::instance()
{
    if (!s_instance) {
        s_instance = new CatalogParserRegistry();
    }
    return s_instance;
}

void CatalogParserRegistry::cleanup()
{
    delete s_instance;
    s_instance = nullptr;
}

void CatalogParserRegistry::registerParser(std::unique_ptr<CatalogParser> parser)
{
    if (!parser) {
        qWarning() << "Attempted to register null parser";
        return;
    }
    
    QString formatName = parser->formatName();
    
    d_parsers.push_back(std::move(parser));
    emit parserRegistered(formatName);
}

CatalogParser* CatalogParserRegistry::findParser(const QString &filePath) const
{
    for (const auto &parser : d_parsers) {
        if (parser->canParse(filePath)) {
            return parser.get();
        }
    }
    return nullptr;
}

std::vector<CatalogParser*> CatalogParserRegistry::getAllParsers() const
{
    std::vector<CatalogParser*> parsers;
    parsers.reserve(d_parsers.size());
    
    for (const auto &parser : d_parsers) {
        parsers.push_back(parser.get());
    }
    
    return parsers;
}

QStringList CatalogParserRegistry::supportedFormats() const
{
    QStringList formats;
    
    for (const auto &parser : d_parsers) {
        formats.append(parser->formatName());
    }
    
    return formats;
}

QStringList CatalogParserRegistry::supportedExtensions() const
{
    QStringList extensions;
    
    for (const auto &parser : d_parsers) {
        extensions.append(parser->fileExtensions());
    }
    
    // Remove duplicates
    extensions.removeDuplicates();
    return extensions;
}

QString CatalogParserRegistry::fileDialogFilter() const
{
    if (d_parsers.empty()) {
        return "All Files (*)";
    }
    
    QStringList filters;
    
    // Add format-specific filters
    for (const auto &parser : d_parsers) {
        QStringList exts = parser->fileExtensions();
        if (!exts.isEmpty()) {
            QString extPattern = exts.join(" ");
            filters.append(QString("%1 Files (%2)").arg(parser->formatName(), extPattern));
        }
    }
    
    // Add "All supported" filter
    QStringList allExts = supportedExtensions();
    if (!allExts.isEmpty()) {
        filters.prepend(QString("All Catalog Files (%1)").arg(allExts.join(" ")));
    }
    
    // Add generic "All Files" filter
    filters.append("All Files (*)");
    
    return filters.join(";;");
}

bool CatalogParserRegistry::canParseFile(const QString &filePath) const
{
    return findParser(filePath) != nullptr;
}
