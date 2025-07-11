#include "catalogparser.h"
#include <QFile>
#include <QFileInfo>
#include <QTextStream>

bool CatalogParser::isFileReadable(const QString &filePath) const
{
    QFileInfo fileInfo(filePath);
    return fileInfo.exists() && fileInfo.isFile() && fileInfo.isReadable();
}

bool CatalogParser::hasMatchingExtension(const QString &filePath, const QStringList &extensions) const
{
    QFileInfo fileInfo(filePath);
    QString suffix = "." + fileInfo.suffix().toLower();
    
    for (const QString &ext : extensions) {
        if (suffix == ext.toLower()) {
            return true;
        }
    }
    return false;
}

QStringList CatalogParser::readFileHeader(const QString &filePath, int headerLines) const
{
    QStringList header;
    QFile file(filePath);
    
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return header;
    }
    
    QTextStream stream(&file);
    int linesRead = 0;
    
    while (!stream.atEnd() && linesRead < headerLines) {
        header.append(stream.readLine());
        ++linesRead;
    }
    
    return header;
}