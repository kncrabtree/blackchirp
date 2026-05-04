#include "fileparser.h"
#include <QFile>
#include <QFileInfo>
#include <QTextStream>

bool FileParser::isFileReadable(const QString &filePath) const
{
    QFileInfo info(filePath);
    return info.exists() && info.isFile() && info.isReadable();
}

bool FileParser::hasMatchingExtension(const QString &filePath, const QStringList &extensions) const
{
    QFileInfo info(filePath);
    QString suffix = "." + info.suffix().toLower();
    
    for (const QString &ext : extensions) {
        if (suffix == ext.toLower()) {
            return true;
        }
    }
    return false;
}

QStringList FileParser::readFileHeader(const QString &filePath, int headerLines) const
{
    QStringList lines;
    
    if (!isFileReadable(filePath))
        return lines;
    
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return lines;
    
    QTextStream stream(&file);
    int count = 0;
    while (!stream.atEnd() && count < headerLines) {
        lines.append(stream.readLine());
        count++;
    }
    
    return lines;
}