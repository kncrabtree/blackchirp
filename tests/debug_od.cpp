#include "../src/data/experiment/genericxyparser.h"
#include <QDebug>
#include <QCoreApplication>

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    
    GenericXYParser parser;
    QString filePath = "/home/kncrabtree/github/blackchirp/src/tests/testdata/xydata/Od_230602_F795A_CSA-X.txt";
    
    qDebug() << "=== Testing Od_230602 file ===";
    qDebug() << "canParse:" << parser.canParse(filePath);
    
    auto settings = parser.autoDetectSettings(filePath);
    qDebug() << "Delimiter:" << settings.delimiter;
    qDebug() << "Header lines:" << settings.headerLines;
    qDebug() << "Has column headers:" << settings.hasColumnHeaders;
    qDebug() << "Column names:" << settings.columnNames;
    
    return 0;
}