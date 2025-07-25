#include <QtTest>
#include <QTextStream>
#include <QStandardPaths>
#include <QDir>

#include "data/processing/parsers/genericxyparser.h"

class GenericXYParserTest : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void testCanParse_data();
    void testCanParse();
    void testAutoDetectSettings_data();
    void testAutoDetectSettings();
    void testRealTestFiles_data();
    void testRealTestFiles();
    void testParseWithSettings_data();
    void testParseWithSettings();
    void testGeneratePreview_data();
    void testGeneratePreview();
    void testLargeFiles();
    void testMalformedData();
    void testEmptyFiles();

private:
    QString d_testDataDir;
    GenericXYParser d_parser;
    void createTestFile(const QString &filename, const QString &content);
    QString getTestFilePath(const QString &filename);
};

void GenericXYParserTest::initTestCase()
{
    d_testDataDir = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/blackchirp_test_data/";
    QDir().mkpath(d_testDataDir);
}

void GenericXYParserTest::createTestFile(const QString &filename, const QString &content)
{
    QFile file(d_testDataDir + filename);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        out << content;
    }
}

QString GenericXYParserTest::getTestFilePath(const QString &filename)
{
    return d_testDataDir + filename;
}

void GenericXYParserTest::testCanParse_data()
{
    QTest::addColumn<QString>("filename");
    QTest::addColumn<QString>("content");
    QTest::addColumn<bool>("expectedCanParse");

    QTest::newRow("simple_tab") << "simple.txt" 
                                << "1.0\t2.0\n3.0\t4.0\n5.0\t6.0\n"
                                << true;

    QTest::newRow("simple_csv") << "simple.csv"
                                << "1.0,2.0\n3.0,4.0\n5.0,6.0\n"
                                << true;

    QTest::newRow("with_headers") << "headers.txt"
                                  << "#Comment\n#Header\n1.0\t2.0\n3.0\t4.0\n"
                                  << true;

    QTest::newRow("empty_file") << "empty.txt"
                                << ""
                                << false;

    QTest::newRow("only_text") << "text_only.txt"
                               << "This is just text\nNo numbers here\n"
                               << false;

    QTest::newRow("only_headers") << "only_headers.txt"
                                  << "#Header 1\n#Header 2\n#Header 3\n"
                                  << false;
}

void GenericXYParserTest::testCanParse()
{
    QFETCH(QString, filename);
    QFETCH(QString, content);
    QFETCH(bool, expectedCanParse);

    createTestFile(filename, content);
    
    bool canParse = d_parser.canParse(getTestFilePath(filename));
    QCOMPARE(canParse, expectedCanParse);
}

void GenericXYParserTest::testAutoDetectSettings_data()
{
    QTest::addColumn<QString>("filename");
    QTest::addColumn<QString>("content");
    QTest::addColumn<QString>("expectedDelimiter");
    QTest::addColumn<int>("expectedHeaderLines");
    QTest::addColumn<bool>("expectedHasColumnHeaders");
    QTest::addColumn<int>("expectedColumns");

    QTest::newRow("tab_with_headers") << "tab_headers.txt"
                                      << "#Comment 1\n#Comment 2\nX\tY\n1.0\t2.0\n3.0\t4.0\n"
                                      << "\t" << 2 << true << 2;

    QTest::newRow("csv_no_headers") << "csv_no_headers.csv"
                                    << "1.0,2.0\n3.0,4.0\n5.0,6.0\n"
                                    << "," << 0 << false << 2;

    QTest::newRow("space_delimited") << "space.txt"
                                     << "1.0 2.0\n3.0 4.0\n5.0 6.0\n"
                                     << " " << 0 << false << 2;

    QTest::newRow("semicolon_delimited") << "semicolon.txt"
                                         << "1.0;2.0\n3.0;4.0\n5.0;6.0\n"
                                         << ";" << 0 << false << 2;

    QTest::newRow("many_headers") << "many_headers.txt"
                                  << "#H1\n#H2\n#H3\n#H4\n#H5\nX\tY\tZ\n1.0\t2.0\t3.0\n4.0\t5.0\t6.0\n"
                                  << "\t" << 5 << true << 3;
}

void GenericXYParserTest::testAutoDetectSettings()
{
    QFETCH(QString, filename);
    QFETCH(QString, content);
    QFETCH(QString, expectedDelimiter);
    QFETCH(int, expectedHeaderLines);
    QFETCH(bool, expectedHasColumnHeaders);
    QFETCH(int, expectedColumns);

    createTestFile(filename, content);
    
    GenericXYParser::ParseSettings settings = d_parser.autoDetectSettings(getTestFilePath(filename));
    
    QCOMPARE(settings.delimiter, expectedDelimiter);
    QCOMPARE(settings.headerLines, expectedHeaderLines);
    QCOMPARE(settings.hasColumnHeaders, expectedHasColumnHeaders);
    QCOMPARE(settings.columnNames.size(), expectedColumns);
}

void GenericXYParserTest::testRealTestFiles_data()
{
    QTest::addColumn<QString>("filename");
    QTest::addColumn<QString>("expectedDelimiter");
    QTest::addColumn<int>("expectedHeaderLines");
    QTest::addColumn<int>("expectedColumns");
    QTest::addColumn<int>("expectedDataPoints");

    QTest::newRow("1011.txt") << "tests/testdata/xydata/1011.txt" << "\t" << 26 << 5 << 172;
    QTest::newRow("1012.txt") << "tests/testdata/xydata/1012.txt" << "\t" << 14 << 2 << 192;
    QTest::newRow("115.txt") << "tests/testdata/xydata/115.txt" << "\t" << 27 << 9 << 72;
    QTest::newRow("59.txt") << "tests/testdata/xydata/59.txt" << "\t" << 8 << 3 << 233;
    QTest::newRow("MT59_926.txt") << "tests/testdata/xydata/MT59_926.txt" << " " << 0 << 2 << 208;
    QTest::newRow("blanked_primos_samp_old.txt") << "tests/testdata/xydata/blanked_primos_samp_old.txt" << " " << 0 << 2 << 200;
    QTest::newRow("cubic") << "tests/testdata/xydata/cubic" << " " << 0 << 4 << 202;
    QTest::newRow("cubic.csv") << "tests/testdata/xydata/cubic.csv" << "," << 0 << 4 << 212;
    QTest::newRow("Od_230602_F795A_CSA-X.txt") << "tests/testdata/xydata/Od_230602_F795A_CSA-X.txt" << "\t" << 2 << 2 << 937;
    QTest::newRow("JMOLplot") << "tests/testdata/xydata/JMOLplot" << " " << 2 << 4 << 9;
}

void GenericXYParserTest::testRealTestFiles()
{
    QFETCH(QString, filename);
    QFETCH(QString, expectedDelimiter);
    QFETCH(int, expectedHeaderLines);
    QFETCH(int, expectedColumns);
    QFETCH(int, expectedDataPoints);

    QString fullPath = QString("/home/kncrabtree/github/blackchirp/src/") + filename;
    
    // Test canParse first
    QVERIFY2(d_parser.canParse(fullPath), 
             QString("Cannot parse file: %1").arg(fullPath).toLocal8Bit());
    
    // Test autoDetectSettings
    GenericXYParser::ParseSettings settings = d_parser.autoDetectSettings(fullPath);
    
    // Accept greedy whitespace as equivalent to single space or tab (improved detection)
    if (expectedDelimiter == " " || expectedDelimiter == "\t") {
        QVERIFY2(settings.delimiter == expectedDelimiter || settings.delimiter == "\\s+",
                 QString("Expected delimiter '%1' or '\\s+', got '%2'")
                 .arg(expectedDelimiter, settings.delimiter).toLocal8Bit());
    } else {
        QCOMPARE(settings.delimiter, expectedDelimiter);
    }
    
    QCOMPARE(settings.headerLines, expectedHeaderLines);
    QCOMPARE(settings.columnNames.size(), expectedColumns);
    
    // Test parseWithSettings to get actual data count
    GenericXYData data = d_parser.parseWithSettings(fullPath, settings);
    if (data.hasError()) {
        QFAIL(QString("Parse failed: %1").arg(data.errorMessage()).toLocal8Bit());
    }
    QCOMPARE(data.dataLines(), expectedDataPoints);
}

void GenericXYParserTest::testParseWithSettings_data()
{
    QTest::addColumn<QString>("filename");
    QTest::addColumn<QString>("content");
    QTest::addColumn<int>("xColumn");
    QTest::addColumn<int>("yColumn");
    QTest::addColumn<QVector<double>>("expectedXValues");
    QTest::addColumn<QVector<double>>("expectedYValues");

    QTest::newRow("simple_data") << "simple.txt"
                                 << "1.0\t2.0\n3.0\t4.0\n5.0\t6.0\n"
                                 << 0 << 1
                                 << QVector<double>({1.0, 3.0, 5.0})
                                 << QVector<double>({2.0, 4.0, 6.0});

    QTest::newRow("with_headers") << "headers.txt"
                                  << "#Comment\nX_Val\tY_Val\n1.5\t2.5\n3.5\t4.5\n"
                                  << 0 << 1
                                  << QVector<double>({1.5, 3.5})
                                  << QVector<double>({2.5, 4.5});

    QTest::newRow("scientific_notation") << "scientific.txt"
                                         << "1.0e-3\t2.5e2\n3.2e1\t4.7e-1\n"
                                         << 0 << 1
                                         << QVector<double>({0.001, 32.0})
                                         << QVector<double>({250.0, 0.47});
}

void GenericXYParserTest::testParseWithSettings()
{
    QFETCH(QString, filename);
    QFETCH(QString, content);
    QFETCH(int, xColumn);
    QFETCH(int, yColumn);
    QFETCH(QVector<double>, expectedXValues);
    QFETCH(QVector<double>, expectedYValues);

    createTestFile(filename, content);
    
    GenericXYParser::ParseSettings settings = d_parser.autoDetectSettings(getTestFilePath(filename));
    settings.xColumn = xColumn;
    settings.yColumn = yColumn;
    
    GenericXYData data = d_parser.parseWithSettings(getTestFilePath(filename), settings);
    if (data.hasError()) {
        QFAIL(QString("Parse failed: %1").arg(data.errorMessage()).toLocal8Bit());
    }
    
    QVector<QPointF> points = data.data();
    QCOMPARE(points.size(), expectedXValues.size());
    QCOMPARE(points.size(), expectedYValues.size());
    
    for (int i = 0; i < points.size(); ++i) {
        QCOMPARE(points[i].x(), expectedXValues[i]);
        QCOMPARE(points[i].y(), expectedYValues[i]);
    }
}

void GenericXYParserTest::testGeneratePreview_data()
{
    QTest::addColumn<QString>("filename");
    QTest::addColumn<QString>("content");
    QTest::addColumn<bool>("expectedSuccess");
    QTest::addColumn<int>("expectedPreviewPoints");

    // Use real data files for testing instead of simulated ones
    QTest::newRow("MT59_926.txt") << "tests/testdata/xydata/MT59_926.txt"
                                  << "" // Use real file
                                  << true << 100; // Should be limited to preview max

    QTest::newRow("blanked_primos_samp_old.txt") << "tests/testdata/xydata/blanked_primos_samp_old.txt"
                                                 << "" // Use real file
                                                 << true << 100;

    QTest::newRow("cubic.csv") << "tests/testdata/xydata/cubic.csv"
                               << "" // Use real file
                               << true << 100;

    QTest::newRow("tmc_kaifu.txt") << "tests/testdata/xydata/tmc_kaifu.txt"
                                   << "" // Use real file, not generated content
                                   << true << 100; // Should be limited to preview max

    QTest::newRow("invalid_data") << "invalid.txt"
                                  << "text\tmore_text\nstill\ttext\n"
                                  << false << 0;
}


void GenericXYParserTest::testGeneratePreview()
{
    QFETCH(QString, filename);
    QFETCH(QString, content);
    QFETCH(bool, expectedSuccess);
    QFETCH(int, expectedPreviewPoints);

    QString filePath;
    if (filename.startsWith("/")) {
        // Absolute path - use as is for real files
        filePath = filename;
    } else {
        // Create test file
        createTestFile(filename, content);
        filePath = getTestFilePath(filename);
    }
    
    GenericXYParser::ParsePreview preview = d_parser.generatePreview(filePath);
    
    QCOMPARE(preview.success, expectedSuccess);
    if (expectedSuccess) {
        QVERIFY(preview.previewData.size() > 0);
        QVERIFY(preview.previewData.size() <= expectedPreviewPoints);
    }
}

void GenericXYParserTest::testLargeFiles()
{
    QString largePath = "tests/testdata/xydata/tmc_kaifu.txt";
    
    QVERIFY2(d_parser.canParse(largePath), 
             QString("Cannot parse large test file: %1").arg(largePath).toLocal8Bit());
    
    GenericXYParser::ParseSettings settings = d_parser.autoDetectSettings(largePath);
    QCOMPARE(settings.delimiter, QString("\t"));
    QCOMPARE(settings.headerLines, 0);
    QCOMPARE(settings.columnNames.size(), 2);
    
    // Test preview (should be limited)
    GenericXYParser::ParsePreview preview = d_parser.generatePreview(largePath);
    QVERIFY(preview.success);
    QVERIFY(preview.previewData.size() <= 100); // Preview should be limited
    
    // Test full parse
    GenericXYData data = d_parser.parseWithSettings(largePath, settings);
    if (data.hasError()) {
        QFAIL(QString("Large file parse failed: %1").arg(data.errorMessage()).toLocal8Bit());
    }
    
    QVector<QPointF> points = data.data();
    QCOMPARE(points.size(), 10000);
    
    // Verify some specific values from the file
    QCOMPARE(points[0].x(), 8739.999);
    QCOMPARE(points[0].y(), -0.01651);
}

void GenericXYParserTest::testMalformedData()
{
    QString content = "X\tY\n"
                      "1.0\t2.0\n"
                      "invalid\t3.0\n"
                      "4.0\tnot_a_number\n"
                      "5.0\t6.0\n";
    
    createTestFile("malformed.txt", content);
    
    GenericXYParser::ParseSettings settings = d_parser.autoDetectSettings(getTestFilePath("malformed.txt"));
    
    GenericXYData data = d_parser.parseWithSettings(getTestFilePath("malformed.txt"), settings);
    if (data.hasError()) {
        QFAIL(QString("Malformed data test failed: %1").arg(data.errorMessage()).toLocal8Bit());
    }
    
    QVector<QPointF> points = data.data();
    // Should only parse valid lines (1.0,2.0) and (5.0,6.0)
    QCOMPARE(points.size(), 2);
    QCOMPARE(points[0].x(), 1.0);
    QCOMPARE(points[0].y(), 2.0);
    QCOMPARE(points[1].x(), 5.0);
    QCOMPARE(points[1].y(), 6.0);
}

void GenericXYParserTest::testEmptyFiles()
{
    createTestFile("empty.txt", "");
    QVERIFY(!d_parser.canParse(getTestFilePath("empty.txt")));
    
    createTestFile("only_headers.txt", "#Header 1\n#Header 2\n");
    QVERIFY(!d_parser.canParse(getTestFilePath("only_headers.txt")));
}


QTEST_MAIN(GenericXYParserTest)
#include "tst_genericxyparser.moc"
