#include <QtTest>
#include <QTemporaryFile>
#include <QDir>

#include <src/data/experiment/spcatparser.h>
#include <src/data/experiment/catalogparserregistry.h>

class SPCATParserTest : public QObject
{
    Q_OBJECT
public:
    SPCATParserTest() {};
    ~SPCATParserTest() {};

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testCanParse();
    void testParseC047527Sample();
    void testParseC052509Sample();
    void testParseC056519Sample();
    void testParserRegistration();
    void testInvalidFile();
    void testEmptyFile();

private:
    QString getTestDataPath(const QString &filename) const;
    void verifyTransitionData(const TransitionData &trans, 
                            double expectedFreq, 
                            double expectedIntensity, 
                            const QString &expectedQN) const;
    
    SPCATParser *m_parser;
    QString m_testDataDir;
};

void SPCATParserTest::initTestCase()
{
    m_parser = new SPCATParser();
    
    // Get test data directory path - look for src directory
    QDir currentDir = QDir::current();
    
    // If we're in a build directory, go up and find src
    if (currentDir.dirName().startsWith("build-")) {
        currentDir.cdUp();
    }
    
    // Look for src directory
    if (currentDir.exists("src")) {
        m_testDataDir = currentDir.absoluteFilePath("src/tests/testdata");
    } else {
        // Fallback: look for tests directory in current or parent directories
        QDir searchDir = currentDir;
        while (!searchDir.exists("tests") && searchDir.cdUp()) {
            // Keep searching upward
        }
        m_testDataDir = searchDir.absoluteFilePath("tests/testdata");
    }
    
    // Debug output to help troubleshoot
    qDebug() << "Current directory:" << QDir::current().absolutePath();
    qDebug() << "Test data directory:" << m_testDataDir;
    
    // Verify test data exists
    QVERIFY(QDir(m_testDataDir).exists());
    QVERIFY(QFile(getTestDataPath("c047527_sample.cat")).exists());
    QVERIFY(QFile(getTestDataPath("c052509_sample.cat")).exists());
    QVERIFY(QFile(getTestDataPath("c056519_sample.cat")).exists());
}

void SPCATParserTest::cleanupTestCase()
{
    delete m_parser;
}

QString SPCATParserTest::getTestDataPath(const QString &filename) const
{
    return QDir(m_testDataDir).absoluteFilePath(filename);
}

void SPCATParserTest::verifyTransitionData(const TransitionData &trans, 
                                         double expectedFreq, 
                                         double expectedIntensity, 
                                         const QString &expectedQN) const
{
    QCOMPARE(trans.frequency, expectedFreq);
    QVERIFY(qAbs(trans.intensity - expectedIntensity) < 0.01); // Allow small precision differences
    QCOMPARE(trans.quantumNumbers.trimmed(), expectedQN.trimmed());
}

void SPCATParserTest::testCanParse()
{
    // Test valid SPCAT files
    QVERIFY(m_parser->canParse(getTestDataPath("c047527_sample.cat")));
    QVERIFY(m_parser->canParse(getTestDataPath("c052509_sample.cat")));
    QVERIFY(m_parser->canParse(getTestDataPath("c056519_sample.cat")));
    
    // Test invalid extensions
    QTemporaryFile invalidFile("test_XXXXXX.txt");
    QVERIFY(invalidFile.open());
    invalidFile.write("1308.6818  0.0110 -8.6114 3   39.0692 19  47527 303");
    invalidFile.close();
    QVERIFY(!m_parser->canParse(invalidFile.fileName()));
}

void SPCATParserTest::testParseC047527Sample()
{
    QString filePath = getTestDataPath("c047527_sample.cat");
    CatalogData catalogData = m_parser->parse(filePath);
    
    // Verify basic properties
    QVERIFY(!catalogData.isEmpty());
    QCOMPARE(catalogData.size(), 10);
    QCOMPARE(catalogData.sourceProgram(), QString("SPCAT"));
    QCOMPARE(catalogData.moleculeName(), QString("c047527_sample"));
    
    // Verify specific transitions (based on actual data from c047527.cat)
    // Line 1: 1308.6818  0.0110 -8.6114 3   39.0692 19  47527 303 9 1 8       8 2 7
    TransitionData trans1 = catalogData.at(0);
    QCOMPARE(trans1.frequency, 1308.6818);
    QCOMPARE(trans1.quantumNumbers, QString("9 1 8 - 8 2 7"));
    QVERIFY(trans1.additionalData.contains("frequencyError"));
    QCOMPARE(trans1.additionalData.value("frequencyError").toDouble(), 0.0110);
    QCOMPARE(trans1.additionalData.value("degeneracy").toInt(), 3);
    QCOMPARE(trans1.additionalData.value("lowerStateEnergy").toDouble(), 39.0692);
    QCOMPARE(trans1.additionalData.value("upperStateDegeneracy").toInt(), 19);
    QCOMPARE(trans1.additionalData.value("speciesTag").toString(), QString("47527"));
    QCOMPARE(trans1.additionalData.value("formatCode").toInt(), 303);
    
    // Line 4: 1507.9177  0.0002 -8.4738 3    3.4659  3  47527 303 1 1 0       1 1 1
    TransitionData trans4 = catalogData.at(3);
    QCOMPARE(trans4.frequency, 1507.9177);
    QCOMPARE(trans4.quantumNumbers, QString("1 1 0 - 1 1 1"));
    QCOMPARE(trans4.additionalData.value("frequencyError").toDouble(), 0.0002);
    QCOMPARE(trans4.additionalData.value("lowerStateEnergy").toDouble(), 3.4659);
}

void SPCATParserTest::testParseC052509Sample()
{
    QString filePath = getTestDataPath("c052509_sample.cat");
    CatalogData catalogData = m_parser->parse(filePath);
    
    // Verify basic properties
    QVERIFY(!catalogData.isEmpty());
    QCOMPARE(catalogData.size(), 10);
    QCOMPARE(catalogData.sourceProgram(), QString("SPCAT"));
    QCOMPARE(catalogData.moleculeName(), QString("c052509_sample"));
    
    // Verify specific transitions (based on actual data from c052509.cat)
    // Line 1: 8816.9010  0.0200 -5.2944 2    0.0000  3 -52509 101 1           0
    TransitionData trans1 = catalogData.at(0);
    QCOMPARE(trans1.frequency, 8816.9010);
    QCOMPARE(trans1.quantumNumbers, QString("1 - 0"));
    QCOMPARE(trans1.additionalData.value("frequencyError").toDouble(), 0.0200);
    QCOMPARE(trans1.additionalData.value("degeneracy").toInt(), 2);
    QCOMPARE(trans1.additionalData.value("lowerStateEnergy").toDouble(), 0.0000);
    QCOMPARE(trans1.additionalData.value("upperStateDegeneracy").toInt(), 3);
    QCOMPARE(trans1.additionalData.value("speciesTag").toString(), QString("-52509"));
    QCOMPARE(trans1.additionalData.value("formatCode").toInt(), 101);
    
    // Line 5: 44084.1622  0.0013 -3.2048 2    2.9410 11  52509 101 5           4
    TransitionData trans5 = catalogData.at(4);
    QCOMPARE(trans5.frequency, 44084.1622);
    QCOMPARE(trans5.quantumNumbers, QString("5 - 4"));
    QCOMPARE(trans5.additionalData.value("frequencyError").toDouble(), 0.0013);
    QCOMPARE(trans5.additionalData.value("lowerStateEnergy").toDouble(), 2.9410);
    QCOMPARE(trans5.additionalData.value("upperStateDegeneracy").toInt(), 11);
    QCOMPARE(trans5.additionalData.value("speciesTag").toString(), QString("52509")); // Note: positive tag
}

void SPCATParserTest::testParseC056519Sample()
{
    QString filePath = getTestDataPath("c056519_sample.cat");
    CatalogData catalogData = m_parser->parse(filePath);
    
    // Verify basic properties
    QVERIFY(!catalogData.isEmpty());
    QCOMPARE(catalogData.sourceProgram(), QString("SPCAT"));
    QCOMPARE(catalogData.moleculeName(), QString("c056519_sample"));
    
    // Should have transitions (exact count depends on valid lines in file)
    QVERIFY(catalogData.size() > 0);
    
    // Verify all transitions have valid frequencies
    for (int i = 0; i < catalogData.size(); ++i) {
        TransitionData trans = catalogData.at(i);
        QVERIFY(trans.frequency > 0);
        QVERIFY(!trans.quantumNumbers.isEmpty());
    }
}

void SPCATParserTest::testParserRegistration()
{
    // Test that SPCAT parser can be registered and found
    CatalogParserRegistry *registry = CatalogParserRegistry::instance();
    
    // Register our parser
    auto parser = std::make_unique<SPCATParser>();
    registry->registerParser(std::move(parser));
    
    // Verify it can be found
    CatalogParser *foundParser = registry->findParser(getTestDataPath("c047527_sample.cat"));
    QVERIFY(foundParser != nullptr);
    QCOMPARE(foundParser->formatName(), QString("SPCAT"));
    
    // Verify supported formats
    QStringList formats = registry->supportedFormats();
    QVERIFY(formats.contains("SPCAT"));
    
    // Verify file extensions
    QStringList extensions = registry->supportedExtensions();
    QVERIFY(extensions.contains("*.cat"));
}

void SPCATParserTest::testInvalidFile()
{
    // Test with non-existent file
    CatalogData emptyData = m_parser->parse("/non/existent/file.cat");
    QVERIFY(emptyData.isEmpty());
    
    // Test with file containing invalid data
    QTemporaryFile invalidFile("invalid_XXXXXX.cat");
    QVERIFY(invalidFile.open());
    invalidFile.write("This is not SPCAT format\n");
    invalidFile.write("Invalid data here\n");
    invalidFile.close();
    
    CatalogData invalidData = m_parser->parse(invalidFile.fileName());
    QVERIFY(invalidData.isEmpty());
}

void SPCATParserTest::testEmptyFile()
{
    // Test with empty file
    QTemporaryFile emptyFile("empty_XXXXXX.cat");
    QVERIFY(emptyFile.open());
    emptyFile.close();
    
    CatalogData emptyData = m_parser->parse(emptyFile.fileName());
    QVERIFY(emptyData.isEmpty());
}

QTEST_MAIN(SPCATParserTest)
#include "tst_spcatparser.moc"