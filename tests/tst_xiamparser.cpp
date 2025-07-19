#include <QtTest>
#include <QTemporaryFile>
#include <QDir>

#include <src/data/processing/parsers/xiamparser.h>
#include <src/data/processing/parsers/fileparserregistry.h>

class XIAMParserTest : public QObject
{
    Q_OBJECT
public:
    XIAMParserTest() {}
    ~XIAMParserTest() {}

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testCanParse();
    void testParseInts2Format();
    void testParseInts3Format();
    void testRigidLinesExcluded();
    void testIntensityCalculation();
    void testIncompleteGroups();
    void testParserRegistration();
    void testInvalidFile();
    void testAprintFormatVariations();
    void testProblematicFixedColumnFormat();

private:
    QString getTestDataPath(const QString &filename) const;
    void createTestFiles();
    
    XIAMParser *m_parser;
    QString m_testDataDir;
};

void XIAMParserTest::initTestCase()
{
    m_parser = new XIAMParser();
    
    // Get test data directory path
    QDir currentDir = QDir::current();
    if (currentDir.dirName().startsWith("build-")) {
        currentDir.cdUp();
    }
    
    if (currentDir.exists("src")) {
        m_testDataDir = currentDir.absoluteFilePath("src/tests/testdata");
    } else {
        QDir searchDir = currentDir;
        while (!searchDir.exists("tests") && searchDir.cdUp()) {}
        m_testDataDir = searchDir.absoluteFilePath("tests/testdata");
    }
    
    // Create test data directory if it doesn't exist
    QDir().mkpath(m_testDataDir);
    
    // Create test files
    createTestFiles();
    
    qDebug() << "Test data directory:" << m_testDataDir;
}

void XIAMParserTest::cleanupTestCase()
{
    delete m_parser;
}

void XIAMParserTest::createTestFiles()
{
    // Create ints=2 test file
    QString ints2Content = R"(Fri Jul 11 06:54:28 2025

 Rotational, Centrifugal Distortion, Internal Rotation Calculation (V2.5e)
                       Holger Hartwig 08-Nov-96 (hartwig@phc.uni-kiel.de)

cis-MMA

   nzyk      5000   print        4   eval         0   dfreq        0
   orger        0   ints         2   maxm         8   woods       33

-- B 1                                 Freq Linestr.    total  stat.w.   popul. hv-ener.
  2  2  0   1  1  1   S 1  V 1     9.670897   3.7831   0.0099   3.0000   0.0136   0.0642  B 1  K  2 -1  t  5  2
  2  2  1   1  1  0   S 1  V 1     9.190608   4.3294   0.0107   3.0000   0.0135   0.0611  B 1  K -2  1  t  4  3
  3  0  3   2  0  2   S 1  V 1     6.285087   0.1938   0.0005   5.0000   0.0133   0.0422  B 1  K  0  0  t  1  1
  3  1  3   2  0  2   S 1  V 1     7.047286   6.1154   0.0192   5.0000   0.0133   0.0472  B 1  K -1  0  t  2  1

total is the product of Linestr., population-factor,
energy factor (hv), and the statistical weight
)";

    QFile ints2File(getTestDataPath("test_ints2.xo"));
    ints2File.open(QIODevice::WriteOnly | QIODevice::Text);
    ints2File.write(ints2Content.toUtf8());
    ints2File.close();

    // Create ints=3 test file
    QString ints3Content = R"(Fri Jul 11 06:56:22 2025

 Rotational, Centrifugal Distortion, Internal Rotation Calculation (V2.5e)
                       Holger Hartwig 08-Nov-96 (hartwig@phc.uni-kiel.de)

cis-MMA

   nzyk      5000   print        4   eval         0   dfreq        0
   orger        0   ints         3   maxm         8   woods       33

-- B 1                                    Freq       Split Linestr.    total  stat.w.   popul. hv-ener.
  2  2  0   1  1  1         rigid       9.633953             3.7902   0.7088   3.0000   0.9753   0.0639  K  2 -1  t  5  2
  2  2  0   1  1  1   S 1  V 1  B 1     9.670897             3.7831   0.0099   3.0000   0.0136   0.0642  K  2 -1  t  5  2
  2  2  1   1  1  0         rigid       9.158358             4.3350   0.7698   3.0000   0.9726   0.0609  K -2  1  t  4  3
  2  2  1   1  1  0   S 1  V 1  B 1     9.190608             4.3294   0.0107   3.0000   0.0135   0.0611  K -2  1  t  4  3
                      S 2               6.095339-3095.2694   0.2533   0.0003   3.0000   0.0082   0.0409  K  2 -1  t  4  3
  3  1  3   2  0  2         rigid       7.037981             6.1221   1.3793   5.0000   0.9565   0.0471  K -1  0  t  2  1
  3  1  3   2  0  2   S 1  V 1  B 1     7.047286             6.1154   0.0192   5.0000   0.0133   0.0472  K -1  0  t  2  1
                      S 2               7.151815  104.5291   5.2448   0.0102   5.0000   0.0081   0.0479  K  1  0  t  2  1

total is the product of Linestr., population-factor,
energy factor (hv), and the statistical weight
)";

    QFile ints3File(getTestDataPath("test_ints3.xo"));
    ints3File.open(QIODevice::WriteOnly | QIODevice::Text);
    ints3File.write(ints3Content.toUtf8());
    ints3File.close();

    // Create file with incomplete groups (intensity cutoff simulation)
    QString incompleteContent = R"(Fri Jul 11 06:56:22 2025

 Rotational, Centrifugal Distortion, Internal Rotation Calculation (V2.5e)
                       Holger Hartwig 08-Nov-96 (hartwig@phc.uni-kiel.de)

test-molecule

   ints         3

-- B 1                                    Freq       Split Linestr.    total  stat.w.   popul. hv-ener.
  2  2  0   1  1  1   S 1  V 1  B 1     9.670897             3.7831   0.0099   3.0000   0.0136   0.0642  K  2 -1  t  5  2
  3  1  3   2  0  2         rigid       7.037981             6.1221   1.3793   5.0000   0.9565   0.0471  K -1  0  t  2  1
  3  1  3   2  0  2   S 1  V 1  B 1     7.047286             6.1154   0.0192   5.0000   0.0133   0.0472  K -1  0  t  2  1
                      S 2               7.151815  104.5291   5.2448   0.0102   5.0000   0.0081   0.0479  K  1  0  t  2  1
  4  1  4   3  0  3   S 1  V 1  B 1     8.575836             8.3828   0.0427   7.0000   0.0127   0.0571  K -1  0  t  2  1
)";

    QFile incompleteFile(getTestDataPath("test_incomplete.xo"));
    incompleteFile.open(QIODevice::WriteOnly | QIODevice::Text);
    incompleteFile.write(incompleteContent.toUtf8());
    incompleteFile.close();
}

QString XIAMParserTest::getTestDataPath(const QString &filename) const
{
    return QDir(m_testDataDir).absoluteFilePath(filename);
}

void XIAMParserTest::testCanParse()
{
    // Test valid XIAM files
    QVERIFY(m_parser->canParse(getTestDataPath("test_ints2.xo")));
    QVERIFY(m_parser->canParse(getTestDataPath("test_ints3.xo")));
    
    // Test invalid extensions
    QTemporaryFile invalidFile("test_XXXXXX.txt");
    QVERIFY(invalidFile.open());
    invalidFile.write("some content");
    invalidFile.close();
    QVERIFY(!m_parser->canParse(invalidFile.fileName()));
}

void XIAMParserTest::testParseInts2Format()
{
    CatalogData catalogData = m_parser->parse(getTestDataPath("test_ints2.xo"));
    
    QVERIFY(!catalogData.isEmpty());
    QCOMPARE(catalogData.size(), 4);
    QCOMPARE(catalogData.sourceProgram(), QString("XIAM"));
    QCOMPARE(catalogData.moleculeName(), QString("cis-MMA"));
    
    // Check first transition (frequency converted from GHz to MHz)
    TransitionData trans1 = catalogData.at(0);
    QCOMPARE(trans1.frequency, 9670.897);
    QCOMPARE(trans1.quantumNumbers, QString("  2  2  0 -   1  1  1,  S 1  V 1  B 1"));
    QVERIFY(trans1.additionalData.contains("quantumAssignment"));
    QCOMPARE(trans1.additionalData.value("quantumAssignment").toString(), QString("B 1  K  2 -1  t  5  2"));
}

void XIAMParserTest::testParseInts3Format()
{
    CatalogData catalogData = m_parser->parse(getTestDataPath("test_ints3.xo"));
    
    QVERIFY(!catalogData.isEmpty());
    QCOMPARE(catalogData.sourceProgram(), QString("XIAM"));
    QCOMPARE(catalogData.moleculeName(), QString("cis-MMA"));
    
    // Should have transitions but NOT the rigid rotor lines
    QVERIFY(catalogData.size() > 0);
    
    
    // Check that no transitions contain "rigid" in their mode
    for (int i = 0; i < catalogData.size(); ++i) {
        TransitionData trans = catalogData.at(i);
        QString mode = trans.additionalData.value("mode").toString();
        QVERIFY(!mode.contains("rigid"));
    }
}

void XIAMParserTest::testRigidLinesExcluded()
{
    CatalogData catalogData = m_parser->parse(getTestDataPath("test_ints3.xo"));
    
    // Count expected transitions (should exclude rigid lines)
    // From test data: 5 non-rigid transitions expected
    // 3 S1 V1 B1 lines + 2 S2 split lines = 5 total
    int expectedCount = 5;
    QCOMPARE(catalogData.size(), expectedCount);
    
    // Verify none of the transitions are rigid rotor references
    for (int i = 0; i < catalogData.size(); ++i) {
        TransitionData trans = catalogData.at(i);
        QString mode = trans.additionalData.value("mode").toString();
        QVERIFY2(!mode.contains("rigid"), 
                 QString("Found rigid line at index %1: %2").arg(i).arg(mode).toLocal8Bit());
    }
}

void XIAMParserTest::testIntensityCalculation()
{
    CatalogData catalogData = m_parser->parse(getTestDataPath("test_ints2.xo"));
    QVERIFY(!catalogData.isEmpty());
    
    // Test that intensity calculation works
    TransitionData trans = catalogData.at(0);
    double linestr = trans.additionalData.value("linestrength").toDouble();
    double total = trans.additionalData.value("total").toDouble();
    double population = trans.additionalData.value("population").toDouble();
    double hvEnergy = trans.additionalData.value("hvEnergy").toDouble();
    double statWeight = trans.additionalData.value("statisticalWeight").toDouble();
    
    // With the new logic: intensity should be total (default) unless linestr, population, 
    // and hvEnergy all have 2+ significant digits, then it should be linestr * statWeight * population * hvEnergy
    
    // Test that intensity is either total or the calculated product
    double calculatedProduct = linestr * statWeight * population * hvEnergy;
    QVERIFY(qAbs(trans.intensity - total) < 1e-6 || qAbs(trans.intensity - calculatedProduct) < 1e-6);
}

void XIAMParserTest::testIncompleteGroups()
{
    CatalogData catalogData = m_parser->parse(getTestDataPath("test_incomplete.xo"));
    
    QVERIFY(!catalogData.isEmpty());
    
    // Should handle incomplete groups gracefully
    QVERIFY(catalogData.size() > 0);
    
    // Verify no rigid lines are included
    for (int i = 0; i < catalogData.size(); ++i) {
        TransitionData trans = catalogData.at(i);
        QString mode = trans.additionalData.value("mode").toString();
        QVERIFY(!mode.contains("rigid"));
    }
}

void XIAMParserTest::testParserRegistration()
{
    FileParserRegistry *registry = FileParserRegistry::instance();
    
    auto parser = std::make_unique<XIAMParser>();
    registry->registerParser(std::move(parser));
    
    CatalogParser *foundParser = registry->findParserOfType<CatalogParser>(getTestDataPath("test_ints2.xo"));
    QVERIFY(foundParser != nullptr);
    QCOMPARE(foundParser->formatName(), QString("XIAM"));
    
    QStringList formats = registry->supportedFormats();
    QVERIFY(formats.contains("XIAM"));
    
    QStringList extensions = registry->supportedExtensions();
    QVERIFY(extensions.contains("*.xo"));
}

void XIAMParserTest::testInvalidFile()
{
    CatalogData emptyData = m_parser->parse("/non/existent/file.xo");
    QVERIFY(emptyData.isEmpty());
    
    QTemporaryFile invalidFile("invalid_XXXXXX.xo");
    QVERIFY(invalidFile.open());
    invalidFile.write("This is not XIAM format\nInvalid data here\n");
    invalidFile.close();
    
    CatalogData invalidData = m_parser->parse(invalidFile.fileName());
    QVERIFY(invalidData.isEmpty());
}

void XIAMParserTest::testAprintFormatVariations()
{
    // Test aprint=10 (standard format)
    QString aprint10File = getTestDataPath("test_aprint10.xo");
    if (QFile::exists(aprint10File)) {
        QVERIFY(m_parser->canParse(aprint10File));
        CatalogData data10 = m_parser->parse(aprint10File);
        QVERIFY(!data10.isEmpty());
        QCOMPARE(data10.sourceProgram(), QString("XIAM"));
        QCOMPARE(data10.moleculeName(), QString("mtbe parent"));
        
        // Should have many transitions from this large catalog
        QVERIFY(data10.size() > 100);
    }
    
    // Test aprint=32778 (extended format with eigenvectors)
    QString aprint32778File = getTestDataPath("test_aprint32778.xo");
    if (QFile::exists(aprint32778File)) {
        QVERIFY(m_parser->canParse(aprint32778File));
        CatalogData data32778 = m_parser->parse(aprint32778File);
        QVERIFY(!data32778.isEmpty());
        QCOMPARE(data32778.sourceProgram(), QString("XIAM"));
        QCOMPARE(data32778.moleculeName(), QString("mtbe parent"));
        
        // Should have many transitions from this large catalog
        QVERIFY(data32778.size() > 100);
        
        // The key test: both formats should produce identical transition data
        // despite different aprint settings
        if (QFile::exists(aprint10File)) {
            CatalogData data10 = m_parser->parse(aprint10File);
            
            // Should have same number of transitions
            QCOMPARE(data32778.size(), data10.size());
            
            // Spot check: first few transitions should be identical
            if (data10.size() > 5 && data32778.size() > 5) {
                for (int i = 0; i < qMin(5, qMin(data10.size(), data32778.size())); ++i) {
                    TransitionData trans10 = data10.at(i);
                    TransitionData trans32778 = data32778.at(i);
                    
                    QCOMPARE(trans32778.frequency, trans10.frequency);
                    QCOMPARE(trans32778.quantumNumbers, trans10.quantumNumbers);
                    // Allow small intensity differences due to precision
                    QVERIFY(qAbs(trans32778.intensity - trans10.intensity) < 1e-6);
                }
            }
        }
    }
}

void XIAMParserTest::testProblematicFixedColumnFormat()
{
    // Test the problematic pred.xo file that uses fixed-column formatting
    QString problematicFile = getTestDataPath("pred_ints3_problematic.xo");
    
    // First, verify the file exists (copied from bcfitting test data)
    if (!QFile::exists(problematicFile)) {
        QSKIP("Problematic test file not found - skipping fixed-column format test");
        return;
    }
    
    // Test that we can identify it as XIAM format
    QVERIFY2(m_parser->canParse(problematicFile), 
             "Parser should be able to identify this as XIAM format");
    
    // Test parsing
    CatalogData catalogData = m_parser->parse(problematicFile);
    
    // The file should parse successfully and contain transitions
    QVERIFY2(!catalogData.isEmpty(), 
             "Parser should successfully extract transitions from fixed-column format");
    
    QCOMPARE(catalogData.sourceProgram(), QString("XIAM"));
    QCOMPARE(catalogData.moleculeName(), QString("cyclotene"));
    
    
    // Verify we get a reasonable number of transitions
    // The file is large so it should have many transitions
    QVERIFY2(catalogData.size() > 10, 
             QString("Expected > 10 transitions, got %1").arg(catalogData.size()).toLocal8Bit());
    
    // Test a specific transition to verify correct parsing
    // Look for a transition we know should be in the file
    bool foundExpectedTransition = false;
    for (int i = 0; i < catalogData.size(); ++i) {
        TransitionData trans = catalogData.at(i);
        
        // Check for transition around 28028.253 MHz (28.028253 GHz converted to MHz)
        if (qAbs(trans.frequency - 28028.253) < 0.001) {
            foundExpectedTransition = true;
            
            // Verify it has the expected quantum numbers structure
            QVERIFY(!trans.quantumNumbers.isEmpty());
            QVERIFY(trans.intensity > 0);
            
            // Verify quantum numbers are parsed correctly (mode parsing was simplified)
            QVERIFY(trans.quantumNumbers.contains("S 1"));
            QVERIFY(trans.quantumNumbers.contains("V 1"));
            QVERIFY(trans.quantumNumbers.contains("B 1"));
            break;
        }
    }
    
    QVERIFY2(foundExpectedTransition, "Should find the expected transition at 28028.253 MHz");
    
    // Verify no rigid lines are included (they should be filtered out)
    for (int i = 0; i < catalogData.size(); ++i) {
        TransitionData trans = catalogData.at(i);
        QString mode = trans.additionalData.value("mode").toString();
        QVERIFY2(!mode.contains("rigid"), 
                 QString("Found rigid line at index %1").arg(i).toLocal8Bit());
    }
}

QTEST_MAIN(XIAMParserTest)
#include "tst_xiamparser.moc"