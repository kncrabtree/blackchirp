#include <QtTest>
#include <QTemporaryDir>
#include <QDir>
#include <QStandardPaths>
#include <memory>

// Core overlay architecture
#include <src/gui/overlay/overlaytypespecificwidget.h>
#include <src/gui/overlay/bcexpoverlaywidget.h>
#include <src/gui/overlay/catalogoverlaywidget.h>
#include <src/gui/overlay/unifiedoverlaywidget.h>
#include <src/gui/overlay/unifiedoverlaydialog.h>

// Data structures and processing
#include <src/data/experiment/overlaybase.h>
#include <src/data/experiment/overlaytypes.h>
#include <src/data/processing/overlayprocessmanager.h>
#include <src/data/processing/overlayoperation.h>
#include <src/data/analysis/ftworker.h>
#include <src/data/analysis/ft.h>
#include <src/data/experiment/catalogdata.h>
#include <src/data/processing/parsers/fileparserregistry.h>
#include <src/data/processing/parsers/spcatparser.h>
#include <src/data/processing/parsers/xiamparser.h>

// Storage and experiment data
#include <src/data/storage/blackchirpcsv.h>
#include <src/data/storage/settingsstorage.h>
#include <src/data/experiment/experiment.h>

/**
 * @brief Comprehensive test suite for overlay processing architecture
 * 
 * Tests the operation declaration interface, intelligent async/sync routing,
 * and end-to-end overlay processing without UI dependencies.
 */
class OverlayProcessingTest : public QObject
{
    Q_OBJECT

public:
    OverlayProcessingTest() = default;
    ~OverlayProcessingTest() = default;

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // Operation Declaration Interface Tests
    void testBCExpOperationCapabilities();
    void testCatalogOperationCapabilities();
    void testOperationCapabilityMetadata();
    void testCompileTimeEvaluation();
    
    // Widget Integration Tests
    void testTypeSpecificWidgetCreation();
    void testWidgetCapabilityQueries();
    void testOperationFactoryMethods();
    
    // BCExpOverlay Integration Tests
    void testBCExpOverlayFromTestData();
    void testBCExpProcessingDecision();
    void testBCExpFtComputation();
    
    // CatalogOverlay Integration Tests
    void testCatalogOverlayFromSPCAT();
    void testCatalogOverlayFromXIAM();
    void testCatalogProcessingDecision();
    void testCatalogConvolutionSupport();
    
    // End-to-End Processing Tests
    void testSynchronousProcessingPath();
    void testAsynchronousProcessingPath();
    void testProcessingDecisionLogic();
    
private:
    // Test setup helpers
    QString getTestDataPath(const QString &filename) const;
    void setupTestEnvironment();
    void setupMockFtData();
    std::shared_ptr<OverlayBase> createTestBCExpOverlay();
    std::shared_ptr<OverlayBase> createTestCatalogOverlay();
    
    // Mock objects and test data
    QString m_testDataDir;
    Ft m_mockFt;
    QStringList m_plotNames;
    double m_xRangeMin = 15000.0;  // MHz
    double m_xRangeMax = 40000.0;  // MHz
    double m_yMax = 10.0;
    
    // Test experiment data paths
    QString m_bcexpTestPath;
    QString m_spcatTestPath;
    QString m_xiamTestPath;
};

void OverlayProcessingTest::initTestCase()
{
    setupTestEnvironment();
    setupMockFtData();
    
    // Initialize catalog parser registry for testing
    auto registry = CatalogParserRegistry::instance();
    registry->registerParser(std::make_unique<SPCATParser>());
    registry->registerParser(std::make_unique<XIAMParser>());
    
    // Verify test data exists
    QVERIFY(QDir(m_testDataDir).exists());
    QVERIFY(QFile(m_bcexpTestPath).exists());
    QVERIFY(QFile(m_spcatTestPath).exists());
    QVERIFY(QFile(m_xiamTestPath).exists());
    
    qDebug() << "Test environment initialized:";
    qDebug() << "  Test data directory:" << m_testDataDir;
    qDebug() << "  BCExp test path:" << m_bcexpTestPath;
    qDebug() << "  SPCAT test path:" << m_spcatTestPath;
    qDebug() << "  XIAM test path:" << m_xiamTestPath;
}

void OverlayProcessingTest::cleanupTestCase()
{
    // Cleanup is handled automatically by smart pointers and Qt object trees
}

void OverlayProcessingTest::testBCExpOperationCapabilities()
{
    BCExpOverlayWidget widget;
    
    // Test constexpr getSupportedOperations()
    auto operations = widget.getSupportedOperations();
    QCOMPARE(operations.size(), 2);
    
    // Verify Creation operation
    auto createOp = operations[0];
    QCOMPARE(createOp.type, OperationCapability::Creation);
    QVERIFY(!createOp.isExpensive);  // BCExp creation should be fast
    QCOMPARE(createOp.estimatedDurationMs, 100);
    QCOMPARE(createOp.description, QString("Create BCExperiment overlay"));
    QCOMPARE(createOp.priority, OverlayProcessManager::Priority::Normal);
    
    // Verify Validation operation
    auto validateOp = operations[1];
    QCOMPARE(validateOp.type, OperationCapability::Validation);
    QVERIFY(!validateOp.isExpensive);  // Validation should be fast
    QCOMPARE(validateOp.estimatedDurationMs, 50);
    QCOMPARE(validateOp.description, QString("Validate experiment files"));
    QCOMPARE(validateOp.priority, OverlayProcessManager::Priority::High);
}

void OverlayProcessingTest::testCatalogOperationCapabilities()
{
    CatalogOverlayWidget widget;
    
    // Test constexpr getSupportedOperations()
    auto operations = widget.getSupportedOperations();
    QCOMPARE(operations.size(), 4);
    
    // Verify Creation operation
    auto createOp = operations[0];
    QCOMPARE(createOp.type, OperationCapability::Creation);
    QVERIFY(!createOp.isExpensive);  // File loading should be fast
    QCOMPARE(createOp.estimatedDurationMs, 200);
    QCOMPARE(createOp.description, QString("Create catalog overlay from file"));
    
    // Verify Convolution operation
    auto convOp = operations[1];
    QCOMPARE(convOp.type, OperationCapability::Convolution);
    QVERIFY(convOp.isExpensive);  // Convolution can be expensive
    QCOMPARE(convOp.estimatedDurationMs, 5000);
    QCOMPARE(convOp.description, QString("Apply convolution to catalog data"));
    QCOMPARE(convOp.priority, OverlayProcessManager::Priority::High);
    
    // Verify Validation operation
    auto validateOp = operations[2];
    QCOMPARE(validateOp.type, OperationCapability::Validation);
    QVERIFY(!validateOp.isExpensive);
    
    // Verify PreviewUpdate operation
    auto previewOp = operations[3];
    QCOMPARE(previewOp.type, OperationCapability::PreviewUpdate);
    QVERIFY(previewOp.isExpensive);  // Preview updates can be expensive
    QCOMPARE(previewOp.estimatedDurationMs, 3000);
}

void OverlayProcessingTest::testOperationCapabilityMetadata()
{
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    // Test supportsBackgroundOperation() for BCExp
    QVERIFY(!bcWidget.supportsBackgroundOperation(OperationCapability::Creation));
    QVERIFY(!bcWidget.supportsBackgroundOperation(OperationCapability::Validation));
    QVERIFY(!bcWidget.supportsBackgroundOperation(OperationCapability::Convolution));
    QVERIFY(!bcWidget.supportsBackgroundOperation(OperationCapability::PreviewUpdate));
    
    // Test supportsBackgroundOperation() for Catalog
    QVERIFY(!catWidget.supportsBackgroundOperation(OperationCapability::Creation));
    QVERIFY(!catWidget.supportsBackgroundOperation(OperationCapability::Validation));
    QVERIFY(catWidget.supportsBackgroundOperation(OperationCapability::Convolution));
    QVERIFY(catWidget.supportsBackgroundOperation(OperationCapability::PreviewUpdate));
}

void OverlayProcessingTest::testCompileTimeEvaluation()
{
    // Test that operations are evaluated at compile time
    // This test verifies constexpr functionality by using the results in constant expressions
    
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    // These should compile as constexpr evaluations
    constexpr auto bcExpSupportsCreation = []() constexpr {
        BCExpOverlayWidget widget;
        return widget.supportsBackgroundOperation(OperationCapability::Creation);
    }();
    
    constexpr auto catalogSupportsConvolution = []() constexpr {
        CatalogOverlayWidget widget;
        return widget.supportsBackgroundOperation(OperationCapability::Convolution);
    }();
    
    // Verify compile-time evaluation results
    QVERIFY(!bcExpSupportsCreation);
    QVERIFY(catalogSupportsConvolution);
}

void OverlayProcessingTest::testTypeSpecificWidgetCreation()
{
    // Test creation of type-specific widgets
    auto bcExpWidget = std::make_unique<BCExpOverlayWidget>();
    auto catalogWidget = std::make_unique<CatalogOverlayWidget>();
    
    QVERIFY(bcExpWidget != nullptr);
    QVERIFY(catalogWidget != nullptr);
    
    // Test that they implement the required interface
    QVERIFY(bcExpWidget->getSourceFileConfigWidget() != nullptr);
    QVERIFY(bcExpWidget->getSourceFileSettingsWidget() != nullptr);
    QVERIFY(bcExpWidget->getOverlaySettingsWidget() != nullptr);
    
    QVERIFY(catalogWidget->getSourceFileConfigWidget() != nullptr);
    QVERIFY(catalogWidget->getSourceFileSettingsWidget() != nullptr);
    QVERIFY(catalogWidget->getOverlaySettingsWidget() != nullptr);
}

void OverlayProcessingTest::testWidgetCapabilityQueries()
{
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    // Test that capability queries are consistent
    auto bcOps = bcWidget.getSupportedOperations();
    auto catOps = catWidget.getSupportedOperations();
    
    // BCExp should have fewer operation types than Catalog
    QVERIFY(bcOps.size() < catOps.size());
    
    // Verify operation types are distinct
    QSet<OperationCapability::Type> bcTypes, catTypes;
    for (const auto &op : bcOps) {
        bcTypes.insert(op.type);
    }
    for (const auto &op : catOps) {
        catTypes.insert(op.type);
    }
    
    QVERIFY(bcTypes.contains(OperationCapability::Creation));
    QVERIFY(bcTypes.contains(OperationCapability::Validation));
    QVERIFY(!bcTypes.contains(OperationCapability::Convolution));
    QVERIFY(!bcTypes.contains(OperationCapability::PreviewUpdate));
    
    QVERIFY(catTypes.contains(OperationCapability::Creation));
    QVERIFY(catTypes.contains(OperationCapability::Validation));
    QVERIFY(catTypes.contains(OperationCapability::Convolution));
    QVERIFY(catTypes.contains(OperationCapability::PreviewUpdate));
}

void OverlayProcessingTest::testOperationFactoryMethods()
{
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    // Test BCExp operation factory
    auto bcOp = bcWidget.createOperation(OperationCapability::Creation);
    QVERIFY(bcOp == nullptr);  // BCExp returns nullptr for synchronous processing
    
    // Test Catalog operation factory (would need a real overlay for full test)
    auto catOp = catWidget.createOperation(OperationCapability::Creation);
    QVERIFY(catOp == nullptr);  // Creation returns nullptr (synchronous)
    
    // Note: Full operation creation tests would require real overlay objects
    // which will be tested in the integration tests below
}

void OverlayProcessingTest::testBCExpOverlayFromTestData()
{
    QSKIP("BCExp overlay integration requires experiment loading implementation");
    
    // TODO: Implement when BCExp test data loading is ready
    // This test should:
    // 1. Load experiment from tests/testdata/2638/
    // 2. Create FtWorker and compute FT synchronously
    // 3. Create BCExpOverlay with the computed FT data
    // 4. Verify overlay properties and data
}

void OverlayProcessingTest::testBCExpProcessingDecision()
{
    // Test that BCExp widgets are correctly identified as synchronous
    BCExpOverlayWidget widget;
    
    // All BCExp operations should be synchronous
    QVERIFY(!widget.supportsBackgroundOperation(OperationCapability::Creation));
    QVERIFY(!widget.supportsBackgroundOperation(OperationCapability::Validation));
    
    // Verify that createOperation returns nullptr (synchronous indicator)
    auto op = widget.createOperation(OperationCapability::Creation);
    QVERIFY(op == nullptr);
}

void OverlayProcessingTest::testBCExpFtComputation()
{
    QSKIP("FT computation test requires FID data loading implementation");
    
    // TODO: Implement synchronous FT computation test
    // This test should:
    // 1. Load FID data from test experiment
    // 2. Create FtWorker instance
    // 3. Compute FT synchronously (not through UI signals)
    // 4. Verify FT properties match expected range and characteristics
}

void OverlayProcessingTest::testCatalogOverlayFromSPCAT()
{
    // Test catalog overlay creation from SPCAT file
    auto registry = CatalogParserRegistry::instance();
    auto parser = registry->findParser(m_spcatTestPath);
    QVERIFY(parser != nullptr);
    QCOMPARE(parser->formatName(), QString("SPCAT"));
    
    // Parse catalog data
    CatalogData catalogData = parser->parse(m_spcatTestPath);
    QVERIFY(!catalogData.isEmpty());
    QCOMPARE(catalogData.sourceProgram(), QString("SPCAT"));
    
    // Create catalog overlay
    auto overlay = std::make_shared<CatalogOverlay>();
    overlay->setSourceFile(m_spcatTestPath);
    overlay->setCatalogData(catalogData);
    
    // Verify overlay properties
    QVERIFY(overlay != nullptr);
    QCOMPARE(overlay->type(), OverlayBase::Catalog);
    QCOMPARE(overlay->getSourceFile(), m_spcatTestPath);
    QVERIFY(!overlay->catalogData().isEmpty());
}

void OverlayProcessingTest::testCatalogOverlayFromXIAM()
{
    // Test catalog overlay creation from XIAM file
    auto registry = CatalogParserRegistry::instance();
    auto parser = registry->findParser(m_xiamTestPath);
    QVERIFY(parser != nullptr);
    QCOMPARE(parser->formatName(), QString("XIAM"));
    
    // Parse catalog data
    CatalogData catalogData = parser->parse(m_xiamTestPath);
    QVERIFY(!catalogData.isEmpty());
    QCOMPARE(catalogData.sourceProgram(), QString("XIAM"));
    
    // Create catalog overlay
    auto overlay = std::make_shared<CatalogOverlay>();
    overlay->setSourceFile(m_xiamTestPath);
    overlay->setCatalogData(catalogData);
    
    // Verify overlay properties
    QVERIFY(overlay != nullptr);
    QCOMPARE(overlay->type(), OverlayBase::Catalog);
    QCOMPARE(overlay->getSourceFile(), m_xiamTestPath);
    QVERIFY(!overlay->catalogData().isEmpty());
}

void OverlayProcessingTest::testCatalogProcessingDecision()
{
    // Test that Catalog widgets correctly identify expensive operations
    CatalogOverlayWidget widget;
    
    // Creation and validation should be synchronous
    QVERIFY(!widget.supportsBackgroundOperation(OperationCapability::Creation));
    QVERIFY(!widget.supportsBackgroundOperation(OperationCapability::Validation));
    
    // Convolution and preview updates should be asynchronous
    QVERIFY(widget.supportsBackgroundOperation(OperationCapability::Convolution));
    QVERIFY(widget.supportsBackgroundOperation(OperationCapability::PreviewUpdate));
}

void OverlayProcessingTest::testCatalogConvolutionSupport()
{
    QSKIP("Convolution operation test requires overlay operation implementation");
    
    // TODO: Implement when ConvolutionOperation is available
    // This test should:
    // 1. Create catalog overlay with test data
    // 2. Create ConvolutionOperation through widget factory
    // 3. Execute convolution operation
    // 4. Verify convolved data properties
}

void OverlayProcessingTest::testSynchronousProcessingPath()
{
    // Test that synchronous operations are handled correctly
    BCExpOverlayWidget widget;
    
    // Simulate checking if background processing is supported
    bool supportsBackground = widget.supportsBackgroundOperation(OperationCapability::Creation);
    QVERIFY(!supportsBackground);
    
    // Verify that createOperation returns nullptr for synchronous processing
    auto operation = widget.createOperation(OperationCapability::Creation);
    QVERIFY(operation == nullptr);
    
    // This indicates that the calling code should use synchronous processing
}

void OverlayProcessingTest::testAsynchronousProcessingPath()
{
    // Test that asynchronous operations are identified correctly
    CatalogOverlayWidget widget;
    
    // Convolution should support background processing
    bool supportsBackground = widget.supportsBackgroundOperation(OperationCapability::Convolution);
    QVERIFY(supportsBackground);
    
    // Note: Full operation creation would require a real overlay object
    // but the capability detection is sufficient for routing logic
}

void OverlayProcessingTest::testProcessingDecisionLogic()
{
    // Test the core logic that UnifiedOverlayDialog uses for routing
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    // Simulate the decision logic in UnifiedOverlayDialog::createOverlayAsync()
    
    // For BCExp widget - should choose synchronous processing
    bool bcNeedsBackground = bcWidget.supportsBackgroundOperation(OperationCapability::Creation);
    QVERIFY(!bcNeedsBackground);  // Should use synchronous path
    
    // For Catalog widget with convolution - should choose asynchronous processing
    bool catNeedsBackground = catWidget.supportsBackgroundOperation(OperationCapability::Convolution);
    QVERIFY(catNeedsBackground);  // Should use asynchronous path
    
    // Verify that the widget-agnostic approach works
    OverlayTypeSpecificWidget* bcPtr = &bcWidget;
    OverlayTypeSpecificWidget* catPtr = &catWidget;
    
    QVERIFY(!bcPtr->supportsBackgroundOperation(OperationCapability::Creation));
    QVERIFY(catPtr->supportsBackgroundOperation(OperationCapability::Convolution));
}

// Helper method implementations

QString OverlayProcessingTest::getTestDataPath(const QString &filename) const
{
    return QDir(m_testDataDir).absoluteFilePath(filename);
}

void OverlayProcessingTest::setupTestEnvironment()
{
    // Find test data directory using the same logic as other tests
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
    
    // Set up test file paths
    m_bcexpTestPath = getTestDataPath("2638");  // Experiment directory
    m_spcatTestPath = getTestDataPath("c047527_full.cat");
    m_xiamTestPath = getTestDataPath("test_aprint32_small.xo");
    
    // Set up plot names for testing
    m_plotNames << "Upper Sideband" << "Lower Sideband";
}

void OverlayProcessingTest::setupMockFtData()
{
    // Create mock Ft data matching the specified test parameters
    // Range: 15000-40000 MHz, yMax: 10
    
    int numPoints = 1000;
    double spacing = (m_xRangeMax - m_xRangeMin) / (numPoints - 1);
    
    QVector<QPointF> ftData;
    ftData.reserve(numPoints);
    
    for (int i = 0; i < numPoints; ++i) {
        double freq = m_xRangeMin + i * spacing;
        double intensity = m_yMax * qExp(-qPow((freq - 27500.0) / 5000.0, 2)); // Gaussian peak at center
        ftData.append(QPointF(freq, intensity));
    }
    
    m_mockFt = Ft(ftData);
    
    qDebug() << "Mock FT data created:";
    qDebug() << "  Range:" << m_xRangeMin << "-" << m_xRangeMax << "MHz";
    qDebug() << "  Points:" << numPoints;
    qDebug() << "  Max intensity:" << m_yMax;
}

std::shared_ptr<OverlayBase> OverlayProcessingTest::createTestBCExpOverlay()
{
    // TODO: Implement when BCExp overlay creation is ready
    return nullptr;
}

std::shared_ptr<OverlayBase> OverlayProcessingTest::createTestCatalogOverlay()
{
    // Create a basic catalog overlay for testing
    auto overlay = std::make_shared<CatalogOverlay>();
    
    // Set basic properties
    overlay->setSourceFile(m_spcatTestPath);
    
    // Parse catalog data
    auto registry = CatalogParserRegistry::instance();
    auto parser = registry->findParser(m_spcatTestPath);
    if (parser) {
        CatalogData catalogData = parser->parse(m_spcatTestPath);
        overlay->setCatalogData(catalogData);
    }
    
    return overlay;
}

QTEST_MAIN(OverlayProcessingTest)
#include "tst_overlayprocessing.moc"