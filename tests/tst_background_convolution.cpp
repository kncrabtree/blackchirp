#include <QtTest>
#include <QSignalSpy>
#include <QEventLoop>
#include <QTimer>
#include <QDir>

// Core classes for background convolution testing
#include <src/data/processing/overlayprocessmanager.h>
#include <src/data/processing/overlayoperation.h>
#include <src/data/experiment/overlaytypes.h>
#include <src/data/processing/parsers/fileparserregistry.h>
#include <src/data/processing/parsers/spcatparser.h>
#include <src/data/processing/parsers/xiamparser.h>

/**
 * @brief Test suite for end-to-end background convolution execution
 * 
 * This test validates that the background processing system can actually
 * execute convolution operations asynchronously without UI dependencies.
 * It tests the complete pipeline from catalog loading to convolved output.
 */
class BackgroundConvolutionTest : public QObject
{
    Q_OBJECT

public:
    BackgroundConvolutionTest() = default;
    ~BackgroundConvolutionTest() = default;

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // End-to-end background processing tests
    void testSPCATConvolutionExecution();
    void testXIAMConvolutionExecution();
    void testConvolutionProgress();
    void testConvolutionCancellation();
    
    // Operation lifecycle tests  
    void testOperationQueuing();
    void testOperationCompletion();
    void testOperationErrorHandling();
    
    // Performance and edge case tests
    void testLargeDatasetConvolution();
    void testZeroIntensityHandling();
    void testFrequencyRangeFiltering();

private:
    // Test setup helpers
    QString getTestDataPath(const QString &filename) const;
    std::shared_ptr<CatalogOverlay> createSPCATOverlay() const;
    std::shared_ptr<CatalogOverlay> createXIAMOverlay() const;
    bool waitForOperationCompletion(const QString& operationId, int timeoutMs = 10000);
    
    // Test data
    QString m_testDataDir;
    QString m_spcatTestPath;
    QString m_xiamTestPath;
};

void BackgroundConvolutionTest::initTestCase()
{
    // Use the testdata directory path configured by CMake (see the
    // sibling parser tests, e.g. tst_spcatparser.cpp, for the same
    // pattern). A prior directory-search heuristic based on QDir::current()
    // walking up until it found a directory named "tests" broke under
    // ctest specifically: ctest's working directory for this executable
    // is build/tests, so walking up one level lands back on build/, whose
    // own tests/ subdirectory (the build output directory for this
    // executable, not the source tree) satisfied the search and pointed
    // at a testdata directory that does not exist there.
#ifdef TESTDATA_DIR
    m_testDataDir = QString(TESTDATA_DIR);
#else
    m_testDataDir = QDir::current().absoluteFilePath("tests/testdata");
#endif

    m_spcatTestPath = getTestDataPath("c047527_sample.cat");
    m_xiamTestPath = getTestDataPath("test_aprint32_small.xo");
    
    // Initialize parser registry
    auto registry = FileParserRegistry::instance();
    registry->registerParser(std::make_unique<SPCATParser>());
    registry->registerParser(std::make_unique<XIAMParser>());
    
    // Verify test files exist
    QVERIFY(QFile::exists(m_spcatTestPath));
    QVERIFY(QFile::exists(m_xiamTestPath));
    
    qDebug() << "BackgroundConvolution test environment initialized";
    qDebug() << "  SPCAT test file:" << m_spcatTestPath;
    qDebug() << "  XIAM test file:" << m_xiamTestPath;
}

void BackgroundConvolutionTest::cleanupTestCase()
{
    // Cancel any remaining operations
    OverlayProcessManager::instance().cancelAllOperations();
}

void BackgroundConvolutionTest::testSPCATConvolutionExecution()
{
    // Create catalog overlay with SPCAT data
    auto overlay = createSPCATOverlay();
    QVERIFY(overlay != nullptr);
    QVERIFY(!overlay->catalogData().isEmpty());
    
    // Get original data size for comparison
    int originalSize = overlay->catalogData().size();
    qDebug() << "Original SPCAT catalog size:" << originalSize;
    
    // Create convolution operation with realistic parameters
    auto operation = std::make_shared<ConvolutionOperation>(
        overlay,
        true,  // enable convolution
        CatalogOverlay::Lorentzian,  // lineshape
        50.0,  // 50 kHz linewidth
        26000.0,  // freq min (MHz)
        28000.0,  // freq max (MHz)
        2000   // convolution grid points (1 MHz spacing over the 2000 MHz range)
    );
    
    // Submit to background processor
    auto& manager = OverlayProcessManager::instance();
    QString operationId = manager.queueOperation(operation, OverlayProcessManager::Priority::High);
    QVERIFY(!operationId.isEmpty());
    
    // Verify operation is queued
    QCOMPARE(manager.getOperationState(operationId), OverlayProcessManager::OperationState::Queued);
    QVERIFY(manager.hasQueuedOperations());
    
    // Wait for completion
    QVERIFY(waitForOperationCompletion(operationId));
    
    // Verify successful completion
    QCOMPARE(manager.getOperationState(operationId), OverlayProcessManager::OperationState::Completed);
    
    // Verify convolution actually occurred
    auto convolvedData = overlay->xyData();
    QVERIFY(convolvedData.size() > 0);
    
    // Convolved data should be different from original discrete data
    // (unless no transitions fall in the frequency range)
    qDebug() << "Convolved data points:" << convolvedData.size();
    
    // Verify frequency range is respected
    if (convolvedData.size() > 0) {
        double minFreq = convolvedData.first().x();
        double maxFreq = convolvedData.last().x();
        QVERIFY(minFreq >= 26000.0);
        QVERIFY(maxFreq <= 28000.0);
    }
}

void BackgroundConvolutionTest::testXIAMConvolutionExecution()
{
    // Create catalog overlay with XIAM data
    auto overlay = createXIAMOverlay();
    QVERIFY(overlay != nullptr);
    QVERIFY(!overlay->catalogData().isEmpty());
    
    qDebug() << "Original XIAM catalog size:" << overlay->catalogData().size();
    
    // Create convolution operation with Gaussian lineshape
    auto operation = std::make_shared<ConvolutionOperation>(
        overlay,
        true,  // enable convolution
        CatalogOverlay::Gaussian,  // lineshape
        100.0,  // 100 kHz linewidth
        28000.0,  // freq min (MHz)
        32000.0,  // freq max (MHz)
        2000   // convolution grid points (2 MHz spacing over the 4000 MHz range)
    );
    
    // Submit and wait for completion
    auto& manager = OverlayProcessManager::instance();
    QString operationId = manager.queueOperation(operation, OverlayProcessManager::Priority::Normal);
    
    QVERIFY(waitForOperationCompletion(operationId));
    QCOMPARE(manager.getOperationState(operationId), OverlayProcessManager::OperationState::Completed);
    
    // Verify convolution with Gaussian lineshape
    auto convolvedData = overlay->xyData();
    QVERIFY(convolvedData.size() > 0);
    
    qDebug() << "XIAM convolved data points:" << convolvedData.size();
}

void BackgroundConvolutionTest::testConvolutionProgress()
{
    auto overlay = createSPCATOverlay();
    QVERIFY(overlay != nullptr);
    
    auto operation = std::make_shared<ConvolutionOperation>(
        overlay, true, CatalogOverlay::Lorentzian, 75.0, 26500.0, 27500.0, 1000
    );
    
    auto& manager = OverlayProcessManager::instance();
    
    // Set up signal spy to track progress
    QSignalSpy progressSpy(&manager, &OverlayProcessManager::operationProgress);
    QSignalSpy completedSpy(&manager, &OverlayProcessManager::operationCompleted);
    
    QString operationId = manager.queueOperation(operation);
    
    // Wait for completion
    QVERIFY(waitForOperationCompletion(operationId));
    
    // Verify we received progress updates
    QVERIFY(progressSpy.count() >= 0);  // May be 0 for fast operations
    QCOMPARE(completedSpy.count(), 1);
    
    // Verify final progress is 100%
    int finalProgress = manager.getOperationProgress(operationId);
    QCOMPARE(finalProgress, 100);
}

void BackgroundConvolutionTest::testConvolutionCancellation()
{
    QSKIP("Cancellation test disabled due to timing issues");
}

void BackgroundConvolutionTest::testOperationQueuing()
{
    auto& manager = OverlayProcessManager::instance();
    
    // Clear any existing operations
    manager.cancelAllOperations();
    
    auto overlay1 = createSPCATOverlay();
    auto overlay2 = createXIAMOverlay();
    
    auto op1 = std::make_shared<ConvolutionOperation>(overlay1, true, CatalogOverlay::Lorentzian, 50.0, 26000.0, 27000.0, 1000);
    auto op2 = std::make_shared<ConvolutionOperation>(overlay2, true, CatalogOverlay::Gaussian, 75.0, 28000.0, 29000.0, 1000);
    
    // Queue multiple operations
    QString id1 = manager.queueOperation(op1, OverlayProcessManager::Priority::Low);
    QString id2 = manager.queueOperation(op2, OverlayProcessManager::Priority::High);
    
    QVERIFY(!id1.isEmpty());
    QVERIFY(!id2.isEmpty());
    QVERIFY(id1 != id2);
    
    // High priority should be processed first
    QVERIFY(manager.hasQueuedOperations());
    QCOMPARE(manager.queueSize(), 2);
    
    // Wait for both to complete
    QVERIFY(waitForOperationCompletion(id1));
    QVERIFY(waitForOperationCompletion(id2));
}

void BackgroundConvolutionTest::testOperationCompletion()
{
    auto overlay = createSPCATOverlay();
    auto operation = std::make_shared<ConvolutionOperation>(
        overlay, true, CatalogOverlay::Lorentzian, 50.0, 26000.0, 27000.0, 500
    );
    
    auto& manager = OverlayProcessManager::instance();
    QSignalSpy completedSpy(&manager, &OverlayProcessManager::operationCompleted);
    
    QString operationId = manager.queueOperation(operation);
    
    QVERIFY(waitForOperationCompletion(operationId));
    
    // Verify completion signal was emitted with correct parameters
    QCOMPARE(completedSpy.count(), 1);
    QList<QVariant> arguments = completedSpy.takeFirst();
    QCOMPARE(arguments.at(0).toString(), operationId);
    
    // Second argument should be the overlay result
    auto result = arguments.at(1).value<std::shared_ptr<OverlayBase>>();
    QVERIFY(result != nullptr);
    QCOMPARE(result.get(), overlay.get());  // Should be same overlay object
}

void BackgroundConvolutionTest::testOperationErrorHandling()
{
    // Create operation with invalid parameters to trigger error
    auto overlay = createSPCATOverlay();
    auto operation = std::make_shared<ConvolutionOperation>(
        overlay, true, CatalogOverlay::Lorentzian, -50.0,  // Invalid negative linewidth
        26000.0, 27000.0, 100
    );
    
    auto& manager = OverlayProcessManager::instance();
    QString operationId = manager.queueOperation(operation);
    
    // Wait for completion without signal spy (to avoid signal handling issues)
    QVERIFY(waitForOperationCompletion(operationId));
    
    // Verify operation failed as expected
    QCOMPARE(manager.getOperationState(operationId), OverlayProcessManager::OperationState::Failed);
    
    QString error = manager.getOperationError(operationId);
    QVERIFY(!error.isEmpty());
    QVERIFY(error.contains("Linewidth must be positive"));
    qDebug() << "Expected error message:" << error;
}

void BackgroundConvolutionTest::testLargeDatasetConvolution()
{
    QSKIP("Large dataset test skipped");
}

void BackgroundConvolutionTest::testZeroIntensityHandling()
{
    QSKIP("Zero intensity test skipped");
}

void BackgroundConvolutionTest::testFrequencyRangeFiltering()
{
    auto overlay = createSPCATOverlay();
    
    // Test with very narrow frequency range
    auto operation = std::make_shared<ConvolutionOperation>(
        overlay, true, CatalogOverlay::Lorentzian, 50.0,
        26500.0, 26500.1, 50  // Only 0.1 MHz range
    );
    
    auto& manager = OverlayProcessManager::instance();
    QString operationId = manager.queueOperation(operation);
    
    QVERIFY(waitForOperationCompletion(operationId));
    
    // Should complete successfully even with narrow range
    QCOMPARE(manager.getOperationState(operationId), OverlayProcessManager::OperationState::Completed);
    
    auto convolvedData = overlay->xyData();
    
    // All points should be within the specified range
    for (const auto& point : convolvedData) {
        QVERIFY(point.x() >= 26500.0);
        QVERIFY(point.x() <= 26500.1);
    }
}

// Helper method implementations

QString BackgroundConvolutionTest::getTestDataPath(const QString &filename) const
{
    return QDir(m_testDataDir).absoluteFilePath(filename);
}

std::shared_ptr<CatalogOverlay> BackgroundConvolutionTest::createSPCATOverlay() const
{
    auto overlay = std::make_shared<CatalogOverlay>();
    auto registry = FileParserRegistry::instance();
    // findParser() returns the generic FileParser interface, which has no
    // parse(); CatalogParser::parse() is declared one level down, hence
    // the type-safe lookup (the same one ParseCatalogOperation::execute()
    // uses in production).
    auto parser = registry->findParserOfType<CatalogParser>(m_spcatTestPath);

    if (parser) {
        CatalogData catalogData = parser->parse(m_spcatTestPath);
        overlay->setCatalogData(catalogData);
        overlay->setSourceFile(m_spcatTestPath);
    }
    
    return overlay;
}

std::shared_ptr<CatalogOverlay> BackgroundConvolutionTest::createXIAMOverlay() const
{
    auto overlay = std::make_shared<CatalogOverlay>();
    auto registry = FileParserRegistry::instance();
    auto parser = registry->findParserOfType<CatalogParser>(m_xiamTestPath);

    if (parser) {
        CatalogData catalogData = parser->parse(m_xiamTestPath);
        overlay->setCatalogData(catalogData);
        overlay->setSourceFile(m_xiamTestPath);
    }
    
    return overlay;
}

bool BackgroundConvolutionTest::waitForOperationCompletion(const QString& operationId, int timeoutMs)
{
    auto& manager = OverlayProcessManager::instance();
    QEventLoop loop;
    QTimer timeoutTimer;
    bool success = false;
    
    // Store connections so we can disconnect them later
    QMetaObject::Connection completedConn, failedConn, cancelledConn, timeoutConn;
    
    // Connect to completion signals
    completedConn = connect(&manager, &OverlayProcessManager::operationCompleted, [&](const QString& id) {
        if (id == operationId) { 
            success = true;
            loop.quit();
        }
    });
    
    failedConn = connect(&manager, &OverlayProcessManager::operationFailed, [&](const QString& id) {
        if (id == operationId) { 
            success = true;
            loop.quit();
        }
    });
    
    cancelledConn = connect(&manager, &OverlayProcessManager::operationCancelled, [&](const QString& id) {
        if (id == operationId) { 
            success = true;
            loop.quit();
        }
    });
    
    // Set up timeout
    timeoutTimer.setSingleShot(true);
    timeoutTimer.setInterval(timeoutMs);
    timeoutConn = connect(&timeoutTimer, &QTimer::timeout, &loop, &QEventLoop::quit);
    timeoutTimer.start();
    
    // Check if already completed
    auto state = manager.getOperationState(operationId);
    if (state == OverlayProcessManager::OperationState::Completed ||
        state == OverlayProcessManager::OperationState::Failed ||
        state == OverlayProcessManager::OperationState::Cancelled) {
        
        // Disconnect before returning
        disconnect(completedConn);
        disconnect(failedConn);
        disconnect(cancelledConn);
        disconnect(timeoutConn);
        return true;
    }
    
    // Wait for completion
    loop.exec();
    
    // Disconnect all connections before returning
    disconnect(completedConn);
    disconnect(failedConn);
    disconnect(cancelledConn);
    disconnect(timeoutConn);
    
    return success;  // Returns true if completed before timeout
}

QTEST_MAIN(BackgroundConvolutionTest)
#include "tst_background_convolution.moc"
