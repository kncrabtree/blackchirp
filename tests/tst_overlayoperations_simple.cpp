#include <QtTest>
#include <QVector>

// Core operation interface classes
#include <src/gui/overlay/overlaytypespecificwidget.h>
#include <src/data/processing/overlayprocessmanager.h>
#include <src/data/storage/settingsstorage.h>

/**
 * @brief Simplified test suite for overlay operation declaration interface
 * 
 * This test focuses on the core operation capability system without heavy 
 * dependencies like BlackchirpCSV, ExperimentViewWidget, etc.
 */
class SimpleOverlayOperationsTest : public QObject
{
    Q_OBJECT

public:
    SimpleOverlayOperationsTest() = default;
    ~SimpleOverlayOperationsTest() = default;

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // OperationCapability Structure Tests
    void testOperationCapabilityConstruction();
    void testOperationCapabilityTypes();
    void testOperationCapabilityMetadata();
    void testOperationCapabilityDefaults();
    
    // Operation Type Tests
    void testAllOperationTypes();
    void testOperationTypeConsistency();
    void testOperationPriorities();
};

void SimpleOverlayOperationsTest::initTestCase()
{
    qDebug() << "SimpleOverlayOperations test environment initialized";
}

void SimpleOverlayOperationsTest::cleanupTestCase()
{
    // Cleanup handled automatically
}

void SimpleOverlayOperationsTest::testOperationCapabilityConstruction()
{
    // Test default constructor
    OperationCapability defaultCap(OperationCapability::Creation);
    QCOMPARE(defaultCap.type, OperationCapability::Creation);
    QVERIFY(!defaultCap.isExpensive);
    QCOMPARE(defaultCap.estimatedDurationMs, 0);
    QVERIFY(defaultCap.description.isEmpty());
    QCOMPARE(defaultCap.priority, OverlayProcessManager::Priority::Normal);
    
    // Test full constructor
    OperationCapability fullCap(
        OperationCapability::Convolution,
        true,  // expensive
        5000,  // duration
        "Test convolution operation",
        OverlayProcessManager::Priority::High
    );
    QCOMPARE(fullCap.type, OperationCapability::Convolution);
    QVERIFY(fullCap.isExpensive);
    QCOMPARE(fullCap.estimatedDurationMs, 5000);
    QCOMPARE(fullCap.description, QString("Test convolution operation"));
    QCOMPARE(fullCap.priority, OverlayProcessManager::Priority::High);
}

void SimpleOverlayOperationsTest::testOperationCapabilityTypes()
{
    // Test all operation types
    QVector<OperationCapability::Type> allTypes = {
        OperationCapability::Creation,
        OperationCapability::Convolution,
        OperationCapability::Validation,
        OperationCapability::PreviewUpdate
    };
    
    for (auto type : allTypes) {
        OperationCapability cap(type);
        QCOMPARE(cap.type, type);
    }
    
    // Verify type enum values are distinct
    QVERIFY(OperationCapability::Creation != OperationCapability::Convolution);
    QVERIFY(OperationCapability::Validation != OperationCapability::PreviewUpdate);
    QVERIFY(OperationCapability::Creation != OperationCapability::PreviewUpdate);
    QVERIFY(OperationCapability::Convolution != OperationCapability::Validation);
}

void SimpleOverlayOperationsTest::testOperationCapabilityMetadata()
{
    // Test metadata consistency for expensive operations
    OperationCapability expensiveCap(
        OperationCapability::Convolution,
        true,   // expensive
        10000,  // 10 seconds
        "Expensive convolution",
        OverlayProcessManager::Priority::High
    );
    
    // Expensive operations should have longer durations and appropriate priorities
    QVERIFY(expensiveCap.isExpensive);
    QVERIFY(expensiveCap.estimatedDurationMs > 1000);  // > 1 second
    QVERIFY(expensiveCap.priority != OverlayProcessManager::Priority::Low);
    
    // Test fast operation metadata
    OperationCapability fastCap(
        OperationCapability::Validation,
        false,  // not expensive
        50,     // 50ms
        "Fast validation",
        OverlayProcessManager::Priority::Normal
    );
    
    // Fast operations should have shorter durations
    QVERIFY(!fastCap.isExpensive);
    QVERIFY(fastCap.estimatedDurationMs < 1000);  // < 1 second
}

void SimpleOverlayOperationsTest::testOperationCapabilityDefaults()
{
    // Test that defaults are reasonable
    OperationCapability cap(OperationCapability::Creation);
    
    QVERIFY(!cap.isExpensive);  // Default should not be expensive
    QVERIFY(cap.estimatedDurationMs >= 0);  // Duration should be non-negative
    QCOMPARE(cap.priority, OverlayProcessManager::Priority::Normal);  // Default priority
    
    // Test that description can be empty (optional)
    QVERIFY(cap.description.isEmpty());
}

void SimpleOverlayOperationsTest::testAllOperationTypes()
{
    // Test creation of capabilities for all operation types
    QVector<OperationCapability::Type> allTypes = {
        OperationCapability::Creation,
        OperationCapability::Convolution,
        OperationCapability::Validation,
        OperationCapability::PreviewUpdate
    };
    
    for (auto type : allTypes) {
        OperationCapability cap(type);
        QCOMPARE(cap.type, type);
        QVERIFY(cap.estimatedDurationMs >= 0);
        
        // Test that we can create detailed capabilities for each type
        OperationCapability detailedCap(
            type,
            type == OperationCapability::Convolution || type == OperationCapability::PreviewUpdate,  // expensive
            type == OperationCapability::Convolution ? 5000 : 100,  // duration
            QString("Test %1 operation").arg(static_cast<int>(type)),
            OverlayProcessManager::Priority::Normal
        );
        
        QCOMPARE(detailedCap.type, type);
        QVERIFY(!detailedCap.description.isEmpty());
    }
}

void SimpleOverlayOperationsTest::testOperationTypeConsistency()
{
    // Test logical consistency in operation type properties
    
    // Creation operations should typically be fast
    OperationCapability createCap(OperationCapability::Creation, false, 100, "Create overlay");
    QVERIFY(!createCap.isExpensive);
    QVERIFY(createCap.estimatedDurationMs < 1000);
    
    // Convolution operations should typically be expensive
    OperationCapability convCap(OperationCapability::Convolution, true, 5000, "Apply convolution");
    QVERIFY(convCap.isExpensive);
    QVERIFY(convCap.estimatedDurationMs > 1000);
    
    // Validation operations should typically be fast
    OperationCapability validateCap(OperationCapability::Validation, false, 50, "Validate data");
    QVERIFY(!validateCap.isExpensive);
    QVERIFY(validateCap.estimatedDurationMs < 1000);
    
    // Preview update operations can be expensive
    OperationCapability previewCap(OperationCapability::PreviewUpdate, true, 3000, "Update preview");
    QVERIFY(previewCap.isExpensive);
    QVERIFY(previewCap.estimatedDurationMs > 1000);
}

void SimpleOverlayOperationsTest::testOperationPriorities()
{
    // Test all priority levels
    QVector<OverlayProcessManager::Priority> allPriorities = {
        OverlayProcessManager::Priority::Low,
        OverlayProcessManager::Priority::Normal,
        OverlayProcessManager::Priority::High
    };
    
    for (auto priority : allPriorities) {
        OperationCapability cap(
            OperationCapability::Creation,
            false,
            100,
            "Test operation",
            priority
        );
        QCOMPARE(cap.priority, priority);
    }
    
    // Test that expensive operations typically have higher priorities
    OperationCapability expensiveCap(
        OperationCapability::Convolution,
        true,  // expensive
        5000,
        "Expensive operation",
        OverlayProcessManager::Priority::High
    );
    
    QVERIFY(expensiveCap.isExpensive);
    QCOMPARE(expensiveCap.priority, OverlayProcessManager::Priority::High);
}

QTEST_MAIN(SimpleOverlayOperationsTest)
#include "tst_overlayoperations_simple.moc"