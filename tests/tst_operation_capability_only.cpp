#include <QtTest>
#include <QVector>

// Only include the minimal header that defines OperationCapability
#include <src/gui/overlay/overlaytypespecificwidget.h>

/**
 * @brief Minimal test suite for OperationCapability struct only
 * 
 * This test bypasses all heavy dependencies and focuses solely on testing
 * the OperationCapability structure and enum values that drive the operation
 * declaration interface system.
 */
class OperationCapabilityOnlyTest : public QObject
{
    Q_OBJECT

public:
    OperationCapabilityOnlyTest() = default;
    ~OperationCapabilityOnlyTest() = default;

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // Core OperationCapability struct tests
    void testOperationCapabilityConstruction();
    void testOperationCapabilityTypes();
    void testOperationCapabilityDefaults();
    void testOperationCapabilityComparison();
    
    // Operation type enum tests
    void testOperationTypeEnumValues();
    void testOperationTypeDistinctness();
    
    // Priority enum tests
    void testPriorityEnumValues();
    void testPriorityOrdering();
    
    // Metadata validation tests
    void testMetadataConsistency();
    void testDurationEstimates();
    void testDescriptionHandling();
};

void OperationCapabilityOnlyTest::initTestCase()
{
    qDebug() << "OperationCapabilityOnly test environment initialized";
    qDebug() << "Testing core operation capability structure without dependencies";
}

void OperationCapabilityOnlyTest::cleanupTestCase()
{
    // No cleanup needed for minimal test
}

void OperationCapabilityOnlyTest::testOperationCapabilityConstruction()
{
    // Test minimal constructor
    OperationCapability minimalCap(OperationCapability::Creation);
    QCOMPARE(minimalCap.type, OperationCapability::Creation);
    QVERIFY(!minimalCap.isExpensive);
    QCOMPARE(minimalCap.estimatedDurationMs, 0);
    QVERIFY(minimalCap.description.isEmpty());
    QCOMPARE(minimalCap.priority, OverlayProcessManager::Priority::Normal);
    
    // Test full constructor
    OperationCapability fullCap(
        OperationCapability::Convolution,
        true,  // expensive
        2500,  // duration
        "Test convolution operation",
        OverlayProcessManager::Priority::High
    );
    QCOMPARE(fullCap.type, OperationCapability::Convolution);
    QVERIFY(fullCap.isExpensive);
    QCOMPARE(fullCap.estimatedDurationMs, 2500);
    QCOMPARE(fullCap.description, QString("Test convolution operation"));
    QCOMPARE(fullCap.priority, OverlayProcessManager::Priority::High);
}

void OperationCapabilityOnlyTest::testOperationCapabilityTypes()
{
    // Test that we can create capabilities for all operation types
    QVector<OperationCapability::Type> allTypes = {
        OperationCapability::Creation,
        OperationCapability::Convolution,
        OperationCapability::Validation,
        OperationCapability::PreviewUpdate
    };
    
    for (auto type : allTypes) {
        OperationCapability cap(type);
        QCOMPARE(cap.type, type);
        
        // All operations should have valid default values
        QVERIFY(cap.estimatedDurationMs >= 0);
        QVERIFY(static_cast<int>(cap.priority) >= 0);
        QVERIFY(static_cast<int>(cap.priority) <= 3);  // Valid priority range
    }
}

void OperationCapabilityOnlyTest::testOperationCapabilityDefaults()
{
    // Test default values for each operation type
    OperationCapability createCap(OperationCapability::Creation);
    QVERIFY(!createCap.isExpensive);  // Creation should default to not expensive
    QCOMPARE(createCap.estimatedDurationMs, 0);  // Default duration
    QCOMPARE(createCap.priority, OverlayProcessManager::Priority::Normal);  // Default priority
    QVERIFY(createCap.description.isEmpty());  // Default description
    
    OperationCapability convCap(OperationCapability::Convolution);
    QVERIFY(!convCap.isExpensive);  // Default should not be expensive
    QCOMPARE(convCap.estimatedDurationMs, 0);  // Default duration
    QCOMPARE(convCap.priority, OverlayProcessManager::Priority::Normal);  // Default priority
}

void OperationCapabilityOnlyTest::testOperationCapabilityComparison()
{
    // Test that capabilities can be compared
    OperationCapability cap1(OperationCapability::Creation);
    OperationCapability cap2(OperationCapability::Creation);
    OperationCapability cap3(OperationCapability::Convolution);
    
    // Same type should have same type field
    QCOMPARE(cap1.type, cap2.type);
    QVERIFY(cap1.type != cap3.type);
    
    // Test that we can distinguish capabilities by type
    QVERIFY(cap1.type == OperationCapability::Creation);
    QVERIFY(cap3.type == OperationCapability::Convolution);
}

void OperationCapabilityOnlyTest::testOperationTypeEnumValues()
{
    // Test that operation type enum has expected values
    QVERIFY(OperationCapability::Creation != OperationCapability::Convolution);
    QVERIFY(OperationCapability::Creation != OperationCapability::Validation);
    QVERIFY(OperationCapability::Creation != OperationCapability::PreviewUpdate);
    
    QVERIFY(OperationCapability::Convolution != OperationCapability::Validation);
    QVERIFY(OperationCapability::Convolution != OperationCapability::PreviewUpdate);
    
    QVERIFY(OperationCapability::Validation != OperationCapability::PreviewUpdate);
    
    // Test that enum values can be used in switch statements (compile-time check)
    auto testSwitch = [](OperationCapability::Type type) -> bool {
        switch (type) {
        case OperationCapability::Creation:
        case OperationCapability::Convolution:
        case OperationCapability::Validation:
        case OperationCapability::PreviewUpdate:
            return true;
        }
        return false;
    };
    
    QVERIFY(testSwitch(OperationCapability::Creation));
    QVERIFY(testSwitch(OperationCapability::Convolution));
    QVERIFY(testSwitch(OperationCapability::Validation));
    QVERIFY(testSwitch(OperationCapability::PreviewUpdate));
}

void OperationCapabilityOnlyTest::testOperationTypeDistinctness()
{
    // Ensure all operation types are distinct by storing in a set
    QSet<OperationCapability::Type> typeSet;
    typeSet.insert(OperationCapability::Creation);
    typeSet.insert(OperationCapability::Convolution);
    typeSet.insert(OperationCapability::Validation);
    typeSet.insert(OperationCapability::PreviewUpdate);
    
    // Should have 4 distinct types
    QCOMPARE(typeSet.size(), 4);
}

void OperationCapabilityOnlyTest::testPriorityEnumValues()
{
    // Test priority enum values
    QVector<OverlayProcessManager::Priority> priorities = {
        OverlayProcessManager::Priority::Low,
        OverlayProcessManager::Priority::Normal,
        OverlayProcessManager::Priority::High
    };
    
    for (auto priority : priorities) {
        OperationCapability cap(OperationCapability::Creation, false, 100, "Test", priority);
        QCOMPARE(cap.priority, priority);
    }
    
    // Test that priorities are distinct
    QVERIFY(OverlayProcessManager::Priority::Low != OverlayProcessManager::Priority::Normal);
    QVERIFY(OverlayProcessManager::Priority::Normal != OverlayProcessManager::Priority::High);
    QVERIFY(OverlayProcessManager::Priority::Low != OverlayProcessManager::Priority::High);
}

void OperationCapabilityOnlyTest::testPriorityOrdering()
{
    // Test that priority values have a logical ordering
    int lowValue = static_cast<int>(OverlayProcessManager::Priority::Low);
    int normalValue = static_cast<int>(OverlayProcessManager::Priority::Normal);
    int highValue = static_cast<int>(OverlayProcessManager::Priority::High);
    
    QVERIFY(lowValue < normalValue);
    QVERIFY(normalValue < highValue);
    QVERIFY(lowValue < highValue);
    
    // Test that we can compare priorities
    QVERIFY(OverlayProcessManager::Priority::Low < OverlayProcessManager::Priority::Normal);
    QVERIFY(OverlayProcessManager::Priority::Normal < OverlayProcessManager::Priority::High);
}

void OperationCapabilityOnlyTest::testMetadataConsistency()
{
    // Test that expensive operations can have high priorities
    OperationCapability expensiveCap(
        OperationCapability::Convolution,
        true,  // expensive
        5000,  // 5 seconds
        "Expensive convolution",
        OverlayProcessManager::Priority::High
    );
    
    QVERIFY(expensiveCap.isExpensive);
    QVERIFY(expensiveCap.estimatedDurationMs > 1000);
    QCOMPARE(expensiveCap.priority, OverlayProcessManager::Priority::High);
    QVERIFY(!expensiveCap.description.isEmpty());
    
    // Test that fast operations work with normal priority
    OperationCapability fastCap(
        OperationCapability::Validation,
        false,  // not expensive
        50,     // 50ms
        "Fast validation",
        OverlayProcessManager::Priority::Normal
    );
    
    QVERIFY(!fastCap.isExpensive);
    QVERIFY(fastCap.estimatedDurationMs < 1000);
    QCOMPARE(fastCap.priority, OverlayProcessManager::Priority::Normal);
}

void OperationCapabilityOnlyTest::testDurationEstimates()
{
    // Test various duration estimates
    QVector<int> durations = { 0, 50, 100, 500, 1000, 2500, 5000, 10000 };
    
    for (int duration : durations) {
        OperationCapability cap(
            OperationCapability::Creation,
            duration > 1000,  // expensive if > 1 second
            duration,
            QString("Operation taking %1ms").arg(duration),
            OverlayProcessManager::Priority::Normal
        );
        
        QCOMPARE(cap.estimatedDurationMs, duration);
        QVERIFY(cap.estimatedDurationMs >= 0);
        
        // Consistency check: expensive flag should match duration
        if (duration > 1000) {
            QVERIFY(cap.isExpensive);
        } else {
            QVERIFY(!cap.isExpensive);
        }
    }
}

void OperationCapabilityOnlyTest::testDescriptionHandling()
{
    // Test empty description
    OperationCapability emptyCap(OperationCapability::Creation);
    QVERIFY(emptyCap.description.isEmpty());
    
    // Test non-empty description
    OperationCapability namedCap(
        OperationCapability::Convolution,
        true,
        2000,
        "Apply Lorentzian convolution to catalog data",
        OverlayProcessManager::Priority::High
    );
    QVERIFY(!namedCap.description.isEmpty());
    QCOMPARE(namedCap.description, QString("Apply Lorentzian convolution to catalog data"));
    
    // Test description with special characters
    OperationCapability specialCap(
        OperationCapability::Validation,
        false,
        100,
        "Validate file: /path/to/file.cat (μs precision)",
        OverlayProcessManager::Priority::Normal
    );
    QVERIFY(!specialCap.description.isEmpty());
    QVERIFY(specialCap.description.contains("μs"));
    QVERIFY(specialCap.description.contains("/path/to/file.cat"));
}

QTEST_MAIN(OperationCapabilityOnlyTest)
#include "tst_operation_capability_only.moc"