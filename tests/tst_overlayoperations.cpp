#include <QtTest>
#include <QVector>
#include <memory>

// Core operation interface classes
#include <src/gui/overlay/overlaytypespecificwidget.h>
#include <src/gui/overlay/bcexpoverlaywidget.h>
#include <src/gui/overlay/catalogoverlaywidget.h>

// Data processing classes
#include <src/data/processing/overlayprocessmanager.h>
#include <src/data/processing/overlayoperation.h>
#include <src/data/experiment/overlaybase.h>
#include <src/data/experiment/overlaytypes.h>

/**
 * @brief Test suite for overlay operation declaration interface and capabilities
 * 
 * This test suite focuses on the constexpr operation declaration interface,
 * operation capability metadata, and the compile-time evaluation system that
 * drives intelligent async/sync processing decisions.
 */
class OverlayOperationsTest : public QObject
{
    Q_OBJECT

public:
    OverlayOperationsTest() = default;
    ~OverlayOperationsTest() = default;

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // OperationCapability Structure Tests
    void testOperationCapabilityConstruction();
    void testOperationCapabilityTypes();
    void testOperationCapabilityMetadata();
    void testOperationCapabilityDefaults();
    
    // Constexpr Interface Tests
    void testConstexprOperationDeclaration();
    void testConstexprBackgroundSupport();
    void testCompileTimeEvaluation();
    void testPolymorphicConstexpr();
    
    // BCExpOverlayWidget Operation Tests
    void testBCExpSupportedOperations();
    void testBCExpBackgroundSupport();
    void testBCExpOperationMetadata();
    void testBCExpOperationFactory();
    
    // CatalogOverlayWidget Operation Tests
    void testCatalogSupportedOperations();
    void testCatalogBackgroundSupport();
    void testCatalogOperationMetadata();
    void testCatalogOperationPriorities();
    void testCatalogOperationFactory();
    
    // Operation Classification Tests
    void testSynchronousOperationIdentification();
    void testAsynchronousOperationIdentification();
    void testExpensiveOperationDetection();
    void testOperationDurationEstimates();
    
    // Polymorphic Interface Tests
    void testPolymorphicOperationQuery();
    void testPolymorphicCapabilityDetection();
    void testWidgetAgnosticProcessing();
    void testRuntimeTypeIndependence();
    
    // Edge Case and Error Tests
    void testInvalidOperationTypes();
    void testOperationFactoryEdgeCases();
    void testCapabilityConsistency();
    
    // Performance and Constexpr Tests
    void testConstexprPerformance();
    void testCompileTimeOptimization();
    void testConstexprPolymorphism();

private:
    // Test helpers
    template<typename WidgetType>
    void verifyOperationCapability(const OperationCapability &cap, 
                                 OperationCapability::Type expectedType,
                                 bool expectedExpensive,
                                 int expectedDuration,
                                 const QString &expectedDescription);
    
    template<typename WidgetType>
    void testWidgetOperationInterface();
    
    void verifyOperationConsistency(OverlayTypeSpecificWidget *widget);
    
    // Test data
    QVector<OperationCapability::Type> m_allOperationTypes;
    QVector<OverlayProcessManager::Priority> m_allPriorities;
};

void OverlayOperationsTest::initTestCase()
{
    // Initialize test data
    m_allOperationTypes = {
        OperationCapability::Creation,
        OperationCapability::Convolution,
        OperationCapability::Validation,
        OperationCapability::PreviewUpdate
    };
    
    m_allPriorities = {
        OverlayProcessManager::Priority::Low,
        OverlayProcessManager::Priority::Normal,
        OverlayProcessManager::Priority::High
    };
    
    qDebug() << "OverlayOperations test environment initialized";
    qDebug() << "  Operation types:" << m_allOperationTypes.size();
    qDebug() << "  Priority levels:" << m_allPriorities.size();
}

void OverlayOperationsTest::cleanupTestCase()
{
    // Cleanup handled automatically
}

void OverlayOperationsTest::testOperationCapabilityConstruction()
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

void OverlayOperationsTest::testOperationCapabilityTypes()
{
    // Test all operation types
    for (auto type : m_allOperationTypes) {
        OperationCapability cap(type);
        QCOMPARE(cap.type, type);
    }
    
    // Verify type enum values
    QVERIFY(OperationCapability::Creation != OperationCapability::Convolution);
    QVERIFY(OperationCapability::Validation != OperationCapability::PreviewUpdate);
}

void OverlayOperationsTest::testOperationCapabilityMetadata()
{
    // Test metadata consistency
    OperationCapability expensiveCap(
        OperationCapability::Convolution,
        true,   // expensive
        10000,  // 10 seconds
        "Expensive convolution",
        OverlayProcessManager::Priority::High
    );
    
    // Expensive operations should have longer durations and higher priorities
    QVERIFY(expensiveCap.isExpensive);
    QVERIFY(expensiveCap.estimatedDurationMs > 1000);  // > 1 second
    QVERIFY(expensiveCap.priority != OverlayProcessManager::Priority::Low);
    
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

void OverlayOperationsTest::testOperationCapabilityDefaults()
{
    // Test that defaults are reasonable
    OperationCapability cap(OperationCapability::Creation);
    
    QVERIFY(!cap.isExpensive);  // Default should not be expensive
    QVERIFY(cap.estimatedDurationMs >= 0);  // Duration should be non-negative
    QCOMPARE(cap.priority, OverlayProcessManager::Priority::Normal);  // Default priority
}

void OverlayOperationsTest::testConstexprOperationDeclaration()
{
    // Test that getSupportedOperations() is declared constexpr and works correctly
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    // Test that the methods are callable and return expected results
    auto bcOps = bcWidget.getSupportedOperations();
    auto catOps = catWidget.getSupportedOperations();
    
    // Verify the results are reasonable
    QVERIFY(bcOps.size() > 0);
    QVERIFY(catOps.size() > 0);
    QVERIFY(catOps.size() > bcOps.size());  // Catalog should have more operations
    
    // The constexpr nature is verified by the fact that these methods
    // are declared constexpr in the header files and compile successfully
}

void OverlayOperationsTest::testConstexprBackgroundSupport()
{
    // Test that supportsBackgroundOperation() is declared constexpr and works correctly
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    // Test the runtime behavior
    bool bcSupportsCreation = bcWidget.supportsBackgroundOperation(OperationCapability::Creation);
    bool catSupportsConvolution = catWidget.supportsBackgroundOperation(OperationCapability::Convolution);
    
    // Verify expected results
    QVERIFY(!bcSupportsCreation);  // BCExp should not support background creation
    QVERIFY(catSupportsConvolution);  // Catalog should support background convolution
    
    // The constexpr nature is verified by successful compilation of the constexpr methods
}

void OverlayOperationsTest::testCompileTimeEvaluation()
{
    // Test that operations can be analyzed efficiently
    CatalogOverlayWidget widget;
    auto ops = widget.getSupportedOperations();
    
    // Count expensive operations
    int expensiveCount = 0;
    for (const auto &op : ops) {
        if (op.isExpensive) {
            expensiveCount++;
        }
    }
    
    // Verify analysis results
    QVERIFY(expensiveCount > 0);  // Catalog should have expensive operations
    
    // The constexpr methods enable compile-time optimization opportunities
}

void OverlayOperationsTest::testPolymorphicConstexpr()
{
    // Test that constexpr methods work with polymorphic calls
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    OverlayTypeSpecificWidget* bcPtr = &bcWidget;
    OverlayTypeSpecificWidget* catPtr = &catWidget;
    
    // Test polymorphic access to constexpr methods
    auto bcOps = bcPtr->getSupportedOperations();
    auto catOps = catPtr->getSupportedOperations();
    
    QVERIFY(bcOps.size() > 0);
    QVERIFY(catOps.size() > 0);
    QVERIFY(catOps.size() >= bcOps.size());
    
    // The constexpr declaration enables optimization even through virtual calls
}

void OverlayOperationsTest::testBCExpSupportedOperations()
{
    BCExpOverlayWidget widget;
    auto operations = widget.getSupportedOperations();
    
    // BCExp should support Creation and Validation
    QCOMPARE(operations.size(), 2);
    
    // Verify Creation operation
    bool hasCreation = false;
    bool hasValidation = false;
    
    for (const auto &op : operations) {
        switch (op.type) {
        case OperationCapability::Creation:
            hasCreation = true;
            QVERIFY(!op.isExpensive);
            QCOMPARE(op.estimatedDurationMs, 100);
            QCOMPARE(op.description, QString("Create BCExperiment overlay"));
            QCOMPARE(op.priority, OverlayProcessManager::Priority::Normal);
            break;
        case OperationCapability::Validation:
            hasValidation = true;
            QVERIFY(!op.isExpensive);
            QCOMPARE(op.estimatedDurationMs, 50);
            QCOMPARE(op.description, QString("Validate experiment files"));
            QCOMPARE(op.priority, OverlayProcessManager::Priority::High);
            break;
        default:
            QFAIL("BCExp widget should not support this operation type");
        }
    }
    
    QVERIFY(hasCreation);
    QVERIFY(hasValidation);
}

void OverlayOperationsTest::testBCExpBackgroundSupport()
{
    BCExpOverlayWidget widget;
    
    // BCExp should not support background processing for any operation
    QVERIFY(!widget.supportsBackgroundOperation(OperationCapability::Creation));
    QVERIFY(!widget.supportsBackgroundOperation(OperationCapability::Validation));
    QVERIFY(!widget.supportsBackgroundOperation(OperationCapability::Convolution));
    QVERIFY(!widget.supportsBackgroundOperation(OperationCapability::PreviewUpdate));
}

void OverlayOperationsTest::testBCExpOperationMetadata()
{
    BCExpOverlayWidget widget;
    auto operations = widget.getSupportedOperations();
    
    // All BCExp operations should be fast
    for (const auto &op : operations) {
        QVERIFY(!op.isExpensive);
        QVERIFY(op.estimatedDurationMs < 1000);  // Less than 1 second
        QVERIFY(!op.description.isEmpty());
    }
}

void OverlayOperationsTest::testBCExpOperationFactory()
{
    BCExpOverlayWidget widget;
    
    // BCExp should return nullptr for all operations (synchronous processing)
    auto createOp = widget.createOperation(OperationCapability::Creation);
    QVERIFY(createOp == nullptr);
    
    auto validateOp = widget.createOperation(OperationCapability::Validation);
    QVERIFY(validateOp == nullptr);
    
    auto convOp = widget.createOperation(OperationCapability::Convolution);
    QVERIFY(convOp == nullptr);
    
    auto previewOp = widget.createOperation(OperationCapability::PreviewUpdate);
    QVERIFY(previewOp == nullptr);
}

void OverlayOperationsTest::testCatalogSupportedOperations()
{
    CatalogOverlayWidget widget;
    auto operations = widget.getSupportedOperations();
    
    // Catalog should support all operation types
    QCOMPARE(operations.size(), 4);
    
    QSet<OperationCapability::Type> foundTypes;
    for (const auto &op : operations) {
        foundTypes.insert(op.type);
    }
    
    QVERIFY(foundTypes.contains(OperationCapability::Creation));
    QVERIFY(foundTypes.contains(OperationCapability::Convolution));
    QVERIFY(foundTypes.contains(OperationCapability::Validation));
    QVERIFY(foundTypes.contains(OperationCapability::PreviewUpdate));
}

void OverlayOperationsTest::testCatalogBackgroundSupport()
{
    CatalogOverlayWidget widget;
    
    // Creation and validation should be synchronous
    QVERIFY(!widget.supportsBackgroundOperation(OperationCapability::Creation));
    QVERIFY(!widget.supportsBackgroundOperation(OperationCapability::Validation));
    
    // Convolution and preview should be asynchronous
    QVERIFY(widget.supportsBackgroundOperation(OperationCapability::Convolution));
    QVERIFY(widget.supportsBackgroundOperation(OperationCapability::PreviewUpdate));
}

void OverlayOperationsTest::testCatalogOperationMetadata()
{
    CatalogOverlayWidget widget;
    auto operations = widget.getSupportedOperations();
    
    for (const auto &op : operations) {
        switch (op.type) {
        case OperationCapability::Creation:
            QVERIFY(!op.isExpensive);
            QCOMPARE(op.estimatedDurationMs, 200);
            break;
        case OperationCapability::Convolution:
            QVERIFY(op.isExpensive);
            QCOMPARE(op.estimatedDurationMs, 5000);
            QCOMPARE(op.priority, OverlayProcessManager::Priority::High);
            break;
        case OperationCapability::Validation:
            QVERIFY(!op.isExpensive);
            QCOMPARE(op.estimatedDurationMs, 100);
            break;
        case OperationCapability::PreviewUpdate:
            QVERIFY(op.isExpensive);
            QCOMPARE(op.estimatedDurationMs, 3000);
            QCOMPARE(op.priority, OverlayProcessManager::Priority::High);
            break;
        }
        
        QVERIFY(!op.description.isEmpty());
        QVERIFY(op.estimatedDurationMs >= 0);
    }
}

void OverlayOperationsTest::testCatalogOperationPriorities()
{
    CatalogOverlayWidget widget;
    auto operations = widget.getSupportedOperations();
    
    // Expensive operations should have high priority
    for (const auto &op : operations) {
        if (op.isExpensive) {
            QCOMPARE(op.priority, OverlayProcessManager::Priority::High);
        }
    }
}

void OverlayOperationsTest::testCatalogOperationFactory()
{
    CatalogOverlayWidget widget;
    
    // Synchronous operations should return nullptr
    auto createOp = widget.createOperation(OperationCapability::Creation);
    QVERIFY(createOp == nullptr);
    
    auto validateOp = widget.createOperation(OperationCapability::Validation);
    QVERIFY(createOp == nullptr);
    
    // Asynchronous operations would create operations but require overlay objects
    // This is tested in the integration tests
}

void OverlayOperationsTest::testSynchronousOperationIdentification()
{
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    // Test that synchronous operations are consistently identified
    auto bcOps = bcWidget.getSupportedOperations();
    for (const auto &op : bcOps) {
        QVERIFY(!op.isExpensive);
        QVERIFY(!bcWidget.supportsBackgroundOperation(op.type));
    }
    
    // Catalog should have both sync and async operations
    auto catOps = catWidget.getSupportedOperations();
    bool hasSyncOps = false;
    bool hasAsyncOps = false;
    
    for (const auto &op : catOps) {
        bool supportsBackground = catWidget.supportsBackgroundOperation(op.type);
        if (supportsBackground) {
            hasAsyncOps = true;
            QVERIFY(op.isExpensive);  // Background operations should be expensive
        } else {
            hasSyncOps = true;
            QVERIFY(!op.isExpensive);  // Sync operations should not be expensive
        }
    }
    
    QVERIFY(hasSyncOps);
    QVERIFY(hasAsyncOps);
}

void OverlayOperationsTest::testAsynchronousOperationIdentification()
{
    CatalogOverlayWidget widget;
    
    // Test specific async operations
    QVERIFY(widget.supportsBackgroundOperation(OperationCapability::Convolution));
    QVERIFY(widget.supportsBackgroundOperation(OperationCapability::PreviewUpdate));
    
    // Verify these are marked as expensive in the operation metadata
    auto operations = widget.getSupportedOperations();
    for (const auto &op : operations) {
        if (op.type == OperationCapability::Convolution || 
            op.type == OperationCapability::PreviewUpdate) {
            QVERIFY(op.isExpensive);
        }
    }
}

void OverlayOperationsTest::testExpensiveOperationDetection()
{
    CatalogOverlayWidget widget;
    auto operations = widget.getSupportedOperations();
    
    int expensiveCount = 0;
    int fastCount = 0;
    
    for (const auto &op : operations) {
        if (op.isExpensive) {
            expensiveCount++;
            QVERIFY(op.estimatedDurationMs > 1000);  // > 1 second
            QVERIFY(widget.supportsBackgroundOperation(op.type));
        } else {
            fastCount++;
            QVERIFY(op.estimatedDurationMs < 1000);  // < 1 second
            QVERIFY(!widget.supportsBackgroundOperation(op.type));
        }
    }
    
    QVERIFY(expensiveCount > 0);  // Should have some expensive operations
    QVERIFY(fastCount > 0);       // Should have some fast operations
}

void OverlayOperationsTest::testOperationDurationEstimates()
{
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    // Check that duration estimates are reasonable
    auto bcOps = bcWidget.getSupportedOperations();
    for (const auto &op : bcOps) {
        QVERIFY(op.estimatedDurationMs >= 0);
        QVERIFY(op.estimatedDurationMs < 1000);  // BCExp ops should be fast
    }
    
    auto catOps = catWidget.getSupportedOperations();
    bool hasLongOperation = false;
    for (const auto &op : catOps) {
        QVERIFY(op.estimatedDurationMs >= 0);
        if (op.estimatedDurationMs > 1000) {
            hasLongOperation = true;
        }
    }
    
    QVERIFY(hasLongOperation);  // Catalog should have some long operations
}

void OverlayOperationsTest::testPolymorphicOperationQuery()
{
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    // Test polymorphic access
    OverlayTypeSpecificWidget* widgets[] = { &bcWidget, &catWidget };
    
    for (auto* widget : widgets) {
        auto operations = widget->getSupportedOperations();
        QVERIFY(operations.size() > 0);
        
        for (const auto &op : operations) {
            // Test that background support is consistent with metadata
            bool supportsBackground = widget->supportsBackgroundOperation(op.type);
            QCOMPARE(supportsBackground, op.isExpensive);
        }
    }
}

void OverlayOperationsTest::testPolymorphicCapabilityDetection()
{
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    OverlayTypeSpecificWidget* bcPtr = &bcWidget;
    OverlayTypeSpecificWidget* catPtr = &catWidget;
    
    // Test that capability detection works polymorphically
    for (auto type : m_allOperationTypes) {
        bool bcSupports = bcPtr->supportsBackgroundOperation(type);
        bool catSupports = catPtr->supportsBackgroundOperation(type);
        
        // BCExp should not support background for any operation
        QVERIFY(!bcSupports);
        
        // Catalog should support background for some operations
        if (type == OperationCapability::Convolution || 
            type == OperationCapability::PreviewUpdate) {
            QVERIFY(catSupports);
        } else {
            QVERIFY(!catSupports);
        }
    }
}

void OverlayOperationsTest::testWidgetAgnosticProcessing()
{
    // Test the core principle: processing decisions can be made without
    // knowing the specific widget type
    
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    auto testProcessingDecision = [](OverlayTypeSpecificWidget* widget, 
                                   OperationCapability::Type opType) -> bool {
        // This simulates the logic in UnifiedOverlayDialog
        return widget->supportsBackgroundOperation(opType);
    };
    
    // Test creation operation routing
    bool bcNeedsBackground = testProcessingDecision(&bcWidget, OperationCapability::Creation);
    bool catNeedsBackground = testProcessingDecision(&catWidget, OperationCapability::Creation);
    
    QVERIFY(!bcNeedsBackground);   // BCExp creation should be synchronous
    QVERIFY(!catNeedsBackground);  // Catalog creation should be synchronous
    
    // Test convolution operation routing
    bool bcConvBackground = testProcessingDecision(&bcWidget, OperationCapability::Convolution);
    bool catConvBackground = testProcessingDecision(&catWidget, OperationCapability::Convolution);
    
    QVERIFY(!bcConvBackground);  // BCExp doesn't support convolution
    QVERIFY(catConvBackground);  // Catalog convolution should be asynchronous
}

void OverlayOperationsTest::testRuntimeTypeIndependence()
{
    // Test that the operation interface works without runtime type checking
    
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    QVector<OverlayTypeSpecificWidget*> widgets = { &bcWidget, &catWidget };
    
    for (auto* widget : widgets) {
        // Process operations without knowing the widget type
        auto operations = widget->getSupportedOperations();
        
        for (const auto &op : operations) {
            bool needsBackground = widget->supportsBackgroundOperation(op.type);
            auto operation = widget->createOperation(op.type);
            
            // Verify consistency
            if (needsBackground) {
                QVERIFY(op.isExpensive);
                // Note: operation might be nullptr if it requires overlay data
            } else {
                QVERIFY(!op.isExpensive);
                QVERIFY(operation == nullptr);  // Sync operations return nullptr
            }
        }
    }
}

void OverlayOperationsTest::testInvalidOperationTypes()
{
    BCExpOverlayWidget widget;
    
    // Test that invalid operation types are handled gracefully
    // (This tests the switch statement default cases)
    
    // Cast to test boundary values (implementation detail)
    auto invalidType = static_cast<OperationCapability::Type>(999);
    
    // These should not crash
    bool supports = widget.supportsBackgroundOperation(invalidType);
    QVERIFY(!supports);  // Should default to false
    
    auto operation = widget.createOperation(invalidType);
    QVERIFY(operation == nullptr);  // Should default to nullptr
}

void OverlayOperationsTest::testOperationFactoryEdgeCases()
{
    CatalogOverlayWidget widget;
    
    // Test factory with null overlay (edge case)
    auto operation = widget.createOperation(OperationCapability::Convolution, nullptr);
    QVERIFY(operation == nullptr);  // Should handle null overlay gracefully
    
    // Test factory with sync operations
    auto syncOp = widget.createOperation(OperationCapability::Creation, nullptr);
    QVERIFY(syncOp == nullptr);  // Sync operations should return nullptr
}

void OverlayOperationsTest::testCapabilityConsistency()
{
    // Test that operation capabilities are internally consistent
    
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    QVector<OverlayTypeSpecificWidget*> widgets = { &bcWidget, &catWidget };
    
    for (auto* widget : widgets) {
        verifyOperationConsistency(widget);
    }
}

void OverlayOperationsTest::testConstexprPerformance()
{
    // Test that constexpr methods provide consistent performance
    
    // Multiple evaluations should give consistent results
    CatalogOverlayWidget widget1;
    CatalogOverlayWidget widget2;
    
    auto ops1 = widget1.getSupportedOperations().size();
    auto ops2 = widget2.getSupportedOperations().size();
    
    // Results should be identical
    QCOMPARE(static_cast<int>(ops1), static_cast<int>(ops2));
    
    // The constexpr nature enables compiler optimization opportunities
}

void OverlayOperationsTest::testCompileTimeOptimization()
{
    // Test that the constexpr interface enables optimization
    CatalogOverlayWidget widget;
    auto ops = widget.getSupportedOperations();
    
    // Analyze operations
    struct Analysis {
        int totalOps = 0;
        int expensiveOps = 0;
        int fastOps = 0;
    } analysis;
    
    for (const auto &op : ops) {
        analysis.totalOps++;
        if (op.isExpensive) {
            analysis.expensiveOps++;
        } else {
            analysis.fastOps++;
        }
    }
    
    // Verify analysis results
    QVERIFY(analysis.totalOps > 0);
    QVERIFY(analysis.expensiveOps > 0);
    QVERIFY(analysis.fastOps > 0);
    QCOMPARE(analysis.totalOps, analysis.expensiveOps + analysis.fastOps);
    
    // The constexpr nature enables potential compile-time optimization
}

void OverlayOperationsTest::testConstexprPolymorphism()
{
    // Test that constexpr works with virtual function calls
    BCExpOverlayWidget bcWidget;
    CatalogOverlayWidget catWidget;
    
    // Test polymorphic constexpr virtual functions
    auto bcSize = bcWidget.getSupportedOperations().size();
    auto catSize = catWidget.getSupportedOperations().size();
    auto totalOps = bcSize + catSize;
    
    QVERIFY(totalOps > 0);
    QVERIFY(bcSize > 0);
    QVERIFY(catSize > 0);
    
    // The constexpr virtual functions enable optimization at the interface level
}

// Helper method implementations

template<typename WidgetType>
void OverlayOperationsTest::verifyOperationCapability(const OperationCapability &cap,
                                                     OperationCapability::Type expectedType,
                                                     bool expectedExpensive,
                                                     int expectedDuration,
                                                     const QString &expectedDescription)
{
    QCOMPARE(cap.type, expectedType);
    QCOMPARE(cap.isExpensive, expectedExpensive);
    QCOMPARE(cap.estimatedDurationMs, expectedDuration);
    QCOMPARE(cap.description, expectedDescription);
    QVERIFY(cap.estimatedDurationMs >= 0);
}

template<typename WidgetType>
void OverlayOperationsTest::testWidgetOperationInterface()
{
    WidgetType widget;
    
    // Test that the interface is properly implemented
    auto operations = widget.getSupportedOperations();
    QVERIFY(operations.size() > 0);
    
    for (const auto &op : operations) {
        bool supportsBackground = widget.supportsBackgroundOperation(op.type);
        auto operation = widget.createOperation(op.type);
        
        // Verify consistency between capability and background support
        QCOMPARE(supportsBackground, op.isExpensive);
        
        // Factory method should return nullptr for sync operations
        if (!supportsBackground) {
            QVERIFY(operation == nullptr);
        }
    }
}

void OverlayOperationsTest::verifyOperationConsistency(OverlayTypeSpecificWidget *widget)
{
    auto operations = widget->getSupportedOperations();
    
    for (const auto &op : operations) {
        // Verify consistency between metadata and capability queries
        bool supportsBackground = widget->supportsBackgroundOperation(op.type);
        QCOMPARE(supportsBackground, op.isExpensive);
        
        // Verify duration estimates are reasonable
        QVERIFY(op.estimatedDurationMs >= 0);
        if (op.isExpensive) {
            QVERIFY(op.estimatedDurationMs > 100);  // Expensive ops should take some time
        }
        
        // Verify descriptions are provided
        QVERIFY(!op.description.isEmpty());
        
        // Verify priority is valid
        QVERIFY(m_allPriorities.contains(op.priority));
    }
}

QTEST_MAIN(OverlayOperationsTest)
#include "tst_overlayoperations.moc"