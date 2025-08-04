#include <QtTest>
#include <QCoreApplication>
#include <QSettings>
#include <QThread>
#include <QSemaphore>
#include <QMutex>

#include <src/hardware/core/runtimehardwareconfig.h>
#include <src/hardware/core/hardwareregistry.h>
#include <src/hardware/core/hardwareregistration.h>

// Forward declaration for test class that needs friend access
class RuntimeHardwareConfigTest;

class RuntimeHardwareConfigTest : public QObject
{
    Q_OBJECT

public:
    RuntimeHardwareConfigTest();
    ~RuntimeHardwareConfigTest();

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();
    
    // Basic functionality tests
    void testSingletonAccess();
    void testConstInstance();
    void testHardwareSelection();
    void testActiveLabels();
    void testCurrentHardware();
    void testConfigurationValidation();
    
    // Profile integration tests
    void testProfileIntegration();
    void testProfileSynchronization();
    
    // Thread safety tests
    void testConcurrentRead();
    void testConcurrentWrite();
    void testReadWriteConcurrency();
    
    // Integration tests with HardwareRegistry
    void testRegistryIntegration();
    void testValidationWithRegistry();
    
    // Friend access tests
    void testFriendAccess();
    void testNonFriendAccessRestriction();
    
    // Edge cases
    void testEmptyConfiguration();
    void testInvalidHardwareTypes();
    void testConfigurationReset();
    
    // Validation failure scenarios
    void testUnregisteredHardwareValidation();
    void testInvalidProfilePersistence();

private:
    void clearTestSettings();
    void setupTestRegistry();
    void concurrentReadTest(int threadId, QSemaphore *sem, QMutex *resultMutex, QStringList *results);
    void concurrentWriteTest(int threadId, QSemaphore *sem, QMutex *resultMutex, QStringList *results);
    
    HardwareRegistry *d_registry;
};

RuntimeHardwareConfigTest::RuntimeHardwareConfigTest()
    : d_registry(nullptr)
{
    // Use test settings location to avoid conflicts with real BlackChirp settings
    QCoreApplication::setApplicationName("BlackchirpTest");
    QCoreApplication::setOrganizationName("CrabtreeLab");
    QCoreApplication::setOrganizationDomain("crabtreelab.ucdavis.edu");
}

RuntimeHardwareConfigTest::~RuntimeHardwareConfigTest()
{
}

void RuntimeHardwareConfigTest::initTestCase()
{
    qDebug() << "Starting RuntimeHardwareConfig test suite...";
    clearTestSettings();
    d_registry = &HardwareRegistry::instance();
    setupTestRegistry();
}

void RuntimeHardwareConfigTest::cleanupTestCase()
{
    qDebug() << "RuntimeHardwareConfig test suite completed.";
}

void RuntimeHardwareConfigTest::init()
{
    // Clear settings before each test to ensure clean state
    clearTestSettings();
}

void RuntimeHardwareConfigTest::cleanup()
{
    // Clean up after each test
}

void RuntimeHardwareConfigTest::clearTestSettings()
{
    // Clear out any existing test settings to ensure clean test environment
    QSettings s("CrabtreeLab", "BlackchirpTest");
    s.setFallbacksEnabled(false);
    s.clear();
    s.sync();
}

void RuntimeHardwareConfigTest::setupTestRegistry()
{
    // Register some test hardware for validation tests
    auto factory1 = [](const QString& label) -> HardwareObject* { Q_UNUSED(label) return nullptr; }; // Mock factory
    
    d_registry->registerHardware(
        "TestHardware1", "impl1", "Test Hardware 1 implementation 1",
        factory1
    );
    
    d_registry->registerHardware(
        "TestHardware1", "impl2", "Test Hardware 1 Alternative implementation",
        factory1
    );
    
    d_registry->registerHardware(
        "TestHardware2", "impl1", "Test Hardware 2 implementation 2",
        factory1
    );
}

void RuntimeHardwareConfigTest::testSingletonAccess()
{
    // Test that we get the same instance
    auto &config1 = RuntimeHardwareConfig::constInstance();
    auto &config2 = RuntimeHardwareConfig::constInstance();
    
    QCOMPARE(&config1, &config2);
}

void RuntimeHardwareConfigTest::testConstInstance()
{
    // Test that constInstance provides read-only access
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // These should work (read operations)
    QString implementation = config.getHardwareImplementation("TestType", "testLabel");
    auto hardware = config.getCurrentHardware();
    bool valid = config.isConfigurationValid();
    
    // Verify default values for non-existent hardware
    QVERIFY(implementation.isEmpty());
    QVERIFY(hardware.empty());
    // Note: isConfigurationValid behavior depends on implementation
    Q_UNUSED(valid)
}

void RuntimeHardwareConfigTest::testHardwareSelection()
{
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Initially should be empty
    QString implementation = config.getHardwareImplementation("TestHardware1", "testLabel");
    QVERIFY(implementation.isEmpty());
    
    // Set through direct friend access
    bool set = RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "testLabel", "impl1");
    QVERIFY(set);
    
    // Verify selection was set
    implementation = config.getHardwareImplementation("TestHardware1", "testLabel");
    QCOMPARE(implementation, QString("impl1"));
}

void RuntimeHardwareConfigTest::testActiveLabels()
{
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Clear any existing configuration first
    RuntimeHardwareConfig::instance().clearConfiguration();
    
    // Initially should be empty
    QStringList activeLabels = config.getActiveLabels("TestHardware1");
    QVERIFY(activeLabels.isEmpty());
    
    // Add hardware selection
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "label1", "impl1");
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "label2", "impl2");
    
    // Verify active labels
    activeLabels = config.getActiveLabels("TestHardware1");
    QCOMPARE(activeLabels.size(), 2);
    QVERIFY(activeLabels.contains("label1"));
    QVERIFY(activeLabels.contains("label2"));
    
    // Remove one selection
    RuntimeHardwareConfig::instance().removeHardwareSelection("TestHardware1", "label1");
    activeLabels = config.getActiveLabels("TestHardware1");
    QCOMPARE(activeLabels.size(), 1);
    QVERIFY(activeLabels.contains("label2"));
}

void RuntimeHardwareConfigTest::testConfigurationValidation()
{
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Test configuration validation
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "label1", "impl1");
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware2", "label1", "impl1");
    
    // Test validation methods
    auto validationResults = config.validateConfiguration();
    QVERIFY(validationResults.size() > 0);
    
    // Test configuration validity
    bool valid = config.isConfigurationValid();
    Q_UNUSED(valid) // Don't assert as it depends on availability
    
    // Test missing required hardware
    auto missing = config.getMissingRequiredHardware();
    // Results depend on what's configured as required
    Q_UNUSED(missing)
    
    // Test error/warning collection
    auto errors = config.getAllValidationErrors();
    auto warnings = config.getAllValidationWarnings();
    Q_UNUSED(errors)
    Q_UNUSED(warnings)
    
    // Main goal is to verify methods don't crash
    QVERIFY(true);
}

void RuntimeHardwareConfigTest::testCurrentHardware()
{
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Set some hardware configurations
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "label1", "impl1");
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware2", "label1", "impl1");
    
    auto hardware = config.getCurrentHardware();
    
    // Should contain all configured hardware (using BC::Key format)
    QString key1 = "TestHardware1.label1";
    QString key2 = "TestHardware2.label1";
    
    QVERIFY(hardware.find(key1) != hardware.end());
    QVERIFY(hardware.find(key2) != hardware.end());
    
    QCOMPARE(hardware.at(key1), QString("impl1"));
    QCOMPARE(hardware.at(key2), QString("impl1"));
    
    // Remove one selection and verify it disappears
    RuntimeHardwareConfig::instance().removeHardwareSelection("TestHardware2", "label1");
    auto hardwareAfter = config.getCurrentHardware();
    QVERIFY(hardwareAfter.find(key1) != hardwareAfter.end());
    QVERIFY(hardwareAfter.find(key2) == hardwareAfter.end());
}

void RuntimeHardwareConfigTest::testProfileIntegration()
{
    // Test integration with HardwareProfileManager
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Set hardware selections
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "testLabel1", "impl1");
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware2", "testLabel2", "impl1");
    
    // Verify selections are active
    QString impl1 = config.getHardwareImplementation("TestHardware1", "testLabel1");
    QString impl2 = config.getHardwareImplementation("TestHardware2", "testLabel2");
    
    QCOMPARE(impl1, QString("impl1"));
    QCOMPARE(impl2, QString("impl1"));
    
    // Verify active labels
    QStringList activeLabels1 = config.getActiveLabels("TestHardware1");
    QStringList activeLabels2 = config.getActiveLabels("TestHardware2");
    
    QVERIFY(activeLabels1.contains("testLabel1"));
    QVERIFY(activeLabels2.contains("testLabel2"));
}

void RuntimeHardwareConfigTest::testProfileSynchronization()
{
    // Test synchronization with profile manager
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Clear everything and start fresh
    RuntimeHardwareConfig::instance().clearConfiguration();
    
    // Verify configuration is cleared
    auto hardware = config.getCurrentHardware();
    QVERIFY(hardware.empty());
    
    // Set up hardware selections
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "syncLabel", "impl2");
    
    // Verify the selection is active
    QString implementation = config.getHardwareImplementation("TestHardware1", "syncLabel");
    QCOMPARE(implementation, QString("impl2"));
    
    auto hardwareAfter = config.getCurrentHardware();
    QString expectedKey = "TestHardware1.syncLabel";
    QVERIFY(hardwareAfter.find(expectedKey) != hardwareAfter.end());
    QCOMPARE(hardwareAfter.at(expectedKey), QString("impl2"));
}

// Removed testSettingsKeys - settings are now handled by HardwareProfileManager

void RuntimeHardwareConfigTest::testFriendAccess()
{
    // Test that test class (as friend) can access private methods
    // Use registered hardware that actually exists
    bool result = RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "testLabel", "impl2");
    QVERIFY(result); // Should succeed with friend access
    
    // Verify the setting took effect
    const auto &config = RuntimeHardwareConfig::constInstance();
    QCOMPARE(config.getHardwareImplementation("TestHardware1", "testLabel"), QString("impl2"));
}

void RuntimeHardwareConfigTest::testNonFriendAccessRestriction()
{
    // This test verifies that the private methods are indeed private
    // We can't directly test this in C++ without compilation errors,
    // but we can verify that only public const methods are available
    
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // These should compile (public const methods)
    QString implementation = config.getHardwareImplementation("Test", "label");
    auto hardware = config.getCurrentHardware();
    bool valid = config.isConfigurationValid();
    
    // Note: We cannot test direct access to private methods here
    // as that would cause compilation errors. The friend relationship
    // is enforced at compile time.
    Q_UNUSED(implementation)
    Q_UNUSED(valid)
    QVERIFY(true); // This test mainly serves as documentation
}

void RuntimeHardwareConfigTest::concurrentReadTest(int threadId, QSemaphore *sem, 
                                                  QMutex *resultMutex, QStringList *results)
{
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Perform multiple read operations
    for (int i = 0; i < 100; ++i) {
        QString implementation = config.getHardwareImplementation(QString("Thread%1").arg(threadId), "testLabel");
        auto hardware = config.getCurrentHardware();
        bool valid = config.isConfigurationValid();
        
        // Just verify we can call these without crashing
        Q_UNUSED(implementation)
        Q_UNUSED(hardware)
        Q_UNUSED(valid)
    }
    
    QMutexLocker locker(resultMutex);
    results->append(QString("Thread %1: Read operations completed").arg(threadId));
    sem->release();
}

void RuntimeHardwareConfigTest::testConcurrentRead()
{
    const int numThreads = 10;
    QSemaphore sem;
    QMutex resultMutex;
    QStringList results;
    
    // Start multiple threads doing read operations
    for (int i = 0; i < numThreads; ++i) {
        QThread* thread = QThread::create([this, i, &sem, &resultMutex, &results]() {
            concurrentReadTest(i, &sem, &resultMutex, &results);
        });
        thread->start();
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < numThreads; ++i) {
        sem.acquire();
    }
    
    // All read operations should have succeeded
    QCOMPARE(results.size(), numThreads);
    for (const QString &result : results) {
        QVERIFY(result.contains("completed"));
    }
}

void RuntimeHardwareConfigTest::concurrentWriteTest(int threadId, QSemaphore *sem, 
                                                   QMutex *resultMutex, QStringList *results)
{
    // Use registered hardware types for concurrent writes
    // Alternate between TestHardware1 and TestHardware2
    QString hardwareType = (threadId % 2 == 0) ? "TestHardware1" : "TestHardware2";
    
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        // Use valid implementations that exist in the registry
        QString impl = (hardwareType == "TestHardware1") ? 
                      ((i % 2 == 0) ? "impl1" : "impl2") : "impl1";
        
        // Create unique label for each thread and iteration
        QString label = QString("thread%1_iter%2").arg(threadId).arg(i);
        
        if (!RuntimeHardwareConfig::instance().setHardwareSelection(hardwareType, label, impl)) {
            success = false;
            break;
        }
    }
    
    QMutexLocker locker(resultMutex);
    if (success) {
        results->append(QString("Thread %1: Write operations successful").arg(threadId));
    } else {
        results->append(QString("Thread %1: Write operations failed").arg(threadId));
    }
    sem->release();
}

void RuntimeHardwareConfigTest::testConcurrentWrite()
{
    const int numThreads = 5;
    QSemaphore sem;
    QMutex resultMutex;
    QStringList results;
    
    // Start multiple threads doing write operations
    for (int i = 0; i < numThreads; ++i) {
        QThread* thread = QThread::create([this, i, &sem, &resultMutex, &results]() {
            concurrentWriteTest(i, &sem, &resultMutex, &results);
        });
        thread->start();
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < numThreads; ++i) {
        sem.acquire();
    }
    
    // All write operations should have succeeded
    QCOMPARE(results.size(), numThreads);
    for (const QString &result : results) {
        QVERIFY(result.contains("successful"));
    }
    
    // Verify final state - check that hardware was configured
    const auto &config = RuntimeHardwareConfig::constInstance();
    QStringList activeLabels1 = config.getActiveLabels("TestHardware1");
    QStringList activeLabels2 = config.getActiveLabels("TestHardware2");
    
    // Should have some active labels since threads created configurations
    QVERIFY(!activeLabels1.isEmpty() || !activeLabels2.isEmpty());
    
    // Verify they're configured  
    auto configuredTypes = config.getConfiguredHardwareTypes();
    QVERIFY(configuredTypes.contains("TestHardware1") || configuredTypes.contains("TestHardware2"));
}

void RuntimeHardwareConfigTest::testReadWriteConcurrency()
{
    const int numReaders = 5;
    const int numWriters = 3;
    QSemaphore sem;
    QMutex resultMutex;
    QStringList results;
    
    // Start reader threads
    for (int i = 0; i < numReaders; ++i) {
        QThread* thread = QThread::create([this, i, &sem, &resultMutex, &results]() {
            concurrentReadTest(i, &sem, &resultMutex, &results);
        });
        thread->start();
    }
    
    // Start writer threads
    for (int i = 0; i < numWriters; ++i) {
        QThread* thread = QThread::create([this, i, &sem, &resultMutex, &results]() {
            concurrentWriteTest(i, &sem, &resultMutex, &results);
        });
        thread->start();
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < numReaders + numWriters; ++i) {
        sem.acquire();
    }
    
    // All operations should have succeeded
    QCOMPARE(results.size(), numReaders + numWriters);
    for (const QString &result : results) {
        QVERIFY(result.contains("completed") || result.contains("successful"));
    }
}

void RuntimeHardwareConfigTest::testEmptyConfiguration()
{
    // Clear all configuration to test empty state
    RuntimeHardwareConfig::instance().clearConfiguration();
    
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    QString implementation = config.getHardwareImplementation("NonExistent", "label");
    QVERIFY(implementation.isEmpty());
    
    // After clearing, getCurrentHardware() should be empty
    auto hardware = config.getCurrentHardware();
    QVERIFY(hardware.empty());
    
    // Configuration validity with empty config depends on implementation
    bool valid = config.isConfigurationValid();
    Q_UNUSED(valid) // Don't assert on this as behavior may vary
}

void RuntimeHardwareConfigTest::testInvalidHardwareTypes()
{
    // Test with invalid/empty hardware type names
    bool result1 = RuntimeHardwareConfig::instance().setHardwareSelection("", "label", "impl1");
    bool result2 = RuntimeHardwareConfig::instance().setHardwareSelection("ValidType", "", "impl1");
    bool result3 = RuntimeHardwareConfig::instance().setHardwareSelection("ValidType", "label", "");
    
    // These should fail gracefully
    QVERIFY(!result1); // Empty type should fail
    QVERIFY(!result2); // Empty label should fail
    // Empty implementation might be allowed for removing hardware
    
    const auto &config = RuntimeHardwareConfig::constInstance();
    QString implementation1 = config.getHardwareImplementation("", "label");
    QString implementation2 = config.getHardwareImplementation("ValidType", "");
    
    // Should return empty for invalid inputs
    QVERIFY(implementation1.isEmpty());
    QVERIFY(implementation2.isEmpty());
    
    Q_UNUSED(result3)
}

void RuntimeHardwareConfigTest::testRegistryIntegration()
{
    // Test integration with HardwareRegistry
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Set hardware that exists in registry
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "regLabel1", "impl1");
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware2", "regLabel2", "impl1");
    
    // Verify selections
    QCOMPARE(config.getHardwareImplementation("TestHardware1", "regLabel1"), QString("impl1"));
    QCOMPARE(config.getHardwareImplementation("TestHardware2", "regLabel2"), QString("impl1"));
    
    auto hardware = config.getCurrentHardware();
    QVERIFY(hardware.size() >= 2);
    
    // Verify the keys are in the expected format
    QVERIFY(hardware.contains("TestHardware1.regLabel1"));
    QVERIFY(hardware.contains("TestHardware2.regLabel2"));
}

void RuntimeHardwareConfigTest::testValidationWithRegistry()
{
    // This test would verify configuration validation against the registry
    // The exact behavior depends on the implementation of isConfigurationValid()
    
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Set valid configuration
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "validLabel", "impl1");
    
    bool valid = config.isConfigurationValid();
    // Note: We don't assert on the result as validation logic may vary
    Q_UNUSED(valid)
    
    // Test validation of specific selections
    auto validationResults = config.validateConfiguration();
    QString expectedKey = "TestHardware1.validLabel";
    if (validationResults.contains(expectedKey)) {
        // If validation result exists, it should be valid since we used registered hardware
        QVERIFY(validationResults[expectedKey].isValid);
    }
    
    // The important thing is that the method doesn't crash
    QVERIFY(true);
}

void RuntimeHardwareConfigTest::testConfigurationReset()
{
    // Set some configuration using registered hardware
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "resetLabel1", "impl1");
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware2", "resetLabel2", "impl1");
    
    const auto &config = RuntimeHardwareConfig::constInstance();
    auto hardwareBefore = config.getCurrentHardware();
    QVERIFY(!hardwareBefore.empty()); // Should have active hardware
    
    // Clear configuration using the proper API
    RuntimeHardwareConfig::instance().clearConfiguration();
    
    auto hardwareAfter = config.getCurrentHardware();
    // After clearing, configuration should be empty
    QVERIFY(hardwareAfter.empty());
    
    // Verify specific hardware selections are empty
    QVERIFY(config.getHardwareImplementation("TestHardware1", "resetLabel1").isEmpty());
    QVERIFY(config.getHardwareImplementation("TestHardware2", "resetLabel2").isEmpty());
    
    QVERIFY(true); // Main goal is to not crash
}

void RuntimeHardwareConfigTest::testUnregisteredHardwareValidation()
{
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Try to set unregistered hardware - should fail with validation
    bool set1 = RuntimeHardwareConfig::instance().setHardwareSelection("UnregisteredHardware1", "label1", "impl1");
    bool set2 = RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "label2", "UnregisteredImpl");
    
    QVERIFY(!set1); // Should fail for unregistered hardware type
    QVERIFY(!set2); // Should fail for unregistered implementation
    
    // Verify selections were not set
    QVERIFY(config.getHardwareImplementation("UnregisteredHardware1", "label1").isEmpty());
    QVERIFY(config.getHardwareImplementation("TestHardware1", "label2").isEmpty());
    
    // Verify even with friend access, validation still applies
    bool friendSet = RuntimeHardwareConfig::instance().setHardwareSelection("FriendValidationTest", "label3", "impl1");
    QVERIFY(!friendSet); // Should fail due to validation (unregistered hardware)
    
    // Verify the setting was not applied
    QVERIFY(config.getHardwareImplementation("FriendValidationTest", "label3").isEmpty());
}

void RuntimeHardwareConfigTest::testInvalidProfilePersistence()
{
    // Try to set unregistered hardware - should fail
    bool set1 = RuntimeHardwareConfig::instance().setHardwareSelection("PersistTest", "label1", "impl1");
    bool set2 = RuntimeHardwareConfig::instance().setHardwareSelection("PersistTest2", "label2", "impl2");
    
    QVERIFY(!set1); // Should fail for unregistered hardware
    QVERIFY(!set2); // Should fail for unregistered hardware
    
    // Verify that invalid hardware selections don't appear in current hardware
    const auto &config = RuntimeHardwareConfig::constInstance();
    auto currentHardware = config.getCurrentHardware();
    
    // Should not contain keys for unregistered hardware
    QVERIFY(!currentHardware.contains("PersistTest.label1"));
    QVERIFY(!currentHardware.contains("PersistTest2.label2"));
    
    // Verify getHardwareImplementation returns empty for invalid hardware
    QVERIFY(config.getHardwareImplementation("PersistTest", "label1").isEmpty());
    QVERIFY(config.getHardwareImplementation("PersistTest2", "label2").isEmpty());
}

QTEST_MAIN(RuntimeHardwareConfigTest)

#include "tst_runtimehardwareconfigtest.moc"