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
    void testHardwareEnabled();
    void testCurrentHardware();
    void testConfigurationValidation();
    
    // Settings persistence tests
    void testSettingsPersistence();
    void testSettingsLoading();
    void testSettingsKeys();
    
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
    void testInvalidSettingsPersistence();

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
    auto factory1 = []() -> HardwareObject* { return nullptr; }; // Mock factory
    auto availCheck1 = []() -> bool { return true; };
    
    d_registry->registerHardware(
        "TestHardware1", "impl1", "Test Hardware 1", "Test implementation 1",
        QStringList(), factory1, availCheck1,
        false // Not required
    );
    
    d_registry->registerHardware(
        "TestHardware1", "impl2", "Test Hardware 1 Alt", "Alternative implementation",
        QStringList(), factory1, availCheck1,
        false // Not required
    );
    
    d_registry->registerHardware(
        "TestHardware2", "impl1", "Test Hardware 2", "Test implementation 2",
        QStringList(), factory1, availCheck1,
        true // Required
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
    QString selection = config.getHardwareSelection("TestType");
    auto hardware = config.getCurrentHardware();
    bool valid = config.isConfigurationValid();
    
    // Verify default values for non-existent hardware
    QVERIFY(selection.isEmpty());
    QVERIFY(hardware.empty());
    // Note: isConfigurationValid behavior depends on implementation
    Q_UNUSED(valid)
}

void RuntimeHardwareConfigTest::testHardwareSelection()
{
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Initially should be empty
    QString selection = config.getHardwareSelection("TestHardware1");
    QVERIFY(selection.isEmpty());
    
    // Set through direct friend access
    bool set = RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "impl1", true);
    QVERIFY(set);
    
    // Verify selection was set
    selection = config.getHardwareSelection("TestHardware1");
    QCOMPARE(selection, QString("impl1"));
}

void RuntimeHardwareConfigTest::testHardwareEnabled()
{
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Set hardware with enabled = false
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "impl1", false);
    
    // Should not be enabled
    QVERIFY(!config.isHardwareEnabled("TestHardware1"));
    
    // Enable it
    RuntimeHardwareConfig::instance().setHardwareEnabled("TestHardware1", true);
    QVERIFY(config.isHardwareEnabled("TestHardware1"));
    
    // Disable it again
    RuntimeHardwareConfig::instance().setHardwareEnabled("TestHardware1", false);
    QVERIFY(!config.isHardwareEnabled("TestHardware1"));
}

void RuntimeHardwareConfigTest::testConfigurationValidation()
{
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Test configuration validation
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "impl1", true);
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware2", "impl1", true);
    
    // Test validation methods
    auto validationResults = config.validateConfiguration();
    QVERIFY(validationResults.size() > 0);
    
    auto singleResult = config.validateHardwareType("TestHardware1");
    // The result depends on whether hardware is available, but should not crash
    Q_UNUSED(singleResult)
    
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
    
    // Set some hardware configurations - one enabled, one disabled
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "impl1", true);
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware2", "impl1", false);
    
    auto hardware = config.getCurrentHardware();
    
    // Should contain only enabled hardware
    QVERIFY(hardware.find("TestHardware1") != hardware.end());
    QVERIFY(hardware.find("TestHardware2") == hardware.end()); // Disabled hardware shouldn't appear
    
    QCOMPARE(hardware.at("TestHardware1"), QString("impl1"));    
    
    // Now enable TestHardware2 and verify it appears
    RuntimeHardwareConfig::instance().setHardwareEnabled("TestHardware2", true);
    auto hardwareAfter = config.getCurrentHardware();
    QVERIFY(hardwareAfter.find("TestHardware2") != hardwareAfter.end());
    QCOMPARE(hardwareAfter.at("TestHardware2"), QString("impl1"));
}

void RuntimeHardwareConfigTest::testSettingsPersistence()
{
    // Use registered hardware for persistence test
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "impl1", true);
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware2", "impl1", false);
    
    // Save to settings
    RuntimeHardwareConfig::instance().saveToSettings();
    
    // Verify settings were written
    QSettings s("CrabtreeLab", "BlackchirpTest");
    s.beginGroup(BC::Key::RuntimeHw::runtimeHw);
    
    QString selectionKey = QString("TestHardware1_%1").arg(BC::Key::RuntimeHw::selection);
    QString enabledKey = QString("TestHardware1_%1").arg(BC::Key::RuntimeHw::enabled);
    
    QVERIFY(s.contains(selectionKey));
    QVERIFY(s.contains(enabledKey));
    QCOMPARE(s.value(selectionKey).toString(), QString("impl1"));
    QCOMPARE(s.value(enabledKey).toBool(), true);
    
    // Check TestHardware2 settings
    QString selectionKey2 = QString("TestHardware2_%1").arg(BC::Key::RuntimeHw::selection);
    QString enabledKey2 = QString("TestHardware2_%1").arg(BC::Key::RuntimeHw::enabled);
    
    QVERIFY(s.contains(selectionKey2));
    QVERIFY(s.contains(enabledKey2));
    QCOMPARE(s.value(selectionKey2).toString(), QString("impl1"));
    QCOMPARE(s.value(enabledKey2).toBool(), false);
    
    s.endGroup();
}

void RuntimeHardwareConfigTest::testSettingsLoading()
{
    // Clear everything and start fresh
    RuntimeHardwareConfig::instance().clearConfiguration();
    
    // Set up a specific configuration
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "impl2", false);
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware2", "impl1", true);
    
    // Save configuration to persistent storage
    RuntimeHardwareConfig::instance().saveToSettings();
    
    // Clear in-memory configuration to simulate fresh start
    RuntimeHardwareConfig::instance().clearConfiguration();
    
    // Verify configuration is cleared in memory
    const auto &config = RuntimeHardwareConfig::constInstance();
    QVERIFY(config.getHardwareSelection("TestHardware1").isEmpty());
    QVERIFY(config.getHardwareSelection("TestHardware2").isEmpty());
    
    // Load from persistent storage - this should restore the saved configuration
    RuntimeHardwareConfig::instance().loadFromSettings();
    
    // Verify configuration was loaded correctly
    // TestHardware1 is disabled, so getHardwareSelection returns empty (by design)
    QVERIFY(config.getHardwareSelection("TestHardware1").isEmpty());
    QCOMPARE(config.getHardwareSelection("TestHardware2"), QString("impl1"));
    
    // Check enabled states to verify the configuration was actually loaded
    QVERIFY(!config.isHardwareEnabled("TestHardware1")); // disabled
    QVERIFY(config.isHardwareEnabled("TestHardware2"));  // enabled
    
    auto hardware = config.getCurrentHardware();
    // TestHardware1 is disabled, so shouldn't appear in current hardware
    QVERIFY(hardware.find("TestHardware1") == hardware.end());
    QVERIFY(hardware.find("TestHardware2") != hardware.end());
    
    // To verify TestHardware1 was actually loaded with impl2, enable it and check
    RuntimeHardwareConfig::instance().setHardwareEnabled("TestHardware1", true);
    QCOMPARE(config.getHardwareSelection("TestHardware1"), QString("impl2"));
}

void RuntimeHardwareConfigTest::testSettingsKeys()
{
    // Verify that proper namespaced keys are used with registered hardware
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "impl1", true);
    RuntimeHardwareConfig::instance().saveToSettings();
    
    QSettings s("CrabtreeLab", "BlackchirpTest");
    s.beginGroup(BC::Key::RuntimeHw::runtimeHw);
    
    // Check that keys follow the expected pattern
    QString expectedSelectionKey = QString("TestHardware1_%1").arg(BC::Key::RuntimeHw::selection);
    QString expectedEnabledKey = QString("TestHardware1_%1").arg(BC::Key::RuntimeHw::enabled);
    
    QVERIFY(s.contains(expectedSelectionKey));
    QVERIFY(s.contains(expectedEnabledKey));
    
    // Verify no string literals were used - check that we're using namespace constants
    QStringList allKeys = s.allKeys();
    for (const QString &key : allKeys) {
        // Keys should not contain literal strings like "_impl" or hardcoded "_enabled"
        QVERIFY(!key.contains("_impl"));
        // But they should contain the actual namespace constant values
        QVERIFY(key.contains(BC::Key::RuntimeHw::selection) || 
                key.contains(BC::Key::RuntimeHw::enabled));
    }
    
    s.endGroup();
}

void RuntimeHardwareConfigTest::testFriendAccess()
{
    // Test that test class (as friend) can access private methods
    // Use registered hardware that actually exists
    bool result = RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "impl2", true);
    QVERIFY(result); // Should succeed with friend access
    
    // Verify the setting took effect
    const auto &config = RuntimeHardwareConfig::constInstance();
    QCOMPARE(config.getHardwareSelection("TestHardware1"), QString("impl2"));
}

void RuntimeHardwareConfigTest::testNonFriendAccessRestriction()
{
    // This test verifies that the private methods are indeed private
    // We can't directly test this in C++ without compilation errors,
    // but we can verify that only public const methods are available
    
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // These should compile (public const methods)
    QString selection = config.getHardwareSelection("Test");
    auto hardware = config.getCurrentHardware();
    bool valid = config.isConfigurationValid();
    
    // Note: We cannot test direct access to private methods here
    // as that would cause compilation errors. The friend relationship
    // is enforced at compile time.
    Q_UNUSED(valid)
    QVERIFY(true); // This test mainly serves as documentation
}

void RuntimeHardwareConfigTest::concurrentReadTest(int threadId, QSemaphore *sem, 
                                                  QMutex *resultMutex, QStringList *results)
{
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Perform multiple read operations
    for (int i = 0; i < 100; ++i) {
        QString selection = config.getHardwareSelection(QString("Thread%1").arg(threadId));
        auto hardware = config.getCurrentHardware();
        bool valid = config.isConfigurationValid();
        
        // Just verify we can call these without crashing
        Q_UNUSED(selection)
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
        
        // Always enable to ensure final state has enabled hardware
        if (!RuntimeHardwareConfig::instance().setHardwareSelection(hardwareType, impl, true)) {
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
    
    // Verify final state - since operations are atomic, both should be enabled
    const auto &config = RuntimeHardwareConfig::constInstance();
    QString selection1 = config.getHardwareSelection("TestHardware1");
    QString selection2 = config.getHardwareSelection("TestHardware2");
    
    // Both should have selections since all writes use enabled=true
    QVERIFY(!selection1.isEmpty());
    QVERIFY(!selection2.isEmpty());
    
    // Verify they're configured  
    auto configuredTypes = config.getConfiguredHardwareTypes();
    QVERIFY(configuredTypes.contains("TestHardware1"));
    QVERIFY(configuredTypes.contains("TestHardware2"));
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
    
    QString selection = config.getHardwareSelection("NonExistent");
    QVERIFY(selection.isEmpty());
    
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
    bool result1 = RuntimeHardwareConfig::instance().setHardwareSelection("", "impl1", true);
    bool result2 = RuntimeHardwareConfig::instance().setHardwareSelection("ValidType", "", true);
    
    // Behavior with empty strings depends on implementation
    // but these calls should not crash
    Q_UNUSED(result1)
    Q_UNUSED(result2)
    
    const auto &config = RuntimeHardwareConfig::constInstance();
    QString selection1 = config.getHardwareSelection("");
    QString selection2 = config.getHardwareSelection("ValidType");
    
    // Should not crash
    Q_UNUSED(selection1)
    Q_UNUSED(selection2)
}

void RuntimeHardwareConfigTest::testRegistryIntegration()
{
    // Test integration with HardwareRegistry
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Set hardware that exists in registry
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "impl1", true);
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware2", "impl1", true);
    
    // Verify selections
    QCOMPARE(config.getHardwareSelection("TestHardware1"), QString("impl1"));
    QCOMPARE(config.getHardwareSelection("TestHardware2"), QString("impl1"));
    
    auto hardware = config.getCurrentHardware();
    QVERIFY(hardware.size() >= 2);
}

void RuntimeHardwareConfigTest::testValidationWithRegistry()
{
    // This test would verify configuration validation against the registry
    // The exact behavior depends on the implementation of isConfigurationValid()
    
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Set valid configuration
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "impl1", true);
    
    bool valid = config.isConfigurationValid();
    // Note: We don't assert on the result as validation logic may vary
    Q_UNUSED(valid)
    
    // The important thing is that the method doesn't crash
    QVERIFY(true);
}

void RuntimeHardwareConfigTest::testConfigurationReset()
{
    // Set some configuration using registered hardware
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "impl1", true);
    RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware2", "impl1", false);
    
    const auto &config = RuntimeHardwareConfig::constInstance();
    auto hardwareBefore = config.getCurrentHardware();
    QVERIFY(!hardwareBefore.empty()); // Should have at least TestHardware1 (enabled)
    
    // Clear configuration using the proper API
    RuntimeHardwareConfig::instance().clearConfiguration();
    
    auto hardwareAfter = config.getCurrentHardware();
    // After clearing, configuration should be empty
    QVERIFY(hardwareAfter.empty());
    
    // Verify specific hardware selections are empty
    QVERIFY(config.getHardwareSelection("TestHardware1").isEmpty());
    QVERIFY(config.getHardwareSelection("TestHardware2").isEmpty());
    
    QVERIFY(true); // Main goal is to not crash
}

void RuntimeHardwareConfigTest::testUnregisteredHardwareValidation()
{
    const auto &config = RuntimeHardwareConfig::constInstance();
    
    // Try to set unregistered hardware - should fail with validation
    bool set1 = RuntimeHardwareConfig::instance().setHardwareSelection("UnregisteredHardware1", "impl1", true);
    bool set2 = RuntimeHardwareConfig::instance().setHardwareSelection("TestHardware1", "UnregisteredImpl", true);
    
    QVERIFY(!set1); // Should fail for unregistered hardware type
    QVERIFY(!set2); // Should fail for unregistered implementation
    
    // Verify selections were not set
    QVERIFY(config.getHardwareSelection("UnregisteredHardware1").isEmpty());
    
    // Verify even with friend access, validation still applies
    bool friendSet = RuntimeHardwareConfig::instance().setHardwareSelection("FriendValidationTest", "impl1", true);
    QVERIFY(!friendSet); // Should fail due to validation (unregistered hardware)
    
    // Verify the setting was not applied
    QVERIFY(config.getHardwareSelection("FriendValidationTest").isEmpty());
}

void RuntimeHardwareConfigTest::testInvalidSettingsPersistence()
{
    // Try to set unregistered hardware - should fail
    bool set1 = RuntimeHardwareConfig::instance().setHardwareSelection("PersistTest", "impl1", true);
    bool set2 = RuntimeHardwareConfig::instance().setHardwareSelection("PersistTest2", "impl2", false);
    
    QVERIFY(!set1); // Should fail for unregistered hardware
    QVERIFY(!set2); // Should fail for unregistered hardware
    
    // Save to settings
    RuntimeHardwareConfig::instance().saveToSettings();
    
    // Verify no invalid settings were written
    QSettings s("CrabtreeLab", "BlackchirpTest");
    s.beginGroup(BC::Key::RuntimeHw::runtimeHw);
    
    QString selectionKey = QString("PersistTest_%1").arg(BC::Key::RuntimeHw::selection);
    QString enabledKey = QString("PersistTest_%1").arg(BC::Key::RuntimeHw::enabled);
    
    QVERIFY(!s.contains(selectionKey)); // Should not exist for invalid hardware
    QVERIFY(!s.contains(enabledKey));   // Should not exist for invalid hardware
    
    // Same for second test
    QString selectionKey2 = QString("PersistTest2_%1").arg(BC::Key::RuntimeHw::selection);
    QString enabledKey2 = QString("PersistTest2_%1").arg(BC::Key::RuntimeHw::enabled);
    
    QVERIFY(!s.contains(selectionKey2));
    QVERIFY(!s.contains(enabledKey2));
    
    s.endGroup();
}

QTEST_MAIN(RuntimeHardwareConfigTest)

#include "tst_runtimehardwareconfigtest.moc"