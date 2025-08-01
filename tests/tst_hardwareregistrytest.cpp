#include <QtTest>
#include <QCoreApplication>
#include <QThread>
#include <QThreadPool>
#include <QRunnable>
#include <QSemaphore>
#include <QMutex>
#include <QWaitCondition>

#include <src/hardware/core/hardwareregistry.h>
#include <src/hardware/core/hardwareregistration.h>
#include <src/hardware/core/hardwareobject.h>

// Mock hardware classes for testing
class MockHardware : public HardwareObject
{
public:
    MockHardware(const QString &subKey, QObject *parent = nullptr)
        : HardwareObject("MockType", subKey, "Mock Hardware", CommunicationProtocol::Virtual, parent)
    {
    }
    
    MockHardware(const QString &hardwareKey, const QString &subKey, QObject *parent = nullptr)
        : HardwareObject(hardwareKey, subKey, "Mock Hardware", CommunicationProtocol::Virtual, parent)
    {
    }
    
    ~MockHardware() override = default;
    
    bool testConnection() override { return true; }
    void initialize() override {}
    QStringList forbiddenKeys() const override { return {}; }
    
    static QString hardwareKey() { return "MockType"; }
    static QString hardwareSubKey() { return "mock"; }
};

class MockHardwareWithDependency : public HardwareObject
{
public:
    MockHardwareWithDependency(const QString &subKey, QObject *parent = nullptr)
        : HardwareObject("MockWithDep", subKey, "Mock Hardware With Dependency", CommunicationProtocol::Virtual, parent)
    {
    }
    
    ~MockHardwareWithDependency() override = default;
    
    bool testConnection() override { return true; }
    void initialize() override {}
    QStringList forbiddenKeys() const override { return {}; }
    
    static QString hardwareKey() { return "MockWithDep"; }
    static QString hardwareSubKey() { return "mockdep"; }
};

// Mock library class for testing
class MockLibrary
{
public:
    static MockLibrary& instance() {
        static MockLibrary lib;
        return lib;
    }
    
    bool isLibraryLoaded() const { return d_loaded; }
    void setLoaded(bool loaded) { d_loaded = loaded; }
    
private:
    bool d_loaded = true; // Default to loaded for basic tests
};

class HardwareRegistryTest : public QObject
{
    Q_OBJECT

public:
    HardwareRegistryTest();
    ~HardwareRegistryTest();

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();
    
    // Basic functionality tests
    void testSingletonAccess();
    void testHardwareRegistration();
    void testHardwareRegistrationWithDependencies();
    void testHardwareAvailabilityChecking();
    void testHardwareCreation();
    void testHardwareCreationWithUnavailableDependency();
    void testDuplicateRegistration();
    void testInvalidRegistration();
    
    // Discovery and listing tests
    void testGetRegisteredHardwareTypes();
    void testGetAvailableImplementations();
    void testGetHardwareInfo();
    void testGetDependencies();
    
    // Thread safety tests
    void testConcurrentRegistration();
    void testConcurrentAccess();
    void testConcurrentCreation();
    
    // Edge cases and error handling
    void testEmptyRegistry();
    void testNullFactoryFunction();
    void testNullAvailabilityCheck();
    void testInvalidHardwareKey();
    void testCreateNonexistentHardware();

private:
    // Helper functions for thread safety testing
    void registerHardwareFromThread(const QString &key, int threadId, QSemaphore *sem, QMutex *resultMutex, QStringList *results);
    void createHardwareFromThread(const QString &key, const QString &subKey, int threadId, QSemaphore *sem, QMutex *resultMutex, QStringList *results);
    void clearTestSettings();
    
    HardwareRegistry *d_registry;
    MockLibrary *d_mockLib;
};

HardwareRegistryTest::HardwareRegistryTest()
    : d_registry(nullptr), d_mockLib(nullptr)
{
    // Use test settings location to avoid conflicts with real BlackChirp settings
    QCoreApplication::setApplicationName("BlackchirpTest");
    QCoreApplication::setOrganizationName("CrabtreeLab");
    QCoreApplication::setOrganizationDomain("crabtreelab.ucdavis.edu");
}

HardwareRegistryTest::~HardwareRegistryTest()
{
}

void HardwareRegistryTest::initTestCase()
{
    qDebug() << "Starting HardwareRegistry test suite...";
    clearTestSettings();
    d_mockLib = &MockLibrary::instance();
    d_mockLib->setLoaded(true); // Start with library available
}

void HardwareRegistryTest::cleanupTestCase()
{
    qDebug() << "HardwareRegistry test suite completed.";
}

void HardwareRegistryTest::init()
{
    // Get fresh registry instance for each test
    d_registry = &HardwareRegistry::instance();
    // Note: We can't easily reset the singleton between tests, so tests must be designed accordingly
}

void HardwareRegistryTest::cleanup()
{
    // Cleanup any created hardware objects
    // Note: Real cleanup would require registry reset functionality
}

void HardwareRegistryTest::testSingletonAccess()
{
    auto &registry1 = HardwareRegistry::instance();
    auto &registry2 = HardwareRegistry::instance();
    
    // Both references should point to the same object
    QCOMPARE(&registry1, &registry2);
    QVERIFY(&registry1 == d_registry);
}

void HardwareRegistryTest::testHardwareRegistration()
{
    QString testKey = "TestHardware";
    QString testSubKey = "test1";
    QString testName = "Test Hardware 1";
    QString testDesc = "A test hardware implementation";
    
    // Create factory function
    auto factory = []() -> HardwareObject* {
        return new MockHardware("test1");
    };
    
    // Create availability check function
    auto availCheck = []() -> bool {
        return true; // Always available for this test
    };
    
    // Register the hardware
    bool registered = d_registry->registerHardware(
        testKey, testSubKey, testName, testDesc,
        QStringList(), // No dependencies
        factory, availCheck,
        false // Not required
    );
    
    QVERIFY(registered);
    
    // Verify registration was successful
    const HardwareRegistration* info = d_registry->getRegistration(testKey, testSubKey);
    QVERIFY(info != nullptr);
    QVERIFY(d_registry->isHardwareAvailable(testKey, testSubKey));
    
    // Check that we can retrieve hardware info
    QCOMPARE(info->prettyName, testName);
    QCOMPARE(info->description, testDesc);
    QCOMPARE(info->isRequired, false);
}

void HardwareRegistryTest::testHardwareRegistrationWithDependencies()
{
    QString testKey = "TestHardwareWithDep";
    QString testSubKey = "testdep1";
    QString testName = "Test Hardware With Dependencies";
    QString testDesc = "A test hardware with dependencies";
    QStringList dependencies = {"MockLibrary", "SomeDriver"};
    
    auto factory = []() -> HardwareObject* {
        return new MockHardwareWithDependency("testdep1");
    };
    
    auto availCheck = []() -> bool {
        return MockLibrary::instance().isLibraryLoaded();
    };
    
    bool registered = d_registry->registerHardware(
        testKey, testSubKey, testName, testDesc,
        dependencies,
        factory, availCheck,
        true // Required hardware
    );
    
    QVERIFY(registered);
    
    // Verify registration and dependency information
    const HardwareRegistration* info = d_registry->getRegistration(testKey, testSubKey);
    QVERIFY(info != nullptr);
    QCOMPARE(info->dependencies, dependencies);
    QCOMPARE(info->isRequired, true);
}

void HardwareRegistryTest::testHardwareAvailabilityChecking()
{
    QString testKey = "AvailabilityTest";
    QString testSubKey = "availtest";
    
    auto factory = []() -> HardwareObject* {
        return new MockHardware("availtest");
    };
    
    auto availCheck = []() -> bool {
        return MockLibrary::instance().isLibraryLoaded();
    };
    
    // Register hardware
    d_registry->registerHardware(
        testKey, testSubKey, "Availability Test", "Test availability checking",
        QStringList(), factory, availCheck,
        false
    );
    
    // Initially should be available
    QVERIFY(d_registry->isHardwareAvailable(testKey, testSubKey));
    
    // Make library unavailable
    d_mockLib->setLoaded(false);
    d_registry->refreshAvailability(); // Refresh cached availability
    
    // Should now be unavailable
    QVERIFY(!d_registry->isHardwareAvailable(testKey, testSubKey));
    
    // Restore availability
    d_mockLib->setLoaded(true);
    d_registry->refreshAvailability(); // Refresh cached availability
    QVERIFY(d_registry->isHardwareAvailable(testKey, testSubKey));
}

void HardwareRegistryTest::testHardwareCreation()
{
    QString testKey = "CreationTest";
    QString testSubKey = "createtest";
    
    auto factory = [testKey, testSubKey]() -> HardwareObject* {
        return new MockHardware(testKey, testSubKey);
    };
    
    auto availCheck = []() -> bool { return true; };
    
    // Register hardware
    d_registry->registerHardware(
        testKey, testSubKey, "Creation Test", "Test hardware creation",
        QStringList(), factory, availCheck,
        false
    );
    
    // Create hardware instance
    HardwareObject* hw = d_registry->createHardware(testKey, testSubKey);
    
    QVERIFY(hw != nullptr);
    QCOMPARE(hw->d_key, QString("CreationTest.0")); // Assuming index 0
    
    // Clean up
    delete hw;
}

void HardwareRegistryTest::testHardwareCreationWithUnavailableDependency()
{
    QString testKey = "UnavailableTest";
    QString testSubKey = "unavailtest";
    
    auto factory = []() -> HardwareObject* {
        return new MockHardware("unavailtest");
    };
    
    auto availCheck = []() -> bool {
        return MockLibrary::instance().isLibraryLoaded();
    };
    
    // Register hardware
    d_registry->registerHardware(
        testKey, testSubKey, "Unavailable Test", "Test unavailable hardware",
        QStringList{"MockLibrary"}, factory, availCheck,
        false
    );
    
    // Make dependency unavailable
    d_mockLib->setLoaded(false);
    d_registry->refreshAvailability(); // Refresh cached availability
    
    // Attempt to create hardware - should fail
    HardwareObject* hw = d_registry->createHardware(testKey, testSubKey);
    QVERIFY(hw == nullptr);
    
    // Restore availability and try again
    d_mockLib->setLoaded(true);
    d_registry->refreshAvailability(); // Refresh cached availability
    hw = d_registry->createHardware(testKey, testSubKey);
    QVERIFY(hw != nullptr);
    
    delete hw;
}

void HardwareRegistryTest::testDuplicateRegistration()
{
    QString testKey = "DuplicateTest";
    QString testSubKey = "duptest";
    
    auto factory = []() -> HardwareObject* {
        return new MockHardware("duptest");
    };
    auto availCheck = []() -> bool { return true; };
    
    // First registration should succeed
    bool first = d_registry->registerHardware(
        testKey, testSubKey, "Duplicate Test 1", "First registration",
        QStringList(), factory, availCheck,
        false
    );
    QVERIFY(first);
    
    // Second registration with same key/subKey should fail
    bool second = d_registry->registerHardware(
        testKey, testSubKey, "Duplicate Test 2", "Second registration",
        QStringList(), factory, availCheck,
        false
    );
    QVERIFY(!second);
    
    // Verify first registration is still intact
    const HardwareRegistration* info = d_registry->getRegistration(testKey, testSubKey);
    QVERIFY(info != nullptr);
    QCOMPARE(info->prettyName, QString("Duplicate Test 1"));
}

void HardwareRegistryTest::testGetRegisteredHardwareTypes()
{
    // The registry might already have hardware types from previous tests
    // or from the actual application, so we'll add a unique one and verify it appears
    QString uniqueKey = "UniqueTypeTest";
    
    auto factory = []() -> HardwareObject* {
        return new MockHardware("unique");
    };
    auto availCheck = []() -> bool { return true; };
    
    QStringList typesBefore = d_registry->getRegisteredHardwareTypes();
    
    d_registry->registerHardware(
        uniqueKey, "unique", "Unique Type Test", "Test type listing",
        QStringList(), factory, availCheck,
        false
    );
    
    QStringList typesAfter = d_registry->getRegisteredHardwareTypes();
    
    // Should have one more type
    QCOMPARE(typesAfter.size(), typesBefore.size() + 1);
    QVERIFY(typesAfter.contains(uniqueKey));
}

void HardwareRegistryTest::testGetAvailableImplementations()
{
    QString testKey = "ImplementationTest";
    
    auto factory1 = []() -> HardwareObject* {
        return new MockHardware("impl1");
    };
    auto factory2 = []() -> HardwareObject* {
        return new MockHardware("impl2");
    };
    auto availCheck = []() -> bool { return true; };
    
    // Register two implementations of the same hardware type
    d_registry->registerHardware(
        testKey, "impl1", "Implementation 1", "First implementation",
        QStringList(), factory1, availCheck,
        false
    );
    
    d_registry->registerHardware(
        testKey, "impl2", "Implementation 2", "Second implementation",
        QStringList(), factory2, availCheck,
        false
    );
    
    QStringList implementations = d_registry->getAvailableImplementations(testKey);
    
    QVERIFY(implementations.contains("impl1"));
    QVERIFY(implementations.contains("impl2"));
    QVERIFY(implementations.size() >= 2); // Might have others from previous tests
}

void HardwareRegistryTest::testEmptyRegistry()
{
    // Test behavior with empty registry (note: real registry might not be empty)
    QStringList nonExistentTypes = d_registry->getAvailableImplementations("NonExistentType");
    QVERIFY(nonExistentTypes.isEmpty());
    
    QVERIFY(!d_registry->getRegistration("NonExistent", "none"));
    QVERIFY(!d_registry->isHardwareAvailable("NonExistent", "none"));
    
    HardwareObject* hw = d_registry->createHardware("NonExistent", "none");
    QVERIFY(hw == nullptr);
}

void HardwareRegistryTest::testCreateNonexistentHardware()
{
    HardwareObject* hw = d_registry->createHardware("CompletelyFakeType", "fakesub");
    QVERIFY(hw == nullptr);
}

// Thread safety test helper functions
void HardwareRegistryTest::registerHardwareFromThread(const QString &key, int threadId, 
                                                     QSemaphore *sem, QMutex *resultMutex, 
                                                     QStringList *results)
{
    auto factory = []() -> HardwareObject* {
        return new MockHardware("thread");
    };
    auto availCheck = []() -> bool { return true; };
    
    QString threadKey = QString("%1_Thread%2").arg(key).arg(threadId);
    QString subKey = QString("thread%1").arg(threadId);
    
    bool registered = d_registry->registerHardware(
        threadKey, subKey, QString("Thread %1 Hardware").arg(threadId), 
        QString("Hardware from thread %1").arg(threadId),
        QStringList(), factory, availCheck,
        false
    );
    
    QMutexLocker locker(resultMutex);
    if (registered) {
        results->append(QString("Thread %1: Registration successful").arg(threadId));
    } else {
        results->append(QString("Thread %1: Registration failed").arg(threadId));
    }
    
    sem->release();
}

void HardwareRegistryTest::testConcurrentRegistration()
{
    const int numThreads = 10;
    QSemaphore sem;
    QMutex resultMutex;
    QStringList results;
    
    // Start multiple threads trying to register hardware simultaneously
    for (int i = 0; i < numThreads; ++i) {
        QThread* thread = QThread::create([this, i, &sem, &resultMutex, &results]() {
            registerHardwareFromThread("ConcurrentReg", i, &sem, &resultMutex, &results);
        });
        thread->start();
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < numThreads; ++i) {
        sem.acquire();
    }
    
    // All registrations should have succeeded (different keys)
    QCOMPARE(results.size(), numThreads);
    for (const QString &result : results) {
        QVERIFY(result.contains("successful"));
    }
    
    // Verify all hardware types were registered
    for (int i = 0; i < numThreads; ++i) {
        QString threadKey = QString("ConcurrentReg_Thread%1").arg(i);
        QString subKey = QString("thread%1").arg(i);
        QVERIFY(d_registry->getRegistration(threadKey, subKey));
    }
}

void HardwareRegistryTest::createHardwareFromThread(const QString &key, const QString &subKey, 
                                                   int threadId, QSemaphore *sem, 
                                                   QMutex *resultMutex, QStringList *results)
{
    HardwareObject* hw = d_registry->createHardware(key, subKey);
    
    QMutexLocker locker(resultMutex);
    if (hw) {
        results->append(QString("Thread %1: Creation successful").arg(threadId));
        delete hw; // Clean up
    } else {
        results->append(QString("Thread %1: Creation failed").arg(threadId));
    }
    
    sem->release();
}

void HardwareRegistryTest::testConcurrentCreation()
{
    // First register hardware that all threads will try to create
    QString testKey = "ConcurrentCreate";
    QString testSubKey = "create";
    
    auto factory = []() -> HardwareObject* {
        return new MockHardware("create");
    };
    auto availCheck = []() -> bool { return true; };
    
    d_registry->registerHardware(
        testKey, testSubKey, "Concurrent Creation Test", "Test concurrent creation",
        QStringList(), factory, availCheck,
        false
    );
    
    const int numThreads = 10;
    QSemaphore sem;
    QMutex resultMutex;
    QStringList results;
    
    // Start multiple threads trying to create the same hardware simultaneously
    for (int i = 0; i < numThreads; ++i) {
        QThread* thread = QThread::create([this, testKey, testSubKey, i, &sem, &resultMutex, &results]() {
            createHardwareFromThread(testKey, testSubKey, i, &sem, &resultMutex, &results);
        });
        thread->start();
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < numThreads; ++i) {
        sem.acquire();
    }
    
    // All creations should have succeeded
    QCOMPARE(results.size(), numThreads);
    for (const QString &result : results) {
        QVERIFY(result.contains("successful"));
    }
}

void HardwareRegistryTest::testConcurrentAccess()
{
    // Test concurrent read operations (should always be safe)
    const int numThreads = 5;
    QSemaphore sem;
    QMutex resultMutex;
    QStringList results;
    
    // Register some test hardware first
    QString testKey = "ConcurrentAccess";
    auto factory = []() -> HardwareObject* { return new MockHardware("access"); };
    auto availCheck = []() -> bool { return true; };
    
    d_registry->registerHardware(testKey, "access", "Concurrent Access Test", "Test",
                                QStringList(), factory, availCheck,
                                false);
    
    // Start threads doing various read operations
    for (int i = 0; i < numThreads; ++i) {
        QThread* thread = QThread::create([this, testKey, i, &sem, &resultMutex, &results]() {
            // Perform various read operations
            bool registered = (d_registry->getRegistration(testKey, "access") != nullptr);
            bool available = d_registry->isHardwareAvailable(testKey, "access");
            QStringList types = d_registry->getRegisteredHardwareTypes();
            QStringList impls = d_registry->getAvailableImplementations(testKey);
            auto info = d_registry->getRegistration(testKey, "access");
            
            QMutexLocker locker(&resultMutex);
            if (registered && available && !types.isEmpty() && !impls.isEmpty()) {
                results.append(QString("Thread %1: All reads successful").arg(i));
            } else {
                results.append(QString("Thread %1: Some reads failed").arg(i));
            }
            sem.release();
        });
        thread->start();
    }
    
    // Wait for all threads
    for (int i = 0; i < numThreads; ++i) {
        sem.acquire();
    }
    
    // All threads should have succeeded
    QCOMPARE(results.size(), numThreads);
    for (const QString &result : results) {
        QVERIFY(result.contains("successful"));
    }
}

void HardwareRegistryTest::testInvalidRegistration()
{
    // Test registration with invalid parameters
    auto factory = []() -> HardwareObject* { return new MockHardware("invalid"); };
    auto availCheck = []() -> bool { return true; };
    
    // Empty key should fail
    bool registered = d_registry->registerHardware("", "subkey", "Name", "Desc", QStringList(), factory, availCheck, false);
    QVERIFY(!registered);
    
    // Empty subkey should fail
    registered = d_registry->registerHardware("key", "", "Name", "Desc", QStringList(), factory, availCheck, false);
    QVERIFY(!registered);
    
    // Empty name should fail
    registered = d_registry->registerHardware("key", "subkey", "", "Desc", QStringList(), factory, availCheck, false);
    QVERIFY(!registered);
}

void HardwareRegistryTest::testGetHardwareInfo()
{
    // Register test hardware
    QString testKey = "InfoTest";
    QString testSubKey = "info";
    QString testName = "Info Test Hardware";
    QString testDesc = "Test getting hardware info";
    QStringList testDeps = {"TestDep1", "TestDep2"};
    
    auto factory = []() -> HardwareObject* { return new MockHardware("info"); };
    auto availCheck = []() -> bool { return true; };
    
    bool registered = d_registry->registerHardware(testKey, testSubKey, testName, testDesc, testDeps, factory, availCheck, true);
    QVERIFY(registered);
    
    // Get hardware info
    const HardwareRegistration* info = d_registry->getRegistration(testKey, testSubKey);
    QVERIFY(info != nullptr);
    QCOMPARE(info->key, testKey);
    QCOMPARE(info->subKey, testSubKey);
    QCOMPARE(info->prettyName, testName);
    QCOMPARE(info->description, testDesc);
    QCOMPARE(info->dependencies, testDeps);
    QCOMPARE(info->isRequired, true);
}

void HardwareRegistryTest::testGetDependencies()
{
    // Register hardware with dependencies
    QString testKey = "DepTest";
    QString testSubKey = "dep";
    QStringList dependencies = {"Library1", "Library2", "Library3"};
    
    auto factory = []() -> HardwareObject* { return new MockHardware("dep"); };
    auto availCheck = []() -> bool { return true; };
    
    bool registered = d_registry->registerHardware(testKey, testSubKey, "Dependency Test", "Test dependencies", 
                                                   dependencies, factory, availCheck, false);
    QVERIFY(registered);
    
    // Verify dependencies are stored correctly
    const HardwareRegistration* info = d_registry->getRegistration(testKey, testSubKey);
    QVERIFY(info != nullptr);
    QCOMPARE(info->dependencies, dependencies);
}

void HardwareRegistryTest::testNullFactoryFunction()
{
    // Test registration with null factory function
    std::function<HardwareObject*()> nullFactory;
    auto availCheck = []() -> bool { return true; };
    
    bool registered = d_registry->registerHardware("NullFactory", "null", "Null Factory Test", "Test null factory",
                                                   QStringList(), nullFactory, availCheck, false);
    QVERIFY(!registered);
}

void HardwareRegistryTest::testNullAvailabilityCheck()
{
    // Test registration with null availability check (should still work)
    auto factory = []() -> HardwareObject* { return new MockHardware("nullavail"); };
    std::function<bool()> nullAvailCheck;
    
    bool registered = d_registry->registerHardware("NullAvail", "null", "Null Availability Test", "Test null availability",
                                                   QStringList(), factory, nullAvailCheck, false);
    QVERIFY(registered);
    
    // Availability check should return false for null availability function
    bool available = d_registry->isHardwareAvailable("NullAvail", "null");
    QVERIFY(!available);
}

void HardwareRegistryTest::testInvalidHardwareKey()
{
    // Test operations with invalid hardware keys
    QVERIFY(!d_registry->isHardwareAvailable("NonExistent", "none"));
    
    HardwareObject* obj = d_registry->createHardware("NonExistent", "none");
    QVERIFY(obj == nullptr);
    
    const HardwareRegistration* info = d_registry->getRegistration("NonExistent", "none");
    QVERIFY(info == nullptr);
    
    QStringList impls = d_registry->getAvailableImplementations("NonExistent");
    QVERIFY(impls.isEmpty());
}

void HardwareRegistryTest::clearTestSettings()
{
    // Clear out any existing test settings to ensure clean test environment
    QSettings s("CrabtreeLab", "BlackchirpTest");
    s.setFallbacksEnabled(false);
    s.clear();
    s.sync();
}

QTEST_MAIN(HardwareRegistryTest)

#include "tst_hardwareregistrytest.moc"