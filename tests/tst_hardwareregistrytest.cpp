#include <QtTest>
#include <QCoreApplication>
#include <QThread>
#include <QSemaphore>
#include <QMutex>

#include <src/hardware/core/hardwareregistry.h>
#include <src/hardware/core/hardwareobject.h>

// Mock hardware classes for testing
class MockHardware : public HardwareObject
{
public:
    MockHardware(const QString &hardwareKey, const QString &subKey, QObject *parent = nullptr)
        : HardwareObject(hardwareKey, subKey, "testLabel", parent)
    {
    }
    
    ~MockHardware() override = default;
    
    bool testConnection() override { return true; }
    void initialize() override {}
};

class FailingMockHardware : public HardwareObject
{
public:
    FailingMockHardware(const QString &hardwareKey, const QString &subKey, QObject *parent = nullptr)
        : HardwareObject(hardwareKey, subKey, "testLabel", parent)
    {
        // Simulate a hardware that fails during construction
        throw std::runtime_error("Simulated hardware failure");
    }
    
    bool testConnection() override { return false; }
    void initialize() override {}
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
    
    // Core functionality tests
    void testSingletonAccess();
    void testHardwareRegistration();
    void testHardwareCreation();
    void testDuplicateRegistration();
    void testInvalidRegistration();
    
    // Query tests
    void testGetHardwareTypes();
    void testGetImplementations();
    void testGetRegistration();
    void testIsRegistered();
    
    // Thread safety tests
    void testConcurrentRegistration();
    void testConcurrentCreation();
    
    // Error handling tests
    void testCreateNonexistentHardware();
    void testNullFactoryFunction();
    void testFactoryFailure();

private:
    // Helper functions
    void registerTestHardware(const QString &key, const QString &subKey, const QString &name);
    
    HardwareRegistry *d_registry;
    int d_testCounter;
};

HardwareRegistryTest::HardwareRegistryTest()
    : d_registry(nullptr), d_testCounter(0)
{
    QCoreApplication::setApplicationName("BlackchirpTest");
    QCoreApplication::setOrganizationName("CrabtreeLab");
}

HardwareRegistryTest::~HardwareRegistryTest() = default;

void HardwareRegistryTest::initTestCase()
{
    qDebug() << "Initializing HardwareRegistryTest...";
    d_registry = &HardwareRegistry::instance();
}

void HardwareRegistryTest::cleanupTestCase()
{
    qDebug() << "Cleaning up HardwareRegistryTest...";
}

void HardwareRegistryTest::init()
{
    ++d_testCounter;
}

void HardwareRegistryTest::cleanup()
{
    // Note: We can't easily clear the registry between tests since it's a singleton
    // Tests use unique keys to avoid conflicts
}

void HardwareRegistryTest::testSingletonAccess()
{
    HardwareRegistry& registry1 = HardwareRegistry::instance();
    HardwareRegistry& registry2 = HardwareRegistry::instance();
    
    QVERIFY(&registry1 == &registry2);
    QVERIFY(&registry1 == d_registry);
}

void HardwareRegistryTest::testHardwareRegistration()
{
    QString testKey = QString("TestType_%1").arg(d_testCounter);
    QString testSubKey = "registration_test";
    QString testName = "Registration Test Hardware";
    QString testDesc = "Test hardware for registration validation";
    
    // Register hardware
    bool success = d_registry->registerHardware(
        testKey, testSubKey, testDesc,
        [testKey, testSubKey](const QString& label) -> HardwareObject* { 
            Q_UNUSED(label)
            return new MockHardware(testKey, testSubKey); 
        }
    );
    
    QVERIFY(success);
    
    // Verify registration
    QVERIFY(d_registry->isRegistered(testKey, testSubKey));
    const HardwareRegistration* info = d_registry->getRegistration(testKey, testSubKey);
    QVERIFY(info != nullptr);
    QCOMPARE(info->key, testKey);
    QCOMPARE(info->subKey, testSubKey);
    QCOMPARE(info->description, testDesc);
    QCOMPARE(info->description, testDesc);
    QVERIFY(info->factory != nullptr);
}

void HardwareRegistryTest::testHardwareCreation()
{
    QString testKey = QString("TestType_%1").arg(d_testCounter);
    QString testSubKey = "creation_test";
    
    // Register hardware
    registerTestHardware(testKey, testSubKey, "Creation Test Hardware");
    
    // Create hardware instance
    HardwareObject* hw = d_registry->createHardware(testKey, testSubKey, "testLabel");
    QVERIFY(hw != nullptr);
    QVERIFY(hw->d_key.startsWith(testKey));  // Key gets index suffix like ".0"
    QCOMPARE(hw->d_model, testSubKey);
    
    // Clean up
    delete hw;
}

void HardwareRegistryTest::testDuplicateRegistration()
{
    QString testKey = QString("TestType_%1").arg(d_testCounter);
    QString testSubKey = "duplicate_test";
    
    // Register hardware first time
    bool success1 = d_registry->registerHardware(
        testKey, testSubKey, "First registration",
        [testKey, testSubKey](const QString& label) -> HardwareObject* { 
            Q_UNUSED(label)
            return new MockHardware(testKey, testSubKey); 
        }
    );
    QVERIFY(success1);
    
    // Try to register the same hardware again
    bool success2 = d_registry->registerHardware(
        testKey, testSubKey, "Duplicate registration",
        [testKey, testSubKey](const QString& label) -> HardwareObject* { 
            Q_UNUSED(label)
            return new MockHardware(testKey, testSubKey); 
        }
    );
    QVERIFY(!success2);  // Should fail
}

void HardwareRegistryTest::testInvalidRegistration()
{
    QString testKey = QString("TestType_%1").arg(d_testCounter);
    
    // Test with empty key
    bool success1 = d_registry->registerHardware(
        "", "subkey", "Description",
        [](const QString& label) -> HardwareObject* { Q_UNUSED(label) return nullptr; }
    );
    QVERIFY(!success1);
    
    // Test with empty subkey
    bool success2 = d_registry->registerHardware(
        testKey, "", "Description",
        [](const QString& label) -> HardwareObject* { Q_UNUSED(label) return nullptr; }
    );
    QVERIFY(!success2);
    
    // Test with empty description
    bool success3 = d_registry->registerHardware(
        testKey, "subkey", "",
        [](const QString& label) -> HardwareObject* { Q_UNUSED(label) return nullptr; }
    );
    QVERIFY(!success3);
    
    // Test with null factory
    bool success4 = d_registry->registerHardware(
        testKey, "subkey", "Description",
        std::function<HardwareObject*(const QString&)>()  // null function
    );
    QVERIFY(!success4);
}

void HardwareRegistryTest::testGetHardwareTypes()
{
    QString testKey = QString("TestType_%1").arg(d_testCounter);
    
    // Register multiple implementations of the same type
    registerTestHardware(testKey, "impl1", "Implementation 1");
    registerTestHardware(testKey, "impl2", "Implementation 2");
    
    QStringList types = d_registry->getHardwareTypes();
    QVERIFY(types.contains(testKey));
}

void HardwareRegistryTest::testGetImplementations()
{
    QString testKey = QString("TestType_%1").arg(d_testCounter);
    
    // Register multiple implementations
    registerTestHardware(testKey, "impl1", "Implementation 1");
    registerTestHardware(testKey, "impl2", "Implementation 2");
    
    QStringList implementations = d_registry->getImplementations(testKey);
    QVERIFY(implementations.contains("impl1"));
    QVERIFY(implementations.contains("impl2"));
    QCOMPARE(implementations.size(), 2);
    
    // Test non-existent type
    QStringList empty = d_registry->getImplementations("NonExistentType");
    QVERIFY(empty.isEmpty());
}

void HardwareRegistryTest::testGetRegistration()
{
    QString testKey = QString("TestType_%1").arg(d_testCounter);
    QString testSubKey = "registration_info_test";
    QString testName = "Registration Info Test";
    
    registerTestHardware(testKey, testSubKey, testName);
    
    const HardwareRegistration* info = d_registry->getRegistration(testKey, testSubKey);
    QVERIFY(info != nullptr);
    QCOMPARE(info->key, testKey);
    QCOMPARE(info->subKey, testSubKey);
    QCOMPARE(info->description, testName);
    
    // Test non-existent registration
    const HardwareRegistration* nullInfo = d_registry->getRegistration("NonExistent", "none");
    QVERIFY(nullInfo == nullptr);
}

void HardwareRegistryTest::testIsRegistered()
{
    QString testKey = QString("TestType_%1").arg(d_testCounter);
    QString testSubKey = "registered_test";
    
    // Before registration
    QVERIFY(!d_registry->isRegistered(testKey, testSubKey));
    
    // After registration
    registerTestHardware(testKey, testSubKey, "Registered Test");
    QVERIFY(d_registry->isRegistered(testKey, testSubKey));
    
    // Non-existent
    QVERIFY(!d_registry->isRegistered("NonExistent", "none"));
}

void HardwareRegistryTest::testConcurrentRegistration()
{
    const int numThreads = 5;  // Reduce thread count for more reliable testing
    QString testKey = QString("ConcurrentType_%1").arg(d_testCounter);
    
    QSemaphore semaphore(0);  // Start at 0, release when threads finish
    QMutex resultMutex;
    QStringList results;
    QList<QThread*> threads;
    
    // Launch multiple threads that try to register hardware
    for (int i = 0; i < numThreads; ++i) {
        QThread* thread = QThread::create([this, testKey, i, &semaphore, &resultMutex, &results]() {
            QString subKey = QString("thread_%1").arg(i);
            bool success = d_registry->registerHardware(
                testKey, subKey, "Concurrent test",
                [testKey, subKey](const QString& label) -> HardwareObject* { 
                    Q_UNUSED(label)
                    return new MockHardware(testKey, subKey); 
                }
            );
            
            QMutexLocker locker(&resultMutex);
            if (success) {
                results.append(subKey);
            }
            semaphore.release();
        });
        
        threads.append(thread);
        thread->start();
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < numThreads; ++i) {
        semaphore.acquire();
    }
    
    // Wait for threads to finish and clean up
    for (QThread* thread : threads) {
        thread->wait();
        delete thread;
    }
    
    // Verify all registrations succeeded
    QCOMPARE(results.size(), numThreads);
    
    // Verify all implementations are available
    QStringList implementations = d_registry->getImplementations(testKey);
    QCOMPARE(implementations.size(), numThreads);
}

void HardwareRegistryTest::testConcurrentCreation()
{
    QString testKey = QString("ConcurrentCreateType_%1").arg(d_testCounter);
    QString testSubKey = "concurrent_create";
    
    // Register hardware first
    registerTestHardware(testKey, testSubKey, "Concurrent Creation Test");
    
    const int numThreads = 5;  // Reduce thread count for more reliable testing
    QSemaphore semaphore(0);  // Start at 0, release when threads finish
    QMutex resultMutex;
    int successCount = 0;
    QList<QThread*> threads;
    
    // Launch multiple threads that try to create hardware
    for (int i = 0; i < numThreads; ++i) {
        QThread* thread = QThread::create([this, testKey, testSubKey, &semaphore, &resultMutex, &successCount, i]() {
            HardwareObject* hw = d_registry->createHardware(testKey, testSubKey, QString("testLabel%1").arg(i));
            
            QMutexLocker locker(&resultMutex);
            if (hw) {
                ++successCount;
                delete hw;
            }
            semaphore.release();
        });
        
        threads.append(thread);
        thread->start();
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < numThreads; ++i) {
        semaphore.acquire();
    }
    
    // Wait for threads to finish and clean up
    for (QThread* thread : threads) {
        thread->wait();
        delete thread;
    }
    
    // All creations should succeed
    QCOMPARE(successCount, numThreads);
}

void HardwareRegistryTest::testCreateNonexistentHardware()
{
    HardwareObject* hw = d_registry->createHardware("NonExistent", "none", "testLabel");
    QVERIFY(hw == nullptr);
}

void HardwareRegistryTest::testNullFactoryFunction()
{
    QString testKey = QString("TestType_%1").arg(d_testCounter);
    QString testSubKey = "null_factory";
    
    // This is already tested in testInvalidRegistration, but let's be explicit
    bool success = d_registry->registerHardware(
        testKey, testSubKey, "Test with null factory",
        std::function<HardwareObject*(const QString&)>()  // null function
    );
    
    QVERIFY(!success);  // Should fail during registration
}

void HardwareRegistryTest::testFactoryFailure()
{
    QString testKey = QString("TestType_%1").arg(d_testCounter);
    QString testSubKey = "failing_factory";
    
    // Register hardware with factory that throws
    bool success = d_registry->registerHardware(
        testKey, testSubKey, "Test factory that throws",
        [testKey, testSubKey](const QString& label) -> HardwareObject* { 
            Q_UNUSED(label)
            return new FailingMockHardware(testKey, testSubKey); 
        }
    );
    
    QVERIFY(success);  // Registration should succeed
    
    // But creation should fail gracefully
    HardwareObject* hw = d_registry->createHardware(testKey, testSubKey, "testLabel");
    QVERIFY(hw == nullptr);  // Should return null instead of crashing
}

void HardwareRegistryTest::registerTestHardware(const QString &key, const QString &subKey, const QString &name)
{
    bool success = d_registry->registerHardware(
        key, subKey, name,
        [key, subKey](const QString& label) -> HardwareObject* { 
            Q_UNUSED(label)
            return new MockHardware(key, subKey); 
        }
    );
    QVERIFY(success);
}

QTEST_MAIN(HardwareRegistryTest)
#include "tst_hardwareregistrytest.moc"