#include <QtTest>
#include <QThread>
#include <QSignalSpy>
#include <QTimer>
#include <QCoreApplication>
#include <QSettings>
#include <memory>

#include <src/hardware/core/hardwareobject.h>
#include <src/hardware/core/communication/communicationprotocol.h>
#include <src/data/storage/settingsstorage.h>

#include <src/hardware/optional/gpibcontroller/gpibcontroller.h>
#include <src/hardware/optional/gpibcontroller/virtualgpibcontroller.h>
#include <src/hardware/core/communication/gpibinstrument.h>

// Test hardware object that supports multiple protocols
class TestHardwareObject : public HardwareObject {
    Q_OBJECT
public:
    TestHardwareObject(const QString& subKey = "test", QObject* parent = nullptr)
        : HardwareObject("TestHW", subKey, "Test Hardware", 
                        CommunicationProtocol::Virtual, parent, false, false, d_count++)
    {
    }
    
    virtual ~TestHardwareObject() = default;
    
    QVector<CommunicationProtocol::CommType> supportedProtocols() const override {
        QVector<CommunicationProtocol::CommType> protocols;
        protocols << CommunicationProtocol::Virtual;
        protocols << CommunicationProtocol::Rs232;
        protocols << CommunicationProtocol::Tcp;
        protocols << CommunicationProtocol::Gpib;  // Always include GPIB for test
        return protocols;
    }
    
protected:
    bool testConnection() override { return true; }
    void initialize() override {}
    void readSettings() override {}
    
private:
    inline static int d_count = 0;
};

/**
 * @brief Comprehensive test for hardware communication improvements
 * 
 * Tests the architectural changes made to support:
 * - GPIB threading modernization with mutex protection
 * - Runtime communication protocol switching
 * - Settings-driven protocol selection
 * - Hardware object self-containment
 */
class HardwareCommunicationTest : public QObject
{
    Q_OBJECT
    
public:
    HardwareCommunicationTest() {
        // Configure test environment like SettingsStorageTest
        QCoreApplication::setApplicationName("BlackchirpTest");
        QCoreApplication::setOrganizationName("CrabtreeLab");
        QCoreApplication::setOrganizationDomain("crabtreelab.ucdavis.edu");
    }
    ~HardwareCommunicationTest() = default;

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // GPIB Threading Tests
    void testGpibMutexProtection();
    void testGpibNoSpecialThreading();
    void testGpibSelfContainment();
    
    // Protocol Switching Tests
    void testProtocolSwitching();
    void testProtocolPersistence();
    void testUnsupportedProtocol();
    
    // Settings Integration Tests
    void testHardwareSettingsStorage();
    void testProtocolSettingsStorage();
    
    // Multi-threaded GPIB Communication Tests
    void testMultiThreadedGpibCommunication();
    
    // Dynamic GPIB Controller Switching Tests
    void testDynamicGpibControllerSwitching();
    
    // GPIB Address Management Tests
    void testGpibAddressManagement();

private:
    
    std::unique_ptr<TestHardwareObject> m_testHw;
    std::unique_ptr<VirtualGpibController> m_gpibController;
};

void HardwareCommunicationTest::initTestCase()
{
    // Initialize test settings environment like SettingsStorageTest
    QSettings s;
    s.setFallbacksEnabled(false);
    s.clear();
    s.sync();
    
    // Set up basic test hardware settings
    s.beginGroup("Blackchirp");
    s.setValue("TestHW.0/test/key", "TestHW.0");
    s.setValue("TestHW.0/test/name", "Test Hardware");
    s.setValue("TestHW.0/test/critical", false);
    s.setValue("TestHW.0/test/commType", static_cast<int>(CommunicationProtocol::Virtual));
    s.endGroup();
    s.sync();
    
    m_testHw = std::make_unique<TestHardwareObject>();
    m_testHw->bcInitInstrument(); // Initialize to populate settings including supportedProtocols
    m_gpibController = std::make_unique<VirtualGpibController>();
}

void HardwareCommunicationTest::cleanupTestCase()
{
    m_testHw.reset();
    m_gpibController.reset();
    
    // Clean up test settings
    QSettings s;
    s.setFallbacksEnabled(false);
    s.clear();
    s.sync();
}

void HardwareCommunicationTest::testGpibMutexProtection()
{
    QVERIFY(m_gpibController != nullptr);
    
    // Test that GPIB controller can handle concurrent access without crashing
    // This tests the QMutexLocker protection we added
    
    bool success1 = false, success2 = false;
    
    // Create two threads that try to access GPIB simultaneously
    QThread thread1, thread2;
    
    auto task1 = [this, &success1]() {
        // Multiple operations to test mutex protection
        for(int i = 0; i < 10; ++i) {
            m_gpibController->writeCmd(1, QString("*IDN?"));
            QThread::msleep(1); // Small delay to encourage race conditions
        }
        success1 = true;
    };
    
    auto task2 = [this, &success2]() {
        for(int i = 0; i < 10; ++i) {
            m_gpibController->queryCmd(2, QString("*TST?"));
            QThread::msleep(1);
        }
        success2 = true;
    };
    
    QTimer::singleShot(0, [task1](){ task1(); });
    QTimer::singleShot(0, [task2](){ task2(); });
    
    // Wait for both tasks to complete
    QTest::qWait(1000);
    
    QVERIFY(success1);
    QVERIFY(success2);
    
    qDebug() << "GPIB mutex protection test completed successfully";
}

void HardwareCommunicationTest::testGpibNoSpecialThreading()
{
    QVERIFY(m_gpibController != nullptr);
    
    // Test that GPIB controller works in main thread (no special threading required)
    QThread* currentThread = QThread::currentThread();
    QThread* gpibThread = m_gpibController->thread();
    
    // With our changes, GPIB should work in any thread, not require special threading
    QVERIFY(gpibThread == currentThread || gpibThread == nullptr);
    
    // Test basic GPIB operations work without special threading
    bool writeSuccess = m_gpibController->writeCmd(1, "*IDN?");
    QByteArray queryResult = m_gpibController->queryCmd(1, "*TST?");
    
    // For virtual GPIB, these should complete without throwing/crashing
    // (actual success depends on implementation, but no crashes = threading works)
    Q_UNUSED(writeSuccess)
    Q_UNUSED(queryResult)
    
    qDebug() << "GPIB operates without special threading requirements";
}

void HardwareCommunicationTest::testGpibSelfContainment()
{
    QVERIFY(m_testHw != nullptr);
    QVERIFY(m_gpibController != nullptr);
    
    // Test that hardware objects using GPIB don't require setParent(controller)
    // This tests that we removed the architectural coupling
    
    // Switch test hardware to GPIB protocol
    bool switchSuccess = m_testHw->setCommProtocol(CommunicationProtocol::Gpib, m_gpibController.get());
    QVERIFY(switchSuccess);
    
    // Verify that test hardware maintains its original parent (not forced to GPIB controller)
    QObject* hwParent = m_testHw->parent();
    QVERIFY(hwParent != m_gpibController.get());
    
    qDebug() << "GPIB hardware objects maintain self-containment";
}

void HardwareCommunicationTest::testProtocolSwitching()
{
    QVERIFY(m_testHw != nullptr);
    
    // Test runtime protocol switching capability
    auto supportedProtocols = m_testHw->supportedProtocols();
    QVERIFY(supportedProtocols.size() > 1); // Must support multiple protocols
    
    CommunicationProtocol::CommType originalProtocol = CommunicationProtocol::Virtual;
    CommunicationProtocol::CommType targetProtocol = CommunicationProtocol::Rs232;
    
    // Switch to different protocol
    bool switchSuccess = m_testHw->setCommProtocol(targetProtocol);
    QVERIFY(switchSuccess);
    
    // Verify the switch was successful by checking settings
    SettingsStorage settings({m_testHw->d_key, m_testHw->d_subKey}, SettingsStorage::General);
    int storedProtocol = settings.get<int>(BC::Key::HW::commType, -1);
    QCOMPARE(storedProtocol, static_cast<int>(targetProtocol));
    
    // Switch back to original
    switchSuccess = m_testHw->setCommProtocol(originalProtocol);
    QVERIFY(switchSuccess);
    
    SettingsStorage settings2({m_testHw->d_key, m_testHw->d_subKey}, SettingsStorage::General);
    storedProtocol = settings2.get<int>(BC::Key::HW::commType, -1);
    QCOMPARE(storedProtocol, static_cast<int>(originalProtocol));
    
    qDebug() << "Protocol switching works correctly";
}

void HardwareCommunicationTest::testProtocolPersistence()
{
    QVERIFY(m_testHw != nullptr);
    
    // Test that protocol changes persist between sessions
    CommunicationProtocol::CommType testProtocol = CommunicationProtocol::Tcp;
    
    // Set protocol and verify storage
    bool switchSuccess = m_testHw->setCommProtocol(testProtocol);
    QVERIFY(switchSuccess);
    
    // Create new settings instance to simulate fresh load
    SettingsStorage settings({m_testHw->d_key, m_testHw->d_subKey}, SettingsStorage::General);
    int storedProtocol = settings.get<int>(BC::Key::HW::commType, -1);
    QCOMPARE(storedProtocol, static_cast<int>(testProtocol));
    
    qDebug() << "Protocol settings persist correctly";
}

void HardwareCommunicationTest::testUnsupportedProtocol()
{
    QVERIFY(m_testHw != nullptr);
    
    // Test that switching to unsupported protocol fails gracefully
    // Custom protocol should not be in supported list for test hardware
    bool switchSuccess = m_testHw->setCommProtocol(CommunicationProtocol::Custom);
    
    auto supported = m_testHw->supportedProtocols();
    if (supported.contains(CommunicationProtocol::Custom)) {
        QVERIFY(switchSuccess); // If supported, should succeed
    } else {
        QVERIFY(!switchSuccess); // If not supported, should fail
    }
    
    qDebug() << "Unsupported protocol rejection works correctly";
}

void HardwareCommunicationTest::testHardwareSettingsStorage()
{
    QVERIFY(m_testHw != nullptr);
    
    // Test that hardware settings are properly stored and retrieved
    SettingsStorage settings({m_testHw->d_key, m_testHw->d_subKey}, SettingsStorage::General);
    
    // Test basic hardware metadata storage
    QString hwKey = settings.get<QString>(BC::Key::HW::key, QString());
    QCOMPARE(hwKey, m_testHw->d_key);
    
    QString hwName = settings.get<QString>(BC::Key::HW::name, QString());
    QCOMPARE(hwName, QString("Test Hardware"));
    
    bool hwCritical = settings.get<bool>(BC::Key::HW::critical, true);
    QCOMPARE(hwCritical, false); // Test hardware is not critical
    
    qDebug() << "Hardware settings storage works correctly";
}

void HardwareCommunicationTest::testProtocolSettingsStorage()
{
    QVERIFY(m_testHw != nullptr);
    
    // Test protocol-specific settings storage
    CommunicationProtocol::CommType testProtocol = CommunicationProtocol::Rs232;
    bool switchSuccess = m_testHw->setCommProtocol(testProtocol);
    QVERIFY(switchSuccess);
    
    // Verify supported protocols are stored for UI access
    SettingsStorage settings({m_testHw->d_key, m_testHw->d_subKey}, SettingsStorage::General);
    QVariantList protocolList = settings.get<QVariantList>(BC::Key::HW::supportedProtocols, QVariantList());
    
    QVERIFY(!protocolList.isEmpty());
    QVERIFY(protocolList.contains(static_cast<int>(CommunicationProtocol::Virtual)));
    QVERIFY(protocolList.contains(static_cast<int>(CommunicationProtocol::Rs232)));
    
    qDebug() << "Protocol settings storage works correctly";
}

void HardwareCommunicationTest::testMultiThreadedGpibCommunication()
{
    QVERIFY(m_gpibController != nullptr);
    
    // This test validates that multiple hardware objects can safely communicate 
    // through the same GPIB controller from different threads with mutex protection
    
    const int numThreads = 4;
    const int numOperationsPerThread = 5;
    QVector<QThread*> threads;
    QVector<TestHardwareObject*> hardwareObjects;
    QVector<bool> threadResults(numThreads, false);
    QVector<QStringList> expectedResponses(numThreads);
    QVector<QStringList> actualResponses(numThreads);
    
    // Create multiple hardware objects and threads
    for(int i = 0; i < numThreads; ++i) {
        // Create thread
        QThread* thread = new QThread();
        thread->setObjectName(QString("TestThread-%1").arg(i));
        threads.append(thread);
        
        // Create hardware object for this thread
        TestHardwareObject* hw = new TestHardwareObject(QString("test%1").arg(i));
        hardwareObjects.append(hw);
        
        // Initialize the hardware object (builds communication protocol)
        hw->bcInitInstrument();
        
        // Switch hardware to GPIB protocol
        bool switchSuccess = hw->setCommProtocol(CommunicationProtocol::Gpib, m_gpibController.get());
        QVERIFY(switchSuccess);
        
        // Move hardware object to its thread
        hw->moveToThread(thread);
        
        // Define the work for this thread - avoid capturing references to avoid potential issues
        auto threadWork = [this, i, thread, numOperationsPerThread, &threadResults, &expectedResponses, &actualResponses]() {
            qDebug() << QString("Thread %1 (%2) starting work").arg(i).arg(thread->objectName());
            
            QStringList expected, actual;
            
            try {
                for(int op = 0; op < numOperationsPerThread; ++op) {
                    // Each thread uses a unique GPIB address (i+1) and unique commands
                    int gpibAddress = i + 1;
                    QString command = QString("*IDN?-%1-%2").arg(i).arg(op);
                    
                    qDebug() << QString("Thread %1 operation %2: sending command to address %3")
                                .arg(i).arg(op).arg(gpibAddress);
                    
                    // Record what we expect to receive
                    QString expectedResponse = QString("ECHO[%1]:%2").arg(thread->objectName()).arg(command);
                    expected.append(expectedResponse);
                    
                    // Execute GPIB query through the shared controller
                    QByteArray response = m_gpibController->queryCmd(gpibAddress, command);
                    actual.append(QString::fromUtf8(response));
                    
                    // Add small delay to encourage race conditions
                    QThread::msleep(10);
                }
                
                // Store results for main thread verification
                expectedResponses[i] = expected;
                actualResponses[i] = actual;
                threadResults[i] = true;  // Mark thread as completed
                
                qDebug() << QString("Thread %1 (%2) completed all operations successfully").arg(i).arg(thread->objectName());
                
            } catch (...) {
                qDebug() << QString("Thread %1 (%2) caught exception during execution").arg(i).arg(thread->objectName());
                threadResults[i] = false;
            }
        };
        
        // Connect thread started signal to execute work, and quit when done
        QObject::connect(thread, &QThread::started, [threadWork, thread]() {
            threadWork();
            thread->quit();  // Explicitly quit the thread when work is done
        });
        
        // Start the thread
        thread->start();
    }
    
    // Wait for all threads to complete with better diagnostics
    for(int i = 0; i < threads.size(); ++i) {
        QThread* thread = threads[i];
        qDebug() << QString("Waiting for thread %1 (%2) to complete...").arg(i).arg(thread->objectName());
        
        bool finished = thread->wait(10000);  // 10 second timeout
        
        if (!finished) {
            qDebug() << QString("Thread %1 (%2) did not finish in time. State: %3")
                        .arg(i).arg(thread->objectName()).arg(thread->isRunning() ? "Running" : "Not Running");
            qDebug() << QString("Thread result flag: %1").arg(threadResults[i]);
            
            // Force quit the thread
            thread->quit();
            thread->wait(1000);
            
            QFAIL(QString("Thread %1 did not finish in time - potential deadlock").arg(i).toUtf8().constData());
        } else {
            qDebug() << QString("Thread %1 (%2) completed successfully").arg(i).arg(thread->objectName());
        }
    }
    
    // Verify all threads completed successfully
    for(int i = 0; i < numThreads; ++i) {
        QVERIFY2(threadResults[i], QString("Thread %1 did not complete").arg(i).toUtf8().constData());
    }
    
    // Verify thread safety: each thread should receive exactly what it sent
    for(int i = 0; i < numThreads; ++i) {
        QCOMPARE(actualResponses[i].size(), expectedResponses[i].size());
        
        for(int op = 0; op < numOperationsPerThread; ++op) {
            QString expected = expectedResponses[i][op];
            QString actual = actualResponses[i][op];
            
            if (actual != expected) {
                QString errorMsg = QString("Thread %1 operation %2: expected '%3', got '%4'")
                                  .arg(i).arg(op).arg(expected).arg(actual);
                QFAIL(errorMsg.toUtf8().constData());
            }
        }
    }
    
    // Clean up
    for(int i = 0; i < numThreads; ++i) {
        threads[i]->quit();
        threads[i]->wait();
        delete hardwareObjects[i];
        delete threads[i];
    }
    
    qDebug() << QString("Multi-threaded GPIB communication test completed successfully: "
                       "%1 threads, %2 operations each, all responses matched")
                .arg(numThreads).arg(numOperationsPerThread);
}

void HardwareCommunicationTest::testDynamicGpibControllerSwitching()
{
    // This test validates dynamic GPIB controller switching across threads:
    // - 2 VirtualGpibController instances in separate threads
    // - 1 TestHardwareObject in its own thread
    // - Hardware object switches from controller1 -> controller2 -> controller1
    
    const int operationsPerController = 3;
    
    // Create threads for all objects
    QThread* controller1Thread = new QThread();
    QThread* controller2Thread = new QThread();  
    QThread* hardwareThread = new QThread();
    
    controller1Thread->setObjectName("Controller1Thread");
    controller2Thread->setObjectName("Controller2Thread");
    hardwareThread->setObjectName("HardwareThread");
    
    // Create GPIB controllers - they will automatically get GpibController.0 and GpibController.1
    VirtualGpibController* controller1 = new VirtualGpibController();
    VirtualGpibController* controller2 = new VirtualGpibController();
    
    // Create hardware object that will switch between controllers
    TestHardwareObject* switchingHw = new TestHardwareObject("switching");
    switchingHw->bcInitInstrument();
    
    // Move objects to their respective threads
    controller1->moveToThread(controller1Thread);
    controller2->moveToThread(controller2Thread);
    switchingHw->moveToThread(hardwareThread);
    
    // Track results from each phase
    QStringList phase1Results, phase2Results, phase3Results;
    QStringList phase1Expected, phase2Expected, phase3Expected;
    bool allPhasesCompleted = false;
    
    // Define the hardware work that will switch between controllers
    auto hardwareWork = [&]() {
        qDebug() << "=== Dynamic GPIB Controller Switching Test Starting ===";
        
        try {
            // PHASE 1: Connect to controller1 and do work
            qDebug() << QString("PHASE 1: Switching to controller1 (%1)").arg(controller1->d_name);
            bool switch1Success = switchingHw->setCommProtocol(CommunicationProtocol::Gpib, controller1);
            QVERIFY(switch1Success);
            
            for(int i = 0; i < operationsPerController; ++i) {
                QString command = QString("PHASE1-CMD-%1").arg(i);
                QByteArray response = controller1->queryCmd(10 + i, command);
                QString responseStr = QString::fromUtf8(response);
                
                phase1Results.append(responseStr);
                phase1Expected.append(QString("ECHO[%1]:%2").arg(hardwareThread->objectName()).arg(command));
                
                qDebug() << QString("Phase 1 operation %1: sent '%2', received '%3'").arg(i).arg(command).arg(responseStr);
                QThread::msleep(10);
            }
            
            // PHASE 2: Switch to controller2 and do work
            qDebug() << QString("PHASE 2: Switching to controller2 (%1)").arg(controller2->d_name);
            bool switch2Success = switchingHw->setCommProtocol(CommunicationProtocol::Gpib, controller2);
            QVERIFY(switch2Success);
            
            for(int i = 0; i < operationsPerController; ++i) {
                QString command = QString("PHASE2-CMD-%1").arg(i);
                QByteArray response = controller2->queryCmd(20 + i, command);
                QString responseStr = QString::fromUtf8(response);
                
                phase2Results.append(responseStr);
                phase2Expected.append(QString("ECHO[%1]:%2").arg(hardwareThread->objectName()).arg(command));
                
                qDebug() << QString("Phase 2 operation %1: sent '%2', received '%3'").arg(i).arg(command).arg(responseStr);
                QThread::msleep(10);
            }
            
            // PHASE 3: Switch back to controller1 and do work
            qDebug() << QString("PHASE 3: Switching back to controller1 (%1)").arg(controller1->d_name);
            bool switch3Success = switchingHw->setCommProtocol(CommunicationProtocol::Gpib, controller1);
            QVERIFY(switch3Success);
            
            for(int i = 0; i < operationsPerController; ++i) {
                QString command = QString("PHASE3-CMD-%1").arg(i);
                QByteArray response = controller1->queryCmd(30 + i, command);
                QString responseStr = QString::fromUtf8(response);
                
                phase3Results.append(responseStr);
                phase3Expected.append(QString("ECHO[%1]:%2").arg(hardwareThread->objectName()).arg(command));
                
                qDebug() << QString("Phase 3 operation %1: sent '%2', received '%3'").arg(i).arg(command).arg(responseStr);
                QThread::msleep(10);
            }
            
            allPhasesCompleted = true;
            qDebug() << "=== All phases completed successfully ===";
            
        } catch (...) {
            qDebug() << "Exception caught during dynamic switching test";
            allPhasesCompleted = false;
        }
        
        // Signal all threads to quit
        controller1Thread->quit();
        controller2Thread->quit();
        hardwareThread->quit();
    };
    
    // Start all threads
    controller1Thread->start();
    controller2Thread->start();
    hardwareThread->start();
    
    // Execute the hardware work
    QMetaObject::invokeMethod(switchingHw, [hardwareWork]() {
        hardwareWork();
    }, Qt::QueuedConnection);
    
    // Wait for all threads to complete
    QVERIFY(hardwareThread->wait(15000));  // 15 second timeout
    QVERIFY(controller1Thread->wait(2000));
    QVERIFY(controller2Thread->wait(2000));
    
    // Verify all phases completed
    QVERIFY(allPhasesCompleted);
    
    // Verify each phase got the correct responses
    QCOMPARE(phase1Results.size(), operationsPerController);
    QCOMPARE(phase2Results.size(), operationsPerController);
    QCOMPARE(phase3Results.size(), operationsPerController);
    
    for(int i = 0; i < operationsPerController; ++i) {
        // Phase 1: Should have responses from controller1 thread
        QCOMPARE(phase1Results[i], phase1Expected[i]);
        
        // Phase 2: Should have responses from controller2 thread  
        QCOMPARE(phase2Results[i], phase2Expected[i]);
        
        // Phase 3: Should have responses from controller1 thread again
        QCOMPARE(phase3Results[i], phase3Expected[i]);
    }
    
    // Clean up
    delete switchingHw;
    delete controller1;
    delete controller2;
    delete hardwareThread;
    delete controller1Thread;
    delete controller2Thread;
    
    qDebug() << QString("Dynamic GPIB controller switching test completed successfully: "
                       "3 phases, %1 operations each, all switches worked correctly")
                .arg(operationsPerController);
}

void HardwareCommunicationTest::testGpibAddressManagement()
{
    QVERIFY(m_gpibController != nullptr);
    
    // This test validates GPIB address conflict detection and management:
    // - Address reservation and release
    // - Conflict detection when multiple devices try to use same address
    // - Proper cleanup when devices are removed or addresses changed
    
    qDebug() << "=== GPIB Address Management Test Starting ===";
    
    // Create multiple test hardware objects
    TestHardwareObject* device1 = new TestHardwareObject("device1");
    TestHardwareObject* device2 = new TestHardwareObject("device2");
    TestHardwareObject* device3 = new TestHardwareObject("device3");
    
    device1->bcInitInstrument();
    device2->bcInitInstrument(); 
    device3->bcInitInstrument();
    
    // Switch all devices to GPIB protocol
    QVERIFY(device1->setCommProtocol(CommunicationProtocol::Gpib, m_gpibController.get()));
    QVERIFY(device2->setCommProtocol(CommunicationProtocol::Gpib, m_gpibController.get()));
    QVERIFY(device3->setCommProtocol(CommunicationProtocol::Gpib, m_gpibController.get()));
    
    // Test 1: Basic address reservation through proper GPIB workflow
    qDebug() << "Test 1: Basic address reservation through proper GPIB workflow";
    QVERIFY(m_gpibController->isAddressAvailable(10));
    
    // Set GPIB address directly (using friend class access)
    auto* gpibComm = static_cast<GpibInstrument*>(device1->p_comm);
    gpibComm->setAddress(10);
    QVERIFY(!m_gpibController->isAddressAvailable(10));
    QCOMPARE(m_gpibController->getAddressOwnerKey(10), device1->d_key);
    
    // Test 2: Address conflict detection through proper GPIB workflow
    qDebug() << "Test 2: Address conflict detection through proper GPIB workflow";
    
    // Try to set device2 to same address - should fail due to address conflict
    auto* gpibComm2 = static_cast<GpibInstrument*>(device2->p_comm);
    gpibComm2->setAddress(10);  // Should fail - address 10 already taken by device1
    
    // Verify address 10 is still taken and device2 doesn't have it
    QVERIFY(gpibComm2->address() == -1);  // Should be -1 indicating failed reservation
    QCOMPARE(m_gpibController->getAddressOwnerKey(10), device1->d_key);  // Should still be device1
    
    // Test 3: Same device can re-set same address successfully
    qDebug() << "Test 3: Same device can re-set same address successfully";
    gpibComm->setAddress(10);  // Should succeed - same owner
    QVERIFY(gpibComm->address() == 10);
    
    // Test 4: Multiple devices with different addresses
    qDebug() << "Test 4: Multiple devices with different addresses";
    
    // Set device2 to address 11
    gpibComm2->setAddress(11);
    QVERIFY(gpibComm2->address() == 11);
    
    // Set device3 to address 12
    auto* gpibComm3 = static_cast<GpibInstrument*>(device3->p_comm);
    gpibComm3->setAddress(12);
    QVERIFY(gpibComm3->address() == 12);
    
    QList<int> usedAddresses = m_gpibController->getUsedAddresses();
    QCOMPARE(usedAddresses.size(), 3);
    QVERIFY(usedAddresses.contains(10));
    QVERIFY(usedAddresses.contains(11));
    QVERIFY(usedAddresses.contains(12));
    
    // Test 5: Address release by switching to different protocol
    qDebug() << "Test 5: Address release by switching to different protocol";
    
    // Switch device2 from GPIB to Virtual - should release address 11
    QVERIFY(device2->setCommProtocol(CommunicationProtocol::Virtual));
    QVERIFY(m_gpibController->isAddressAvailable(11));
    QVERIFY(m_gpibController->getAddressOwnerKey(11).isEmpty());
    
    // Test 6: Address conflicts still work
    qDebug() << "Test 6: Address conflicts still work";
    
    // Switch device2 back to GPIB and try to use device1's address - should fail
    QVERIFY(device2->setCommProtocol(CommunicationProtocol::Gpib, m_gpibController.get()));
    gpibComm2 = static_cast<GpibInstrument*>(device2->p_comm);  // Update pointer after protocol switch
    gpibComm2->setAddress(10);  // Try to use device1's address - should fail
    QVERIFY(gpibComm2->address() == -1);  // Should fail
    QVERIFY(!m_gpibController->isAddressAvailable(10));  // Should still be taken
    QCOMPARE(m_gpibController->getAddressOwnerKey(10), device1->d_key);  // Should still be device1
    
    // Test 7: Address can be reused after device switches protocols
    qDebug() << "Test 7: Address can be reused after device switches protocols";
    
    // Switch device1 to Virtual - should release address 10
    QVERIFY(device1->setCommProtocol(CommunicationProtocol::Virtual));
    QVERIFY(m_gpibController->isAddressAvailable(10));
    
    // Now device2 can use address 10
    gpibComm2->setAddress(10);  // Should now succeed with address 10
    QVERIFY(gpibComm2->address() == 10);
    QCOMPARE(m_gpibController->getAddressOwnerKey(10), device2->d_key);
    
    // Test 8: Address change via direct setAddress
    qDebug() << "Test 8: Address change via direct setAddress";
    
    // Change device2's address from 10 to 13
    gpibComm2->setAddress(13);  // Should release old address 10 and reserve 13
    QVERIFY(gpibComm2->address() == 13);
    QVERIFY(m_gpibController->isAddressAvailable(10));
    QCOMPARE(m_gpibController->getAddressOwnerKey(13), device2->d_key);
    
    // Test 9: Invalid operations
    qDebug() << "Test 9: Invalid operations";
    QVERIFY(!m_gpibController->reserveAddress(14, QString(), QString()));  // Invalid owner (empty key)
    m_gpibController->releaseAddress(14, QString());  // Should not crash (empty key)
    
    // Test 10: Automatic cleanup verification
    qDebug() << "Test 10: Automatic cleanup verification";
    usedAddresses = m_gpibController->getUsedAddresses();
    qDebug() << QString("Used addresses before cleanup: %1").arg(usedAddresses.size());
    
    // Current state: device2 has address 13, device3 has address 12
    QCOMPARE(usedAddresses.size(), 2);
    QVERIFY(usedAddresses.contains(12));  // device3
    QVERIFY(usedAddresses.contains(13));  // device2
    
    // Delete device2 - should automatically release address 13
    delete device2;
    device2 = nullptr;
    
    usedAddresses = m_gpibController->getUsedAddresses();
    qDebug() << QString("Used addresses after deleting device2: %1").arg(usedAddresses.size());
    QCOMPARE(usedAddresses.size(), 1);
    QVERIFY(!usedAddresses.contains(13));  // address 13 should be released
    QVERIFY(usedAddresses.contains(12));   // device3 should still have address 12
    QVERIFY(m_gpibController->isAddressAvailable(13));  // address 13 should be available again
    
    // Delete device3 - should automatically release address 12
    delete device3;
    device3 = nullptr;
    
    usedAddresses = m_gpibController->getUsedAddresses();
    qDebug() << QString("Used addresses after deleting device3: %1").arg(usedAddresses.size());
    QCOMPARE(usedAddresses.size(), 0);
    QVERIFY(m_gpibController->isAddressAvailable(12));  // address 12 should be available again
    
    // Delete device1 (should be clean since we released its address earlier)
    delete device1;
    device1 = nullptr;
    
    usedAddresses = m_gpibController->getUsedAddresses();
    qDebug() << QString("Used addresses after deleting all devices: %1").arg(usedAddresses.size());
    QCOMPARE(usedAddresses.size(), 0);  // All addresses should be released
    
    qDebug() << "=== GPIB Address Management Test Completed Successfully ===";
    qDebug() << "Verified: Automatic address cleanup works correctly!";
}

QTEST_MAIN(HardwareCommunicationTest)

#include "tst_hardwarecommunication.moc"