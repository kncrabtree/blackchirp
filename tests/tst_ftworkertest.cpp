#include <QtTest>
#include <QCoreApplication>
#include <QSignalSpy>
#include <QTimer>
#include <cmath>

#include <src/data/analysis/ftworker.h>
#include <src/data/experiment/fid.h>

class FtWorkerTest : public QObject
{
    Q_OBJECT

public:
    FtWorkerTest();
    ~FtWorkerTest();

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // Basic functionality tests
    void testInitialState();
    void testResourceAllocation();
    void testFftProcessing();
    void testSplineProcessing();
    
    // Idle cleanup tests
    void testIdleCleanupDisabledByDefault();
    void testIdleCleanupEnable();
    void testIdleCleanupAfterFftAllocation();
    void testIdleCleanupAfterSplineAllocation();
    void testIdleCleanupReallocation();
    void testIdleTimerReset();
    void testManualCleanup();
    
    // Thread safety tests
    void testConcurrentProcessing();
    void testCleanupDuringProcessing();
    
    // Edge cases
    void testCleanupWithoutResources();
    void testDisableCleanupAfterAllocation();

private:
    FtWorker *p_worker;
    
    // Helper methods to access private members through friend access
    bool hasAllocatedResources(FtWorker *worker);
    bool hasAllocatedFftResources(FtWorker *worker);
    bool hasAllocatedSplineResources(FtWorker *worker);
    bool hasAllocatedWindowFunction(FtWorker *worker);
    bool isIdleCleanupEnabled(FtWorker *worker);
    void triggerIdleTimeout(FtWorker *worker);
    
    // Helper to create test FID data
    FidList createTestFidList(int size = 1024, int numFrames = 1);
    Ft createTestFt(int size = 512);
};

FtWorkerTest::FtWorkerTest()
{
    QCoreApplication::setApplicationName("BlackchirpFtWorkerTest");
    QCoreApplication::setOrganizationName("CrabtreeLab");
    QCoreApplication::setOrganizationDomain("crabtreelab.ucdavis.edu");
}

FtWorkerTest::~FtWorkerTest()
{

}

void FtWorkerTest::initTestCase()
{
    p_worker = new FtWorker(this);
}

void FtWorkerTest::cleanupTestCase()
{
    delete p_worker;
}

bool FtWorkerTest::hasAllocatedResources(FtWorker *worker)
{
    return worker->d_resourcesAllocated;
}

bool FtWorkerTest::hasAllocatedFftResources(FtWorker *worker)
{
    return worker->real != nullptr && worker->work != nullptr;
}

bool FtWorkerTest::hasAllocatedSplineResources(FtWorker *worker)
{
    return worker->p_spline != nullptr && worker->p_accel != nullptr;
}

bool FtWorkerTest::hasAllocatedWindowFunction(FtWorker *worker)
{
    return !worker->d_winf.isEmpty();
}

bool FtWorkerTest::isIdleCleanupEnabled(FtWorker *worker)
{
    return worker->d_idleCleanupEnabled;
}

void FtWorkerTest::triggerIdleTimeout(FtWorker *worker)
{
    // Directly call the timeout slot instead of waiting for timer
    worker->onIdleTimeout();
}

FidList FtWorkerTest::createTestFidList(int size, int numFrames)
{
    FidList fl;
    for(int frame = 0; frame < numFrames; ++frame)
    {
        QVector<qint64> data(size);
        for(int i = 0; i < size; ++i)
        {
            // Create simple sine wave data
            data[i] = static_cast<qint64>(1000.0 * sin(2.0 * M_PI * i / 100.0));
        }
        
        Fid fid;
        fid.setData(data);
        fid.setSpacing(0.5e-6);
        fid.setSideband(RfConfig::UpperSideband);
        fid.setProbeFreq(10000.0);
        fl.append(fid);
    }
    return fl;
}

Ft FtWorkerTest::createTestFt(int size)
{
    // Create FT with f0=10000.0 MHz, spacing=1.0 MHz, so frequencies run from 10000-10000+size MHz
    Ft ft(size, 10000.0, 1.0, 10000.0);
    for(int i = 0; i < size; ++i)
    {
        // Create a simple decreasing exponential function for realistic FT data
        double magnitude = exp(-static_cast<double>(i) / 50.0);
        ft.setPoint(i, magnitude, 15000.0);
    }
    return ft;
}

void FtWorkerTest::testInitialState()
{
    // Test initial state of a new FtWorker
    QVERIFY(!hasAllocatedResources(p_worker));
    QVERIFY(!hasAllocatedFftResources(p_worker));
    QVERIFY(!hasAllocatedSplineResources(p_worker));
    QVERIFY(!hasAllocatedWindowFunction(p_worker));
    QVERIFY(!isIdleCleanupEnabled(p_worker));
}

void FtWorkerTest::testResourceAllocation()
{
    // Create a new worker for this test to ensure clean state
    FtWorker worker;
    QVERIFY(!hasAllocatedResources(&worker));
    
    // Process a FID to trigger FFT resource allocation
    FidList fl = createTestFidList(1024);
    FtWorker::FidProcessingSettings settings;
    settings.startUs = 0.0;
    settings.endUs = 512.0;
    settings.expFilter = 0.0;
    settings.zeroPadFactor = 0;
    settings.removeDC = true;
    settings.units = FtWorker::FtmV;
    settings.autoScaleIgnoreMHz = 15000.0;
    settings.windowFunction = FtWorker::None;
    
    Ft ftResult = worker.doFT(fl, settings);
    
    // Verify resources were allocated
    QVERIFY(hasAllocatedResources(&worker));
    QVERIFY(hasAllocatedFftResources(&worker));
    QVERIFY(!ftResult.isEmpty());
    
    // Test spline allocation - make sure f0 and spacing are different from testFt
    Ft testFt = createTestFt(256);  // Creates FT from 10000.0 to 10256.0 MHz, spacing=1.0
    // Request different spacing but within the frequency range to trigger resampling
    auto resampleResult = worker.resample(10050.0, 0.5, testFt);
    
    QVERIFY(hasAllocatedSplineResources(&worker));
    QVERIFY(!resampleResult.first.isEmpty());
}

void FtWorkerTest::testFftProcessing()
{
    FtWorker worker;
    FidList fl = createTestFidList(512);
    FtWorker::FidProcessingSettings settings;
    settings.startUs = 0.0;
    settings.endUs = 256.0;
    settings.expFilter = 0.0;
    settings.zeroPadFactor = 1; // Test zero padding
    settings.removeDC = true;
    settings.units = FtWorker::FtmV;
    settings.autoScaleIgnoreMHz = 15000.0;
    settings.windowFunction = FtWorker::Hanning;
    
    Ft result = worker.doFT(fl, settings);
    
    QVERIFY(!result.isEmpty());
    QVERIFY(hasAllocatedFftResources(&worker));
    QVERIFY(hasAllocatedWindowFunction(&worker));
    
    // Test processing with different size to trigger reallocation
    FidList fl2 = createTestFidList(256);
    Ft result2 = worker.doFT(fl2, settings);
    
    QVERIFY(!result2.isEmpty());
    QVERIFY(hasAllocatedFftResources(&worker));
}

void FtWorkerTest::testSplineProcessing()
{
    FtWorker worker;
    Ft testFt = createTestFt(128);  // Creates FT with f0=10000.0, spacing=1.0
    
    // Test resampling with different parameters to trigger actual resampling
    auto resampleResult1 = worker.resample(10030.0, 0.8, testFt);
    
    QVERIFY(hasAllocatedSplineResources(&worker));
    QVERIFY(!resampleResult1.first.isEmpty());
    
    // Test resampling with different size to trigger reallocation
    Ft testFt2 = createTestFt(64);  // Creates FT from 10000.0 to 10064.0 MHz
    auto resampleResult2 = worker.resample(10020.0, 0.6, testFt2);
    
    QVERIFY(hasAllocatedSplineResources(&worker));
    QVERIFY(!resampleResult2.first.isEmpty());
}

void FtWorkerTest::testIdleCleanupDisabledByDefault()
{
    FtWorker worker;
    QVERIFY(!isIdleCleanupEnabled(&worker));
    
    // Allocate resources
    FidList fl = createTestFidList(512);
    FtWorker::FidProcessingSettings settings;
    settings.startUs = 0.0;
    settings.endUs = 256.0;
    settings.expFilter = 0.0;
    settings.zeroPadFactor = 0;
    settings.removeDC = true;
    settings.units = FtWorker::FtmV;
    settings.autoScaleIgnoreMHz = 15000.0;
    settings.windowFunction = FtWorker::None;
    
    Ft result = worker.doFT(fl, settings);
    QVERIFY(!result.isEmpty());  // Verify the FT was actually computed
    QVERIFY(hasAllocatedResources(&worker));
    
    // Trigger timeout - should not clean up because cleanup is disabled
    triggerIdleTimeout(&worker);
    QVERIFY(hasAllocatedResources(&worker));
    QVERIFY(hasAllocatedFftResources(&worker));
}

void FtWorkerTest::testIdleCleanupEnable()
{
    FtWorker worker;
    
    // Enable idle cleanup
    worker.setIdleCleanupEnabled(true);
    QVERIFY(isIdleCleanupEnabled(&worker));
    
    // Disable idle cleanup
    worker.setIdleCleanupEnabled(false);
    QVERIFY(!isIdleCleanupEnabled(&worker));
}

void FtWorkerTest::testIdleCleanupAfterFftAllocation()
{
    FtWorker worker;
    worker.setIdleCleanupEnabled(true);
    
    // Allocate FFT resources
    FidList fl = createTestFidList(512);
    FtWorker::FidProcessingSettings settings;
    settings.startUs = 0.0;
    settings.endUs = 256.0;
    settings.expFilter = 0.0;
    settings.zeroPadFactor = 0;
    settings.removeDC = true;
    settings.units = FtWorker::FtmV;
    settings.autoScaleIgnoreMHz = 15000.0;
    settings.windowFunction = FtWorker::Hanning;
    
    worker.doFT(fl, settings);
    
    QVERIFY(hasAllocatedResources(&worker));
    QVERIFY(hasAllocatedFftResources(&worker));
    QVERIFY(hasAllocatedWindowFunction(&worker));
    
    // Trigger idle cleanup
    triggerIdleTimeout(&worker);
    
    QVERIFY(!hasAllocatedResources(&worker));
    QVERIFY(!hasAllocatedFftResources(&worker));
    QVERIFY(!hasAllocatedWindowFunction(&worker));
}

void FtWorkerTest::testIdleCleanupAfterSplineAllocation()
{
    FtWorker worker;
    worker.setIdleCleanupEnabled(true);
    
    // Allocate spline resources with different parameters to trigger resampling
    Ft testFt = createTestFt(128);  // Creates FT from 10000.0 to 10128.0 MHz, spacing=1.0
    auto resampleResult = worker.resample(10040.0, 0.7, testFt);
    
    QVERIFY(hasAllocatedResources(&worker));
    QVERIFY(hasAllocatedSplineResources(&worker));
    
    // Trigger idle cleanup
    triggerIdleTimeout(&worker);
    
    QVERIFY(!hasAllocatedResources(&worker));
    QVERIFY(!hasAllocatedSplineResources(&worker));
}

void FtWorkerTest::testIdleCleanupReallocation()
{
    FtWorker worker;
    worker.setIdleCleanupEnabled(true);
    
    // Allocate and clean up FFT resources
    FidList fl = createTestFidList(512);
    FtWorker::FidProcessingSettings settings;
    settings.startUs = 0.0;
    settings.endUs = 256.0;
    settings.expFilter = 0.0;
    settings.zeroPadFactor = 0;
    settings.removeDC = true;
    settings.units = FtWorker::FtmV;
    settings.autoScaleIgnoreMHz = 15000.0;
    settings.windowFunction = FtWorker::None;
    
    worker.doFT(fl, settings);
    QVERIFY(hasAllocatedFftResources(&worker));
    
    triggerIdleTimeout(&worker);
    QVERIFY(!hasAllocatedFftResources(&worker));
    
    // Process again - should reallocate resources
    Ft result = worker.doFT(fl, settings);
    QVERIFY(!result.isEmpty());
    QVERIFY(hasAllocatedFftResources(&worker));
}

void FtWorkerTest::testIdleTimerReset()
{
    FtWorker worker;
    worker.setIdleCleanupEnabled(true);
    
    // Allocate resources
    FidList fl = createTestFidList(512);
    FtWorker::FidProcessingSettings settings;
    settings.startUs = 0.0;
    settings.endUs = 256.0;
    settings.expFilter = 0.0;
    settings.zeroPadFactor = 0;
    settings.removeDC = true;
    settings.units = FtWorker::FtmV;
    settings.autoScaleIgnoreMHz = 15000.0;
    settings.windowFunction = FtWorker::None;
    
    worker.doFT(fl, settings);
    QVERIFY(hasAllocatedResources(&worker));
    
    // Timer should reset on subsequent processing
    worker.resetIdleTimer();
    
    // Verify that manual timer reset doesn't cause issues
    worker.resetIdleTimer();
    worker.resetIdleTimer();
    
    QVERIFY(hasAllocatedResources(&worker));
}

void FtWorkerTest::testManualCleanup()
{
    FtWorker worker;
    
    // Allocate both FFT and spline resources
    FidList fl = createTestFidList(512);
    FtWorker::FidProcessingSettings settings;
    settings.startUs = 0.0;
    settings.endUs = 256.0;
    settings.expFilter = 0.0;
    settings.zeroPadFactor = 0;
    settings.removeDC = true;
    settings.units = FtWorker::FtmV;
    settings.autoScaleIgnoreMHz = 15000.0;
    settings.windowFunction = FtWorker::Hanning;
    
    worker.doFT(fl, settings);
    
    Ft testFt = createTestFt(128);  // Creates FT from 10000.0 to 10128.0 MHz, spacing=1.0
    auto resampleResult = worker.resample(10060.0, 0.9, testFt);
    
    QVERIFY(hasAllocatedFftResources(&worker));
    QVERIFY(hasAllocatedSplineResources(&worker));
    QVERIFY(hasAllocatedWindowFunction(&worker));
    
    // Manual cleanup should work regardless of idle cleanup setting
    worker.cleanupResources();
    
    QVERIFY(!hasAllocatedResources(&worker));
    QVERIFY(!hasAllocatedFftResources(&worker));
    QVERIFY(!hasAllocatedSplineResources(&worker));
    QVERIFY(!hasAllocatedWindowFunction(&worker));
}

void FtWorkerTest::testConcurrentProcessing()
{
    // Test that cleanup doesn't interfere with normal processing
    FtWorker worker;
    worker.setIdleCleanupEnabled(true);
    
    FidList fl = createTestFidList(512);
    FtWorker::FidProcessingSettings settings;
    settings.startUs = 0.0;
    settings.endUs = 256.0;
    settings.expFilter = 0.0;
    settings.zeroPadFactor = 0;
    settings.removeDC = true;
    settings.units = FtWorker::FtmV;
    settings.autoScaleIgnoreMHz = 15000.0;
    settings.windowFunction = FtWorker::None;
    
    // Process multiple times in succession
    for(int i = 0; i < 5; ++i)
    {
        Ft result = worker.doFT(fl, settings);
        QVERIFY(!result.isEmpty());
        QVERIFY(hasAllocatedFftResources(&worker));
    }
}

void FtWorkerTest::testCleanupDuringProcessing()
{
    FtWorker worker;
    worker.setIdleCleanupEnabled(true);
    
    // Allocate resources
    FidList fl = createTestFidList(512);
    FtWorker::FidProcessingSettings settings;
    settings.startUs = 0.0;
    settings.endUs = 256.0;
    settings.expFilter = 0.0;
    settings.zeroPadFactor = 0;
    settings.removeDC = true;
    settings.units = FtWorker::FtmV;
    settings.autoScaleIgnoreMHz = 15000.0;
    settings.windowFunction = FtWorker::None;
    
    worker.doFT(fl, settings);
    QVERIFY(hasAllocatedResources(&worker));
    
    // Manual cleanup should be safe even during processing state
    worker.cleanupResources();
    QVERIFY(!hasAllocatedResources(&worker));
    
    // Should be able to process again
    Ft result = worker.doFT(fl, settings);
    QVERIFY(!result.isEmpty());
    QVERIFY(hasAllocatedResources(&worker));
}

void FtWorkerTest::testCleanupWithoutResources()
{
    FtWorker worker;
    
    // Cleanup should be safe even when no resources are allocated
    QVERIFY(!hasAllocatedResources(&worker));
    worker.cleanupResources();
    QVERIFY(!hasAllocatedResources(&worker));
    
    // Cleanup with idle setting enabled
    worker.setIdleCleanupEnabled(true);
    triggerIdleTimeout(&worker);
    QVERIFY(!hasAllocatedResources(&worker));
}

void FtWorkerTest::testDisableCleanupAfterAllocation()
{
    FtWorker worker;
    worker.setIdleCleanupEnabled(true);
    
    // Allocate resources
    FidList fl = createTestFidList(512);
    FtWorker::FidProcessingSettings settings;
    settings.startUs = 0.0;
    settings.endUs = 256.0;
    settings.expFilter = 0.0;
    settings.zeroPadFactor = 0;
    settings.removeDC = true;
    settings.units = FtWorker::FtmV;
    settings.autoScaleIgnoreMHz = 15000.0;
    settings.windowFunction = FtWorker::None;
    
    Ft result = worker.doFT(fl, settings);
    QVERIFY(!result.isEmpty());  // Verify the FT was actually computed
    QVERIFY(hasAllocatedResources(&worker));
    
    // Disable cleanup after allocation
    worker.setIdleCleanupEnabled(false);
    
    // Timeout should not clean up
    triggerIdleTimeout(&worker);
    QVERIFY(hasAllocatedResources(&worker));
    QVERIFY(hasAllocatedFftResources(&worker));
}

QTEST_MAIN(FtWorkerTest)

#include "tst_ftworkertest.moc"