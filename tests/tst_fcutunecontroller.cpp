#include <QtTest>

#include <cmath>

#include <data/lif/fcutunecontroller.h>

using namespace Qt::Literals::StringLiterals;
using namespace BC::FcuTune;

/*!
 * Drives FcuTuneController with synthetic interleaved LIF/reference
 * waveforms, playing the part of HardwareManager by answering each
 * requestTrim() with trimUpdated().
 */
class FcuTuneControllerTest : public QObject
{
    Q_OBJECT

private slots:
    void testSuccessAppliesCenter();
    void testClippedRestoresStart();
    void testMoveFailureRestoresStart();
    void testAbortRestoresStart();
    void testRefDisabledRejected();
    void testWaveformsIgnoredWhileMoving();
    void testSaturationLevelRejects();
    void testRecenterFindsPeakOutsideWindow();
    void testRecenterLimit();

private:
    static constexpr int recordLength = 10;
    static constexpr auto stage = "LaserFreqConversionStage.fcu"_L1;

    static LifDigitizerConfig makeConfig(bool refEnabled = true)
    {
        LifDigitizerConfig c(u"LifDigitizer.test"_s);
        c.d_recordLength = recordLength;
        c.d_bytesPerPoint = 1;
        c.d_sampleRate = 1e9;
        c.d_numAverages = 1;
        c.d_lifChannel = 1;
        c.d_refChannel = 2;
        c.d_refEnabled = refEnabled;
        c.d_channelOrder = LifDigitizerConfig::Interleaved;
        DigitizerConfig::AnalogChannel ch;
        ch.enabled = true;
        ch.fullScale = 1.0;
        c.d_analogChannels.insert({1,ch});
        c.d_analogChannels.insert({2,ch});
        return c;
    }

    static LifTrace::LifProcSettings makeProc()
    {
        LifTrace::LifProcSettings p;
        p.lifGateStart = 2;
        p.lifGateEnd = 7;
        p.refGateStart = 2;
        p.refGateEnd = 7;
        return p;
    }

    //! Interleaved record whose reference pulse has height \a refHeight inside the gate.
    static QVector<qint8> makeWaveform(int refHeight)
    {
        QVector<qint8> b(2*recordLength,0);
        for(int i=2; i<=7; i++)
            b[2*i+1] = static_cast<qint8>(refHeight);
        return b;
    }

    static int refHeightAt(double trim, double center, double peak)
    {
        auto a = M_PI*(trim - center)/4000.0;
        auto s = std::abs(a) < 1e-12 ? 1.0 : std::sin(a)/a;
        return static_cast<int>(std::lround(peak*s*s));
    }

    static Settings makeSettings()
    {
        Settings s;
        s.halfWidth = 3000.0;
        s.points = 13;
        s.waveformsPerPoint = 3;
        s.discardPerPoint = 1;
        return s;
    }
};

void FcuTuneControllerTest::testSuccessAppliesCenter()
{
    FcuTuneController c;
    double held = 0.0;
    connect(&c,&FcuTuneController::requestTrim,this,[&](const QString &k, double t){
        held = t;
        c.trimUpdated(k,t,1,true);
    });
    QSignalSpy finished(&c,&FcuTuneController::finished);
    QSignalSpy points(&c,&FcuTuneController::pointComplete);

    QVERIFY(c.start(stage,0.0,makeSettings(),makeConfig(),makeProc()));
    while(c.isRunning())
        c.processWaveform(makeWaveform(refHeightAt(held,-700.0,100.0)));

    QCOMPARE(finished.count(),1);
    QCOMPARE(points.count(),13);
    auto r = finished.at(0).at(0).value<Result>();
    QCOMPARE(r.status,Status::Success);
    QVERIFY2(std::abs(r.center + 700.0) < 150.0, qPrintable(u"center %1"_s.arg(r.center)));
    QCOMPARE(finished.at(0).at(1).toDouble(),std::round(r.center));
    QVERIFY(finished.at(0).at(2).toBool());
    QCOMPARE(held,std::round(r.center));
}

void FcuTuneControllerTest::testClippedRestoresStart()
{
    FcuTuneController c;
    double held = 0.0;
    connect(&c,&FcuTuneController::requestTrim,this,[&](const QString &k, double t){
        held = t;
        c.trimUpdated(k,t,1,true);
    });
    QSignalSpy finished(&c,&FcuTuneController::finished);

    QVERIFY(c.start(stage,250.0,makeSettings(),makeConfig(),makeProc()));
    while(c.isRunning())
        c.processWaveform(makeWaveform(refHeightAt(held,250.0,127.0)));

    QCOMPARE(finished.count(),1);
    QCOMPARE(finished.at(0).at(0).value<Result>().status,Status::Clipped);
    QCOMPARE(finished.at(0).at(1).toDouble(),250.0);
    QCOMPARE(held,250.0);
}

void FcuTuneControllerTest::testMoveFailureRestoresStart()
{
    FcuTuneController c;
    int requests = 0;
    double held = 0.0;
    connect(&c,&FcuTuneController::requestTrim,this,[&](const QString &k, double t){
        // Fail the third sweep move; accept everything else.
        requests++;
        bool ok = requests != 3;
        if(ok)
            held = t;
        c.trimUpdated(k,held,1,ok);
    });
    QSignalSpy finished(&c,&FcuTuneController::finished);

    QVERIFY(c.start(stage,-100.0,makeSettings(),makeConfig(),makeProc()));
    while(c.isRunning())
        c.processWaveform(makeWaveform(50));

    QCOMPARE(finished.count(),1);
    QCOMPARE(finished.at(0).at(0).value<Result>().status,Status::MoveFailed);
    QCOMPARE(held,-100.0);
}

void FcuTuneControllerTest::testAbortRestoresStart()
{
    FcuTuneController c;
    double held = 0.0;
    connect(&c,&FcuTuneController::requestTrim,this,[&](const QString &k, double t){
        held = t;
        c.trimUpdated(k,t,1,true);
    });
    QSignalSpy finished(&c,&FcuTuneController::finished);

    QVERIFY(c.start(stage,40.0,makeSettings(),makeConfig(),makeProc()));
    for(int i=0; i<10; i++)
        c.processWaveform(makeWaveform(50));
    QVERIFY(c.isRunning());
    c.abort();

    QVERIFY(!c.isRunning());
    QCOMPARE(finished.count(),1);
    QCOMPARE(finished.at(0).at(0).value<Result>().status,Status::Aborted);
    QCOMPARE(held,40.0);
}

void FcuTuneControllerTest::testRefDisabledRejected()
{
    FcuTuneController c;
    QSignalSpy requests(&c,&FcuTuneController::requestTrim);
    QVERIFY(!c.start(stage,0.0,makeSettings(),makeConfig(false),makeProc()));
    QVERIFY(!c.isRunning());
    QCOMPARE(requests.count(),0);
}

void FcuTuneControllerTest::testWaveformsIgnoredWhileMoving()
{
    FcuTuneController c;
    QSignalSpy requests(&c,&FcuTuneController::requestTrim);
    QSignalSpy progress(&c,&FcuTuneController::progress);

    QVERIFY(c.start(stage,0.0,makeSettings(),makeConfig(),makeProc()));
    QCOMPARE(requests.count(),1);
    progress.clear();

    // No trimUpdated() yet: the first move is still pending.
    for(int i=0; i<20; i++)
        c.processWaveform(makeWaveform(50));
    QCOMPARE(progress.count(),0);

    // An update for a different stage does not release the controller,
    // nor does one for this stage that carries some other trim.
    c.trimUpdated(u"LaserFreqConversionStage.other"_s,requests.at(0).at(1).toDouble(),1,true);
    c.trimUpdated(stage,12345.0,1,true);
    c.processWaveform(makeWaveform(50));
    QCOMPARE(progress.count(),0);

    c.trimUpdated(stage,requests.at(0).at(1).toDouble(),1,true);
    c.processWaveform(makeWaveform(50));
    QCOMPARE(progress.count(),1);
}

void FcuTuneControllerTest::testSaturationLevelRejects()
{
    // Peak height 80 counts is well inside the 8-bit range, but with
    // 1/128 V per count it is 0.625 V, above a 0.5 V saturation level.
    FcuTuneController c;
    double held = 0.0;
    connect(&c,&FcuTuneController::requestTrim,this,[&](const QString &k, double t){
        held = t;
        c.trimUpdated(k,t,1,true);
    });
    QSignalSpy finished(&c,&FcuTuneController::finished);

    auto s = makeSettings();
    s.saturationVolts = 0.5;
    QVERIFY(c.start(stage,0.0,s,makeConfig(),makeProc()));
    while(c.isRunning())
        c.processWaveform(makeWaveform(refHeightAt(held,0.0,80.0)));

    QCOMPARE(finished.count(),1);
    QCOMPARE(finished.at(0).at(0).value<Result>().status,Status::Clipped);
    QCOMPARE(held,0.0);

    // The same sweep below the level succeeds.
    s.saturationVolts = 0.7;
    finished.clear();
    QVERIFY(c.start(stage,0.0,s,makeConfig(),makeProc()));
    while(c.isRunning())
        c.processWaveform(makeWaveform(refHeightAt(held,0.0,80.0)));
    QCOMPARE(finished.at(0).at(0).value<Result>().status,Status::Success);
}

void FcuTuneControllerTest::testRecenterFindsPeakOutsideWindow()
{
    // Window ±3000 around 0; the peak at 4500 lies outside it, so the
    // first sweep ends at the edge and a re-centered sweep finds it.
    FcuTuneController c;
    double held = 0.0;
    connect(&c,&FcuTuneController::requestTrim,this,[&](const QString &k, double t){
        held = t;
        c.trimUpdated(k,t,1,true);
    });
    QSignalSpy finished(&c,&FcuTuneController::finished);
    QSignalSpy recentered(&c,&FcuTuneController::recentered);

    QVERIFY(c.start(stage,0.0,makeSettings(),makeConfig(),makeProc()));
    while(c.isRunning())
        c.processWaveform(makeWaveform(refHeightAt(held,4500.0,100.0) + 2));

    QCOMPARE(recentered.count(),1);
    QCOMPARE(recentered.at(0).at(0).toDouble(),3000.0);
    QCOMPARE(finished.count(),1);
    auto r = finished.at(0).at(0).value<Result>();
    QCOMPARE(r.status,Status::Success);
    QVERIFY2(std::abs(r.center - 4500.0) < 150.0, qPrintable(u"center %1"_s.arg(r.center)));
}

void FcuTuneControllerTest::testRecenterLimit()
{
    // A monotonic signal never yields an interior maximum; the controller
    // gives up after maxRecenters and restores the starting trim.
    FcuTuneController c;
    double held = 0.0;
    connect(&c,&FcuTuneController::requestTrim,this,[&](const QString &k, double t){
        held = t;
        c.trimUpdated(k,t,1,true);
    });
    QSignalSpy finished(&c,&FcuTuneController::finished);
    QSignalSpy recentered(&c,&FcuTuneController::recentered);

    auto s = makeSettings();
    s.maxRecenters = 2;
    QVERIFY(c.start(stage,100.0,s,makeConfig(),makeProc()));
    while(c.isRunning())
        c.processWaveform(makeWaveform(std::clamp(static_cast<int>(10.0 + held/200.0),1,120)));

    QCOMPARE(recentered.count(),2);
    QCOMPARE(finished.count(),1);
    QCOMPARE(finished.at(0).at(0).value<Result>().status,Status::PeakAtEdge);
    QCOMPARE(held,100.0);
}

QTEST_GUILESS_MAIN(FcuTuneControllerTest)

#include "tst_fcutunecontroller.moc"
