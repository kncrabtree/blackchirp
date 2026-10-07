#include <QtTest>

#include <cmath>
#include <random>

#include <data/lif/fcutunesweep.h>

using namespace Qt::Literals::StringLiterals;
using namespace BC::FcuTune;

class FcuTuneSweepTest : public QObject
{
    Q_OBJECT

private slots:
    void testTrimsIncreasing();
    void testTrimsDecreasing();
    void testSettingsSanitized();
    void testDiscardAndCount();
    void testSincPeakRecovered();
    void testSincPeakRecoveredReversed();
    void testNegativeSignalOriented();
    void testEqualWeightFallback();
    void testPeakAtEdge();
    void testLowContrast();
    void testNoSignal();
    void testNarrowPeakFails();
    void testClippedOverridesSuccess();
    void testMissingPointInsufficient();

private:
    //! sinc²-shaped phase-match curve with optional Gaussian noise.
    static double phaseMatch(double x, double x0, double width, double amp)
    {
        auto a = M_PI*(x - x0)/width;
        auto s = std::abs(a) < 1e-12 ? 1.0 : std::sin(a)/a;
        return amp*s*s;
    }

    //! Run a full sweep against phaseMatch() with per-waveform noise.
    static Result runSweep(const Settings &s, double start, double x0, double width,
                           double amp, double noise, unsigned seed = 1)
    {
        FcuTuneSweep sweep(s,start);
        std::mt19937 gen(seed);
        std::normal_distribution<double> dist(0.0,noise > 0.0 ? noise : 1.0);
        while(!sweep.isComplete())
        {
            auto t = sweep.currentTrim();
            while(!sweep.addWaveform(phaseMatch(t,x0,width,amp) + (noise > 0.0 ? dist(gen) : 0.0)))
                ;
            sweep.advance();
        }
        return sweep.result();
    }
};

void FcuTuneSweepTest::testTrimsIncreasing()
{
    Settings s;
    s.halfWidth = 100.0;
    s.points = 5;
    FcuTuneSweep sweep(s,1000.0);

    const std::vector<double> expected{900.0,950.0,1000.0,1050.0,1100.0};
    QCOMPARE(sweep.trims(),expected);
    QCOMPARE(sweep.currentTrim(),900.0);
}

void FcuTuneSweepTest::testTrimsDecreasing()
{
    Settings s;
    s.halfWidth = 100.0;
    s.points = 5;
    s.direction = -1;
    FcuTuneSweep sweep(s,0.0);

    const std::vector<double> expected{100.0,50.0,0.0,-50.0,-100.0};
    QCOMPARE(sweep.trims(),expected);
}

void FcuTuneSweepTest::testSettingsSanitized()
{
    Settings s;
    s.points = 1;
    s.waveformsPerPoint = 0;
    s.discardPerPoint = -3;
    s.halfWidth = -5.0;
    s.direction = 7;
    FcuTuneSweep sweep(s,0.0);

    QCOMPARE(sweep.settings().points,3);
    QCOMPARE(sweep.settings().waveformsPerPoint,1);
    QCOMPARE(sweep.settings().discardPerPoint,0);
    QVERIFY(sweep.settings().halfWidth > 0.0);
    QCOMPARE(sweep.settings().direction,1);
}

void FcuTuneSweepTest::testDiscardAndCount()
{
    Settings s;
    s.points = 3;
    s.waveformsPerPoint = 2;
    s.discardPerPoint = 2;
    FcuTuneSweep sweep(s,0.0);

    // The discarded waveforms carry a value that would skew the mean.
    QVERIFY(!sweep.addWaveform(1000.0));
    QVERIFY(!sweep.addWaveform(1000.0));
    QVERIFY(!sweep.addWaveform(1.0));
    QVERIFY(sweep.addWaveform(3.0));

    auto [mean,err] = sweep.pointStats(0);
    QCOMPARE(mean,2.0);
    QCOMPARE(err,1.0);

    QCOMPARE(sweep.perMilComplete(),333);
    QVERIFY(sweep.advance());
    QCOMPARE(sweep.currentIndex(),1);
    QVERIFY(sweep.advance());
    QVERIFY(!sweep.advance());
    QVERIFY(sweep.isComplete());
}

void FcuTuneSweepTest::testSincPeakRecovered()
{
    Settings s;
    s.halfWidth = 3000.0;
    s.points = 15;
    s.waveformsPerPoint = 20;
    s.discardPerPoint = 0;

    // Peak offset from the window center, as after a period of drift.
    auto r = runSweep(s,0.0,-850.0,4000.0,1.0,0.03);
    QCOMPARE(r.status,Status::Success);
    QVERIFY2(std::abs(r.center + 850.0) < 100.0, qPrintable(u"center %1"_s.arg(r.center)));
    QVERIFY(std::isfinite(r.centerUncertainty));
    QVERIFY(r.centerUncertainty > 0.0);
    QCOMPARE(r.trims.size(),std::size_t{15});
}

void FcuTuneSweepTest::testSincPeakRecoveredReversed()
{
    Settings s;
    s.halfWidth = 3000.0;
    s.points = 15;
    s.waveformsPerPoint = 20;
    s.discardPerPoint = 0;
    s.direction = -1;

    auto r = runSweep(s,500.0,1200.0,4000.0,1.0,0.03);
    QCOMPARE(r.status,Status::Success);
    QVERIFY2(std::abs(r.center - 1200.0) < 100.0, qPrintable(u"center %1"_s.arg(r.center)));
    QVERIFY(r.trims.front() > r.trims.back());
}

void FcuTuneSweepTest::testNegativeSignalOriented()
{
    Settings s;
    s.halfWidth = 3000.0;
    s.points = 15;
    s.waveformsPerPoint = 10;
    s.discardPerPoint = 0;

    auto r = runSweep(s,0.0,400.0,4000.0,-2.5,0.05);
    QCOMPARE(r.status,Status::Success);
    QVERIFY2(std::abs(r.center - 400.0) < 100.0, qPrintable(u"center %1"_s.arg(r.center)));
}

void FcuTuneSweepTest::testEqualWeightFallback()
{
    // Noiseless points have zero standard error, so the fit falls back to
    // equal weights. A sampled Gaussian is recovered exactly.
    std::vector<double> x, y, e;
    for(int i=0; i<9; i++)
    {
        auto xi = -400.0 + 100.0*i;
        x.push_back(xi);
        y.push_back(5.0*std::exp(-0.5*std::pow((xi - 37.0)/150.0,2)));
        e.push_back(0.0);
    }

    auto r = FcuTuneSweep::fitPeak(x,y,e,0.2);
    QCOMPARE(r.status,Status::Success);
    QVERIFY2(std::abs(r.center - 37.0) < 1e-6, qPrintable(u"center %1"_s.arg(r.center)));
}

void FcuTuneSweepTest::testPeakAtEdge()
{
    Settings s;
    s.halfWidth = 1000.0;
    s.points = 11;
    s.waveformsPerPoint = 5;
    s.discardPerPoint = 0;

    auto r = runSweep(s,0.0,2500.0,6000.0,1.0,0.0);
    QCOMPARE(r.status,Status::PeakAtEdge);
    QCOMPARE(r.center,1000.0);
    QVERIFY(!r.success());
}

void FcuTuneSweepTest::testLowContrast()
{
    Settings s;
    s.halfWidth = 50.0;
    s.points = 11;
    s.waveformsPerPoint = 5;
    s.discardPerPoint = 0;

    auto r = runSweep(s,0.0,10.0,8000.0,1.0,0.0);
    QCOMPARE(r.status,Status::LowContrast);
    QVERIFY(r.contrast < 0.2);
}

void FcuTuneSweepTest::testNoSignal()
{
    std::vector<double> x{-1.0,0.0,1.0}, y{0.0,0.0,0.0}, e{0.0,0.0,0.0};
    auto r = FcuTuneSweep::fitPeak(x,y,e,0.2);
    QCOMPARE(r.status,Status::LowContrast);
}

void FcuTuneSweepTest::testNarrowPeakFails()
{
    // Only the center point carries signal; neighbors are at or below zero.
    std::vector<double> x{-200.0,-100.0,0.0,100.0,200.0};
    std::vector<double> y{0.0,-0.01,1.0,0.0,0.0};
    std::vector<double> e(5,0.01);
    auto r = FcuTuneSweep::fitPeak(x,y,e,0.2);
    QCOMPARE(r.status,Status::FitFailed);
}

void FcuTuneSweepTest::testClippedOverridesSuccess()
{
    Settings s;
    s.halfWidth = 3000.0;
    s.points = 7;
    s.waveformsPerPoint = 1;
    s.discardPerPoint = 0;
    FcuTuneSweep sweep(s,0.0);
    while(!sweep.isComplete())
    {
        sweep.addWaveform(phaseMatch(sweep.currentTrim(),0.0,4000.0,1.0),sweep.currentIndex() == 3);
        sweep.advance();
    }

    auto r = sweep.result();
    QCOMPARE(r.status,Status::Clipped);
    QVERIFY(!r.success());
}

void FcuTuneSweepTest::testMissingPointInsufficient()
{
    Settings s;
    s.points = 3;
    s.waveformsPerPoint = 1;
    s.discardPerPoint = 0;
    FcuTuneSweep sweep(s,0.0);
    sweep.addWaveform(1.0);
    sweep.advance();

    QCOMPARE(sweep.result().status,Status::InsufficientData);
}

QTEST_GUILESS_MAIN(FcuTuneSweepTest)

#include "tst_fcutunesweep.moc"
