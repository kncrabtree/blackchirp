#include <QtTest>
#include <QApplication>
#include <QVector>
#include <QPointF>
#include <vector>
#include <memory>

#include <src/gui/plot/ftplot.h>
#include <src/gui/plot/blackchirpplotcurve.h>
#include <src/gui/plot/curvefactory.h>

/// Thin wrapper that promotes waitForFilterComplete() to public so the test
/// body can synchronize with the worker without touching production code.
class TestFtPlot : public FtPlot
{
    Q_OBJECT
public:
    explicit TestFtPlot(const QString &id, QWidget *parent = nullptr)
        : FtPlot(id, parent) {}

    void drainWorker() { waitForFilterComplete(); }
};

/// Integration test: concurrent attach/detach and data mutation during the
/// asynchronous filter pass in ZoomPanPlot.
///
/// Each test creates a fresh TestFtPlot (a concrete ZoomPanPlot subclass),
/// drives it into scenarios that exposed races before the snapshot-model fix,
/// then verifies the plot survives without crashing or asserting.
class ZoomPanPlotThreadSafetyTest : public QObject
{
    Q_OBJECT

public:
    ZoomPanPlotThreadSafetyTest() = default;
    ~ZoomPanPlotThreadSafetyTest() = default;

private slots:
    void init();
    void cleanup();

    void testRapidAttachDetach();
    void testAttachDuringFilter();
    void testDestroyCurveDuringFilter();
    void testResetPlotDuringFilter();

private:
    /// Builds a QVector<QPointF> with \a n evenly spaced points over [0, 1].
    static QVector<QPointF> makePoints(int n);

    TestFtPlot *p_plot{nullptr};
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

QVector<QPointF> ZoomPanPlotThreadSafetyTest::makePoints(int n)
{
    QVector<QPointF> pts;
    pts.reserve(n);
    for (int i = 0; i < n; ++i)
        pts.append({static_cast<double>(i) / n, static_cast<double>(i % 100) / 100.0});
    return pts;
}

// ---------------------------------------------------------------------------
// Per-test setup / teardown
// ---------------------------------------------------------------------------

void ZoomPanPlotThreadSafetyTest::init()
{
    // Construct a new plot for each test; show() so isVisible() returns true
    // and replot() does not early-return.
    p_plot = new TestFtPlot("test"_L1);
    p_plot->show();
    QCoreApplication::processEvents();
}

void ZoomPanPlotThreadSafetyTest::cleanup()
{
    // Drain any lingering worker before letting the destructor run.
    p_plot->drainWorker();
    delete p_plot;
    p_plot = nullptr;
}

// ---------------------------------------------------------------------------
// testRapidAttachDetach
//
// In a tight loop: create a BlackchirpPlotCurve, attach it, push data,
// call replot() to kick the worker, then immediately detach and destroy.
// Before the fix the worker could dereference the deleted curve pointer.
// ---------------------------------------------------------------------------
void ZoomPanPlotThreadSafetyTest::testRapidAttachDetach()
{
    const int iterations = 1000;
    const QVector<QPointF> pts = makePoints(200);

    for (int i = 0; i < iterations; ++i)
    {
        auto curve = CurveFactory::createStandardCurve<BlackchirpPlotCurve>(
            QString("rapid%1"_L1).arg(i));

        p_plot->attachCurve(curve.get());
        curve->setCurveData(pts, 0.0, 1.0);
        p_plot->replot();

        // Detach before the worker finishes — the formerly racy path.
        p_plot->detachCurve(curve.get());
        // unique_ptr falls out of scope; destructor calls _unregisterCurve.
    }

    // Drain any queued events (watcher::finished signal).
    p_plot->drainWorker();
    QCoreApplication::processEvents();
    QVERIFY(true); // reaching here means no crash
}

// ---------------------------------------------------------------------------
// testAttachDuringFilter
//
// Attach several curves with large data so the filter pass takes measurable
// time, start it with replot(), then attach/detach additional curves on the
// UI thread while processEvents() allows the watcher signal to arrive.
// ---------------------------------------------------------------------------
void ZoomPanPlotThreadSafetyTest::testAttachDuringFilter()
{
    const int baseCurves = 5;
    const int bigN = 60'000;
    const QVector<QPointF> bigPts = makePoints(bigN);

    std::vector<std::unique_ptr<BlackchirpPlotCurve>> baseCurveVec;
    for (int i = 0; i < baseCurves; ++i)
    {
        auto c = CurveFactory::createStandardCurve<BlackchirpPlotCurve>(
            QString("base%1"_L1).arg(i));
        c->setCurveData(bigPts, 0.0, 1.0);
        p_plot->attachCurve(c.get());
        baseCurveVec.push_back(std::move(c));
    }

    // Kick off the worker.
    p_plot->replot();

    // While the worker may be running, attach and detach extra curves.
    const int extraCurves = 50;
    for (int i = 0; i < extraCurves; ++i)
    {
        auto extra = CurveFactory::createStandardCurve<BlackchirpPlotCurve>(
            QString("extra%1"_L1).arg(i));
        extra->setCurveData(makePoints(100), 0.0, 1.0);
        p_plot->attachCurve(extra.get());
        QCoreApplication::processEvents();
        p_plot->detachCurve(extra.get());
        QCoreApplication::processEvents();
    }

    p_plot->drainWorker();
    QCoreApplication::processEvents();

    for (auto &c : baseCurveVec)
        p_plot->detachCurve(c.get());

    QVERIFY(true);
}

// ---------------------------------------------------------------------------
// testDestroyCurveDuringFilter
//
// Attach a curve with lots of data, fire replot(), then immediately reset the
// unique_ptr. The destructor must drain the worker via _unregisterCurve before
// tearing down the QwtPlotCurve vtable. Before the fix this was a UAF.
// ---------------------------------------------------------------------------
void ZoomPanPlotThreadSafetyTest::testDestroyCurveDuringFilter()
{
    const int bigN = 60'000;
    const QVector<QPointF> bigPts = makePoints(bigN);

    auto curve = CurveFactory::createStandardCurve<BlackchirpPlotCurve>("destroy"_L1);
    curve->setCurveData(bigPts, 0.0, 1.0);
    p_plot->attachCurve(curve.get());

    // Kick off the filter pass.
    p_plot->replot();

    // Destroy the curve while the worker may be iterating it.
    // ~BlackchirpPlotCurveBase -> _unregisterCurve -> waitForFilterComplete.
    curve.reset();

    QCoreApplication::processEvents();
    QVERIFY(true);
}

// ---------------------------------------------------------------------------
// testResetPlotDuringFilter
//
// Attach several large curves, fire replot(), then immediately detach all of
// them on the UI thread before the worker finishes. This exercises the path
// where the registry is drained mid-flight: each detachCurve() call blocks
// until waitForFilterComplete() returns, confirming the worker cannot hold a
// stale pointer to a curve that has already been removed from the registry.
// ---------------------------------------------------------------------------
void ZoomPanPlotThreadSafetyTest::testResetPlotDuringFilter()
{
    const int n = 4;
    const QVector<QPointF> bigPts = makePoints(60'000);

    std::vector<std::unique_ptr<BlackchirpPlotCurve>> curves;
    for (int i = 0; i < n; ++i)
    {
        auto c = CurveFactory::createStandardCurve<BlackchirpPlotCurve>(
            QString("reset%1"_L1).arg(i));
        c->setCurveData(bigPts, 0.0, 1.0);
        p_plot->attachCurve(c.get());
        curves.push_back(std::move(c));
    }

    // Fire the worker.
    p_plot->replot();

    // Immediately detach all curves — each call drains the worker before
    // removing the curve from the registry.
    for (auto &c : curves)
        p_plot->detachCurve(c.get());

    p_plot->drainWorker();
    QCoreApplication::processEvents();

    // Registry is now empty; curves fall out of scope safely.
    QVERIFY(true);
}

// Entry point — QTEST_MAIN provides main() with a QApplication.
QTEST_MAIN(ZoomPanPlotThreadSafetyTest)
#include "tst_zoompanplotthreadsafety.moc"
