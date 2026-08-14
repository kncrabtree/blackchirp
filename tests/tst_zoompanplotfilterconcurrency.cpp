#include <QtTest>
#include <QApplication>
#include <QVector>
#include <QPointF>
#include <QRectF>
#include <QThread>
#include <QDeadlineTimer>
#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>

#include <gui/plot/zoompanplot.h>
#include <gui/plot/blackchirpplotcurve.h>
#include <gui/plot/curvefactory.h>

using namespace Qt::Literals::StringLiterals;

namespace {

/// \brief Shared concurrency counters for one or more ConcurrencyProbeCurve instances.
///
/// A single tracker shared across multiple curves lets a test detect
/// overlap between filter passes even when the overlap spans two different
/// curves — e.g. one pass's worker thread still inside curve A's _filter()
/// while a second, wrongly-concurrent pass has already started on curve B.
struct PassTracker {
    std::atomic<int> active{0};      ///< Number of _filter() calls currently executing.
    std::atomic<int> peakActive{0};  ///< High-water mark of \c active observed by any thread.
    std::atomic<int> passCount{0};   ///< Total number of completed _filter() calls.
    std::atomic<int> delayMs{0};     ///< Artificial delay injected into _filter() to widen race windows.

    /// \brief Monotonic count of _filter() entries.
    ///
    /// \c active is transient: a test that polls it for "a pass is in
    /// flight" can miss the window entirely if the worker starts and
    /// finishes between two polls, which is exactly what happens on a
    /// loaded machine. This counter only ever increases, so waiting on
    /// it cannot miss.
    std::atomic<int> entered{0};

    /// \brief When set, _filter() parks on entry until it is cleared.
    ///
    /// Lets a test hold a pass open for as long as it needs to perform
    /// the adversarial action, rather than racing the pass's natural
    /// duration. The park happens before the real _filter() body runs,
    /// so p_dataMutex is not held while parked and the test is free to
    /// mutate curve data.
    std::atomic<bool> hold{false};
};

/// Upper bound on how long a parked filter pass waits to be released.
/// Only a guard against a test that forgets to clear the gate — it must
/// never be reached in a passing run.
constexpr int holdReleaseTimeoutMs = 10000;

/// \brief BlackchirpPlotCurve subclass instrumenting _filter() for concurrency testing.
///
/// Records, via a shared PassTracker, how many _filter() calls are
/// concurrently in progress (and the peak observed) so a test can assert
/// that ZoomPanPlot never runs two filter passes at once. Delegates the
/// actual downsampling to BlackchirpPlotCurve::_filter() so the tests also
/// exercise the real bounding-rect-publish logic, not a stub.
class ConcurrencyProbeCurve : public BlackchirpPlotCurve
{
public:
    ConcurrencyProbeCurve(std::unique_ptr<CurveStorageInterface> storage,
                          const QString key,
                          const QString title = QString(""),
                          Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                          QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol,
                          QwtPlotCurve::CurveStyle defaultStyle = QwtPlotCurve::Lines,
                          std::shared_ptr<PassTracker> tracker = std::make_shared<PassTracker>())
        : BlackchirpPlotCurve(std::move(storage), key, title, defaultLineStyle, defaultMarker, defaultStyle),
          ps_tracker(std::move(tracker))
    {}

    std::shared_ptr<PassTracker> ps_tracker;

    /// \brief Sets the artificial per-_filter() delay (milliseconds) used to widen race windows.
    void setDelayMs(int ms) { ps_tracker->delayMs.store(ms); }

protected:
    QVector<QPointF> _filter(int w, const QwtScaleMap map) override
    {
        const int now = ps_tracker->active.fetch_add(1) + 1;
        int prevPeak = ps_tracker->peakActive.load();
        while (now > prevPeak && !ps_tracker->peakActive.compare_exchange_weak(prevPeak, now))
        { }

        ps_tracker->entered.fetch_add(1);

        // Park before touching any curve state, so a test holding this
        // pass open can still mutate the curve.
        QDeadlineTimer holdDeadline(holdReleaseTimeoutMs);
        while (ps_tracker->hold.load() && !holdDeadline.hasExpired())
            QThread::msleep(1);

        const int delay = ps_tracker->delayMs.load();
        if (delay > 0)
            QThread::msleep(static_cast<unsigned long>(delay));

        auto result = BlackchirpPlotCurve::_filter(w, map);

        ps_tracker->active.fetch_sub(1);
        ps_tracker->passCount.fetch_add(1);
        return result;
    }
};

} // namespace

/// \brief Test-only ZoomPanPlot subclass exposing protected hooks needed by the test body.
///
/// ZoomPanPlot has no pure-virtual hooks, so a direct subclass with no
/// additional behavior is otherwise sufficient. Two things are exposed
/// beyond the public API:
///  - drainWorker(), so the test body can synchronize with the worker
///    without polling private state (d_busy is private).
///  - limitRect(), so the test body can observe the per-axis bounding rect
///    ZoomPanPlot maintains internally (there is no public getter; d_config
///    is private) — this is what the bounding-rect-coherence tests below
///    are actually checking.
class TestZoomPanPlot : public ZoomPanPlot
{
public:
    explicit TestZoomPanPlot(const QString &name, QWidget *parent = nullptr)
        : ZoomPanPlot(name, parent) {}

    void drainWorker() { waitForFilterComplete(); }

    QRectF limitRect(QwtPlot::Axis xAx, QwtPlot::Axis yAx) const
    { return getLimitRect(xAx, yAx); }
};

/*!
 * \brief Adversarial coverage for ZoomPanPlot's asynchronous filter pass.
 *
 * ZoomPanPlot farms curve downsampling to a pool thread and coordinates it
 * with a curve registry guarded by a mutex, a d_busy flag, and an xDirty
 * re-kick handled in the QFutureWatcher::finished lambda. Two invariants
 * are under test:
 *
 *  1. No two filter passes ever run concurrently, and a request that
 *     arrives while a pass is already in flight is not dropped — it is
 *     serviced by the xDirty re-kick once the in-flight pass completes.
 *  2. Once a filter pass (and any chained re-kick) has settled, each
 *     axis's internally tracked bounding rect matches the union of its
 *     curves' actual data extents, including when curve data is mutated
 *     while a pass is in flight.
 *
 * Registry mutation during a pass (attach, detach, curve destruction,
 * resetPlot) is covered separately by tst_zoompanplotthreadsafety and is
 * not duplicated here.
 */
class ZoomPanPlotFilterConcurrencyTest : public QObject
{
    Q_OBJECT

private slots:
    void init();
    void cleanup();

    void testReplotRequiresVisiblePlot();
    void testNoOverlappingFilterPasses();
    void testDirtyRequestWhileBusyEventuallyRuns();
    void testMultiCurveMultiAxisNoOverlap();
    void testBoundingRectCoherenceAfterDataMutation();
    void testAppendPointDuringInFlightPass();
    void testSmallDatasetFilterRepublishesBoundingRectSafely();
    void testAutoScaleResetPlotUnderLoad();

private:
    /// \brief Builds \a n points with x = xOffset + i and y = yScale * (i % 100) / 100.
    ///
    /// x is strictly increasing, so first()/last() give exact min/max x —
    /// matching what BlackchirpPlotCurve::setCurveData() assumes internally.
    static QVector<QPointF> makePoints(int n, double yScale = 1.0, double xOffset = 0.0);

    /// \brief Blocks until \a counter() stops changing, or \a timeoutMs elapses.
    ///
    /// A single waitForFilterComplete() call is not sufficient to observe
    /// quiescence: processing the finished-handler's queued signal can
    /// itself launch a re-kicked pass (via xDirty), so this repeatedly
    /// drains and pumps the event loop until the observed counter is
    /// stable across several consecutive rounds. Deterministic in that it
    /// terminates as soon as the state has genuinely settled rather than
    /// after a fixed sleep; \a timeoutMs is only a safety net against a
    /// hang.
    static void drainUntilQuiescent(TestZoomPanPlot *plot, const std::function<int()> &counter,
                                     int timeoutMs = 5000);

    TestZoomPanPlot *p_plot{nullptr};
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

QVector<QPointF> ZoomPanPlotFilterConcurrencyTest::makePoints(int n, double yScale, double xOffset)
{
    QVector<QPointF> pts;
    pts.reserve(n);
    for (int i = 0; i < n; ++i)
        pts.append({xOffset + static_cast<double>(i), yScale * static_cast<double>(i % 100) / 100.0});
    return pts;
}

void ZoomPanPlotFilterConcurrencyTest::drainUntilQuiescent(TestZoomPanPlot *plot,
                                                            const std::function<int()> &counter,
                                                            int timeoutMs)
{
    QDeadlineTimer deadline(timeoutMs);
    int last = counter();
    int stableRounds = 0;
    while (stableRounds < 3 && !deadline.hasExpired())
    {
        plot->drainWorker();
        QCoreApplication::processEvents();
        const int now = counter();
        if (now == last)
            ++stableRounds;
        else
        {
            stableRounds = 0;
            last = now;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-test setup / teardown
// ---------------------------------------------------------------------------

void ZoomPanPlotFilterConcurrencyTest::init()
{
    p_plot = new TestZoomPanPlot("filterConcurrency"_L1);
    // ZoomPanPlot::replot() returns immediately when !isVisible(). Every
    // other test in this file depends on show() having been called here;
    // testReplotRequiresVisiblePlot asserts that dependency directly so a
    // future edit that drops this call fails loudly instead of leaving the
    // rest of the suite silently exercising nothing.
    p_plot->show();
    QCoreApplication::processEvents();
}

void ZoomPanPlotFilterConcurrencyTest::cleanup()
{
    p_plot->drainWorker();
    delete p_plot;
    p_plot = nullptr;
}

// ---------------------------------------------------------------------------
// testReplotRequiresVisiblePlot
//
// A plot that is never shown must never kick off a filter pass. This is a
// precondition every other test here relies on (via init()'s show() call);
// assert it explicitly so that dependency cannot be silently broken.
// ---------------------------------------------------------------------------
void ZoomPanPlotFilterConcurrencyTest::testReplotRequiresVisiblePlot()
{
    auto hiddenPlot = std::make_unique<TestZoomPanPlot>("filterConcurrencyHidden"_L1);
    QVERIFY(!hiddenPlot->isVisible());

    auto curve = CurveFactory::createStandardCurve<ConcurrencyProbeCurve>("hiddenCurve"_L1);
    curve->setCurveData(makePoints(20000));
    hiddenPlot->attachCurve(curve.get());

    hiddenPlot->replot();
    QCoreApplication::processEvents();
    hiddenPlot->drainWorker();
    QCoreApplication::processEvents();

    QCOMPARE(curve->ps_tracker->passCount.load(), 0);

    // Showing the plot and repeating the same call sequence must kick a
    // real pass.
    hiddenPlot->show();
    QCoreApplication::processEvents();
    hiddenPlot->replot();

    QTRY_VERIFY_WITH_TIMEOUT(curve->ps_tracker->passCount.load() > 0, 5000);

    hiddenPlot->detachCurve(curve.get());
    hiddenPlot->drainWorker();
}

// ---------------------------------------------------------------------------
// testNoOverlappingFilterPasses
//
// Hammers the plot with rapid replot()/resize() calls — the source of
// xDirty re-kicks — while a slow filter pass may still be in flight. d_busy
// plus xDirty are the only things standing between this loop and two
// overlapping worker invocations.
// ---------------------------------------------------------------------------
void ZoomPanPlotFilterConcurrencyTest::testNoOverlappingFilterPasses()
{
    auto curve = CurveFactory::createStandardCurve<ConcurrencyProbeCurve>("hammer"_L1);
    curve->setDelayMs(5);
    curve->setCurveData(makePoints(30000));
    p_plot->attachCurve(curve.get());

    const int iterations = 40;
    for (int i = 0; i < iterations; ++i)
    {
        p_plot->replot();
        if (i % 3 == 0)
            p_plot->resize(p_plot->width() + ((i / 3) % 2 == 0 ? 3 : -3), p_plot->height());
        QCoreApplication::processEvents();
    }

    drainUntilQuiescent(p_plot, [&]{ return curve->ps_tracker->passCount.load(); });
    p_plot->detachCurve(curve.get());

    QCOMPARE(curve->ps_tracker->peakActive.load(), 1);
    // A request arriving while busy must not be dropped: it must eventually
    // run via the xDirty re-kick, so more than the single initial pass
    // should have executed.
    QVERIFY2(curve->ps_tracker->passCount.load() >= 2,
             "expected the xDirty re-kick to service at least one request "
             "deferred while the plot was busy");
}

// ---------------------------------------------------------------------------
// testDirtyRequestWhileBusyEventuallyRuns
//
// Synchronizes deterministically (via the active counter, not a timing
// race) on a pass being provably in flight before issuing a second request,
// then confirms that request is serviced afterward rather than dropped.
// ---------------------------------------------------------------------------
void ZoomPanPlotFilterConcurrencyTest::testDirtyRequestWhileBusyEventuallyRuns()
{
    auto curve = CurveFactory::createStandardCurve<ConcurrencyProbeCurve>("dirty"_L1);
    curve->setDelayMs(30);
    curve->setCurveData(makePoints(40000));
    p_plot->attachCurve(curve.get());

    // Hold pass 1 open rather than racing its natural duration.
    curve->ps_tracker->hold.store(true);
    p_plot->replot(); // kicks off pass 1

    QTRY_VERIFY_WITH_TIMEOUT(curve->ps_tracker->entered.load() > 0, 5000);

    // Pass 1 is parked inside _filter(). resizeEvent() sets
    // d_config.xDirty and calls replot(); replot() sees d_busy and must
    // defer rather than launch a second, overlapping pass.
    p_plot->resize(p_plot->width() + 10, p_plot->height());
    QCoreApplication::processEvents();

    curve->ps_tracker->hold.store(false);
    drainUntilQuiescent(p_plot, [&]{ return curve->ps_tracker->passCount.load(); });
    p_plot->detachCurve(curve.get());

    QCOMPARE(curve->ps_tracker->peakActive.load(), 1);
    QVERIFY2(curve->ps_tracker->passCount.load() >= 2,
             "the resize issued while pass 1 was in flight must be serviced "
             "by a re-kick, not dropped");
}

// ---------------------------------------------------------------------------
// testMultiCurveMultiAxisNoOverlap
//
// Two curves on different y axes share one PassTracker, so the peak-active
// count catches overlap across curves and axes, not just within a single
// curve's own passes.
// ---------------------------------------------------------------------------
void ZoomPanPlotFilterConcurrencyTest::testMultiCurveMultiAxisNoOverlap()
{
    auto tracker = std::make_shared<PassTracker>();
    tracker->delayMs.store(8);

    auto storageA = std::make_unique<SettingsStorageWrapper>("multiA"_L1);
    auto curveA = std::make_unique<ConcurrencyProbeCurve>(std::move(storageA), "multiA"_L1,
                                                           QString(""), Qt::SolidLine, QwtSymbol::NoSymbol,
                                                           QwtPlotCurve::Lines, tracker);
    curveA->setCurveData(makePoints(20000, 1.0));
    p_plot->attachCurve(curveA.get());

    auto storageB = std::make_unique<SettingsStorageWrapper>("multiB"_L1);
    auto curveB = std::make_unique<ConcurrencyProbeCurve>(std::move(storageB), "multiB"_L1,
                                                           QString(""), Qt::SolidLine, QwtSymbol::NoSymbol,
                                                           QwtPlotCurve::Lines, tracker);
    curveB->setCurveAxisY(QwtPlot::yRight);
    curveB->setCurveData(makePoints(25000, 3.0, 500.0));
    p_plot->attachCurve(curveB.get());

    const int iterations = 30;
    for (int i = 0; i < iterations; ++i)
    {
        p_plot->replot();
        if (i % 4 == 0)
            p_plot->autoScale();
        QCoreApplication::processEvents();
    }

    drainUntilQuiescent(p_plot, [&]{ return tracker->passCount.load(); });

    p_plot->detachCurve(curveA.get());
    p_plot->detachCurve(curveB.get());

    QCOMPARE(tracker->peakActive.load(), 1);
    QVERIFY(tracker->passCount.load() >= 2);
}

// ---------------------------------------------------------------------------
// testBoundingRectCoherenceAfterDataMutation
//
// Replaces a curve's data outright while a pass is provably mid-flight,
// then confirms ZoomPanPlot's internally tracked axis bounding rect matches
// the mutated data both immediately (replot() re-reads boundingRect()
// synchronously) and after everything settles (no half-published rect).
// ---------------------------------------------------------------------------
void ZoomPanPlotFilterConcurrencyTest::testBoundingRectCoherenceAfterDataMutation()
{
    auto curve = CurveFactory::createStandardCurve<ConcurrencyProbeCurve>("coherence"_L1);
    curve->setDelayMs(15);
    curve->setCurveData(makePoints(40000, 1.0));
    p_plot->attachCurve(curve.get());

    curve->ps_tracker->hold.store(true);
    p_plot->replot();
    QTRY_VERIFY_WITH_TIMEOUT(curve->ps_tracker->entered.load() > 0, 5000);

    // BlackchirpPlotCurve::setCurveData() and _filter() both take
    // p_dataMutex, so replacing the data here must be race-free even
    // though the worker thread is inside a pass at this exact moment.
    const auto mutated = makePoints(60000, 3.0, 100000.0);
    curve->setCurveData(mutated);
    curve->ps_tracker->hold.store(false);

    // replot() re-reads curve->boundingRect() synchronously in its own
    // item-list loop, so the axis limit rect reflects the mutated data
    // immediately — it does not need to wait for the async pass.
    p_plot->replot();

    const auto [xloIt, xhiIt] = std::minmax_element(mutated.cbegin(), mutated.cend(),
                                                      [](const QPointF &a, const QPointF &b){ return a.x() < b.x(); });
    const auto [yloIt, yhiIt] = std::minmax_element(mutated.cbegin(), mutated.cend(),
                                                      [](const QPointF &a, const QPointF &b){ return a.y() < b.y(); });

    auto checkRect = [&](const QRectF &r) {
        QCOMPARE(r.left(), xloIt->x());
        QCOMPARE(r.right(), xhiIt->x());
        QCOMPARE(r.top(), yloIt->y());
        QCOMPARE(r.bottom(), yhiIt->y());
    };

    checkRect(p_plot->limitRect(QwtPlot::xBottom, QwtPlot::yLeft));

    drainUntilQuiescent(p_plot, [&]{ return curve->ps_tracker->passCount.load(); });
    p_plot->detachCurve(curve.get());

    // Still coherent once the in-flight pass (working from the stale
    // pre-mutation snapshot) and any re-kicked pass have both settled.
    checkRect(p_plot->limitRect(QwtPlot::xBottom, QwtPlot::yLeft));
    QCOMPARE(curve->ps_tracker->peakActive.load(), 1);
}

// ---------------------------------------------------------------------------
// testAppendPointDuringInFlightPass
//
// Exercises the incremental appendPoint() path (as opposed to a bulk
// setCurveData() replacement) while a pass is in flight.
// ---------------------------------------------------------------------------
void ZoomPanPlotFilterConcurrencyTest::testAppendPointDuringInFlightPass()
{
    auto curve = CurveFactory::createStandardCurve<ConcurrencyProbeCurve>("append"_L1);
    curve->setDelayMs(20);
    curve->setCurveData(makePoints(20000, 1.0)); // y in [0, 0.99]
    p_plot->attachCurve(curve.get());

    curve->ps_tracker->hold.store(true);
    p_plot->replot();
    QTRY_VERIFY_WITH_TIMEOUT(curve->ps_tracker->entered.load() > 0, 5000);

    // appendPoint() takes the same p_dataMutex _filter() reads d_curveData
    // under, so appending while the worker is mid-pass must not race or
    // corrupt the bounding rect.
    for (int i = 0; i < 500; ++i)
        curve->appendPoint(QPointF(20000.0 + i, 5.0)); // well outside the initial y range

    curve->ps_tracker->hold.store(false);
    p_plot->replot();

    drainUntilQuiescent(p_plot, [&]{ return curve->ps_tracker->passCount.load(); });
    const auto rect = p_plot->limitRect(QwtPlot::xBottom, QwtPlot::yLeft);
    p_plot->detachCurve(curve.get());

    QCOMPARE(curve->ps_tracker->peakActive.load(), 1);
    QCOMPARE(rect.right(), 20499.0);
    QCOMPARE(rect.bottom(), 5.0); // Qwt convention: bottom of the bounding rect is the max y value.
    QCOMPARE(rect.top(), 0.0);
}

// ---------------------------------------------------------------------------
// testSmallDatasetFilterRepublishesBoundingRectSafely
//
// Below the 2.5*canvas-width downsampling threshold, BlackchirpPlotCurve::
// _filter() takes the branch that recomputes the curve's bounding rect via
// boundingRect() and republishes it under p_dataMutex rather than
// downsampling. Mutate the data on every round to exercise that branch
// under concurrent access and confirm the final state is coherent.
// ---------------------------------------------------------------------------
void ZoomPanPlotFilterConcurrencyTest::testSmallDatasetFilterRepublishesBoundingRectSafely()
{
    auto curve = CurveFactory::createStandardCurve<ConcurrencyProbeCurve>("small"_L1);
    curve->setDelayMs(10);

    const int rounds = 15;
    QVector<QPointF> lastData;
    for (int round = 0; round < rounds; ++round)
    {
        lastData = makePoints(40, 1.0 + 0.1 * round, static_cast<double>(round) * 1000.0);
        curve->setCurveData(lastData);
        if (round == 0)
            p_plot->attachCurve(curve.get());
        p_plot->replot();
        QCoreApplication::processEvents();
    }

    drainUntilQuiescent(p_plot, [&]{ return curve->ps_tracker->passCount.load(); });
    p_plot->replot(); // final synchronous re-read, matching the coherence test above
    const auto rect = p_plot->limitRect(QwtPlot::xBottom, QwtPlot::yLeft);
    p_plot->detachCurve(curve.get());

    QCOMPARE(curve->ps_tracker->peakActive.load(), 1);

    const auto [xloIt, xhiIt] = std::minmax_element(lastData.cbegin(), lastData.cend(),
                                                      [](const QPointF &a, const QPointF &b){ return a.x() < b.x(); });
    const auto [yloIt, yhiIt] = std::minmax_element(lastData.cbegin(), lastData.cend(),
                                                      [](const QPointF &a, const QPointF &b){ return a.y() < b.y(); });
    QCOMPARE(rect.left(), xloIt->x());
    QCOMPARE(rect.right(), xhiIt->x());
    QCOMPARE(rect.top(), yloIt->y());
    QCOMPARE(rect.bottom(), yhiIt->y());
}

// ---------------------------------------------------------------------------
// testAutoScaleResetPlotUnderLoad
//
// Repeatedly calls autoScale() and resetPlot() while a filter pass may be
// in flight. resetPlot() blocks on waitForFilterComplete() before clearing
// the registry and detaching items; this must not deadlock or race even
// when driven back-to-back with a busy plot.
// ---------------------------------------------------------------------------
void ZoomPanPlotFilterConcurrencyTest::testAutoScaleResetPlotUnderLoad()
{
    auto curve = CurveFactory::createStandardCurve<ConcurrencyProbeCurve>("resetload"_L1);
    curve->setDelayMs(5);
    curve->setCurveData(makePoints(30000));
    p_plot->attachCurve(curve.get());

    const int iterations = 20;
    for (int i = 0; i < iterations; ++i)
    {
        p_plot->replot();
        p_plot->autoScale();
        if (i % 5 == 4)
        {
            p_plot->resetPlot();
            p_plot->attachCurve(curve.get());
            curve->setCurveData(makePoints(30000 + i * 137));
        }
        QCoreApplication::processEvents();
    }

    drainUntilQuiescent(p_plot, [&]{ return curve->ps_tracker->passCount.load(); });
    p_plot->detachCurve(curve.get());

    QVERIFY(true); // reaching here means no crash/deadlock across the interleaved resetPlot()/autoScale() load
    QCOMPARE(curve->ps_tracker->peakActive.load(), 1);
}

// Entry point — QTEST_MAIN provides main() with a QApplication.
QTEST_MAIN(ZoomPanPlotFilterConcurrencyTest)
#include "tst_zoompanplotfilterconcurrency.moc"
