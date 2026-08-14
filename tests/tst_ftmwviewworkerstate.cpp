#include <QtTest>

#include <QCoreApplication>
#include <cmath>
#include <memory>

#include <gui/widget/ftmwviewwidget.h>
#include <gui/plot/ftplot.h>
#include <data/analysis/ftworker.h>
#include <data/experiment/fid.h>

namespace {

// FtmwViewWidget assigns each of its workers a fixed integer id
// (FtmwViewWidget::d_liveId, d_plot1Id, d_plot2Id; d_mainId is handled
// separately -- processDiff() takes no id argument) that is private to
// the class; process() takes that id as a plain int, so a test driving
// it directly has to know the same values. They mirror the objectName
// each id's FtPlot is given in the constructor ("liveFtPlot", "ftPlot1",
// "ftPlot2"), which the tests below cross-check results against.
constexpr int kLiveId = 0;
constexpr int kPlot1Id = 1;
constexpr int kPlot2Id = 2;

/// Builds a single-frame FidList with sine-wave sample data, distinct
/// enough (via probe frequency and amplitude) to tell apart from other
/// lists built by this helper.
FidList makeFidList(int size, double probeFreqMHz, double amplitude)
{
    QVector<qint64> data(size);
    for(int i=0; i<size; ++i)
        data[i] = static_cast<qint64>(amplitude*std::sin(2.0*M_PI*i/64.0));

    Fid fid;
    fid.setData(data);
    fid.setSpacing(5e-9);
    fid.setSideband(RfConfig::UpperSideband);
    fid.setProbeFreq(probeFreqMHz);
    fid.setShots(1);

    FidList fl;
    fl.append(fid);
    return fl;
}

FtWorker::FidProcessingSettings makeSettings(FtWorker::FtUnits units = FtWorker::FtV)
{
    FtWorker::FidProcessingSettings s;
    s.startUs = 0.0;
    s.endUs = 0.0;
    s.expFilter = 0.0;
    s.zeroPadFactor = 0;
    s.removeDC = false;
    s.units = units;
    s.autoScaleIgnoreMHz = 0.0;
    s.windowFunction = FtWorker::None;
    return s;
}

/// Computes doFT() synchronously on a scratch FtWorker instance, wholly
/// independent of any FtmwViewWidget, to serve as ground truth for what a
/// dispatch with the given inputs must produce.
Ft referenceFt(const FidList &fl, int frame, const FtWorker::FidProcessingSettings &settings)
{
    FtWorker worker;
    return worker.doFT(fl,settings,frame,-1);
}

/// Computes doFtDiff() synchronously on a scratch FtWorker instance. The
/// slot only emits its result rather than returning it, so it is captured
/// via a direct (same-thread) connection.
Ft referenceDiff(const FidList &fl1, const FidList &fl2, int frame1, int frame2,
                  const FtWorker::FidProcessingSettings &settings)
{
    FtWorker worker;
    Ft result;
    QObject::connect(&worker,&FtWorker::ftDiffDone,[&result](Ft ft){ result = ft; });
    worker.doFtDiff(fl1,fl2,frame1,frame2,settings);
    return result;
}

bool ftEquals(const Ft &a, const Ft &b)
{
    if(a.size() != b.size())
        return false;
    for(int i=0; i<a.size(); ++i)
    {
        const double tol = 1e-6*(1.0+std::abs(b.at(i)));
        if(std::abs(a.at(i)-b.at(i)) > tol)
            return false;
    }
    return true;
}

} // namespace

/*!
 * \brief Adversarial coverage for FtmwViewWidget's worker coordination.
 *
 * The widget drives four independently-watched FT workers (live, main,
 * and two comparison plots), each with a busy flag and a
 * reprocessWhenDone flag that coalesces requests arriving while a worker
 * is running. Processing settings are snapshotted per dispatch, because
 * a settings edit rewrites the shared struct on the UI thread while
 * workers for the other plot ids may still be reading it.
 *
 * process()/processDiff() are driven directly here (the only headless
 * entry points into that machinery), which surfaces one contract worth
 * recording: process() itself never writes its FidList argument into the
 * per-id plot state. Only its normal callers (fidLoadComplete(),
 * updateFid(), both private and reachable only through a live
 * FidStorageBase) keep that state current before re-invoking process().
 * A coalesced replay triggered by reprocessWhenDone therefore always
 * re-processes whatever FidList was last recorded for that id by those
 * callers, not whatever was most recently passed to process() directly.
 * Driven the way this test drives it, that recorded FidList is always
 * empty, so a correctly-coalesced replay settles on an empty Ft
 * deterministically -- which is what coalescingBoundsRepeatedRequests()
 * checks for, in place of asserting on the content of "the last
 * request" (unreachable without a full Experiment/FidStorageBase, which
 * this target is not currently wired to load).
 *
 * Wherever a test needs an independently-computed reference Ft, that
 * reference is always computed on a scratch FtWorker before any dispatch
 * on the widget under test, never interleaved with or after it. Letting
 * the scratch worker's synchronous GSL FFT run on this thread overlap in
 * time with the widget's own QtConcurrent-dispatched worker (a second,
 * independent FtWorker instance, but both ultimately inside GSL's FFT
 * routines at once) was observed to occasionally leave the dispatched
 * worker's queued completion signal undelivered, hanging the test until
 * an external timeout. Computing references up front sidesteps that
 * overlap; the results do not depend on when they are computed.
 */
class FtmwViewWorkerStateTest : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();

    void constructHeadless();
    void dispatchSnapshotsProcessingSettingsAtCallTime();
    void dispatchIsIndependentPerId();
    void coalescingBoundsRepeatedRequests();
    void destructionWhileWorkersInFlightDoesNotCrash();
};

void FtmwViewWorkerStateTest::initTestCase()
{
    QCoreApplication::setOrganizationName("CrabtreeLab");
    QCoreApplication::setApplicationName("BlackchirpFtmwViewWorkerStateTest");
}

void FtmwViewWorkerStateTest::constructHeadless()
{
    // main = false is the "viewer" configuration: no acquisition dock, no
    // hardware behind it. This must be constructible and minimally usable
    // without an experiment ever having been loaded.
    FtmwViewWidget widget(false);

    QVERIFY(widget.getMainPlotFt().isEmpty());
    QVERIFY(!widget.getPlotNames().isEmpty());

    // getProcessingSettings() must return a usable snapshot even though
    // no experiment/processing panel interaction has happened yet.
    auto s = widget.getProcessingSettings();
    QVERIFY(s.zeroPadFactor >= 0);
}

void FtmwViewWorkerStateTest::dispatchSnapshotsProcessingSettingsAtCallTime()
{
    FtmwViewWidget widget(false);

    auto fl1 = makeFidList(4096,10000.0,1000.0);
    auto fl2 = makeFidList(4096,10000.0,400.0);

    // FtV and FtuV differ by a factor of 10^6 in output magnitude (see
    // FtWorker::FtUnits), which makes "which settings a result reflects"
    // trivially distinguishable.
    const auto settingsA = makeSettings(FtWorker::FtV);
    const auto settingsB = makeSettings(FtWorker::FtuV);

    // Computed up front, before anything is dispatched on the widget:
    // referenceDiff() runs a second FtWorker instance synchronously on
    // this thread, and letting that overlap with the widget's own
    // QtConcurrent-dispatched worker (a separate FtWorker instance, but
    // both ultimately calling into GSL's FFT routines at the same time)
    // was observed to occasionally leave the dispatched worker's queued
    // completion signal undelivered. Computing both references before
    // dispatch avoids that overlap entirely; the results themselves do
    // not depend on when they are computed.
    const Ft expected = referenceDiff(fl1,fl2,0,0,settingsA);
    const Ft wrongIfLeaked = referenceDiff(fl1,fl2,0,0,settingsB);
    QVERIFY(!expected.isEmpty());
    QVERIFY(!wrongIfLeaked.isEmpty());

    widget.updateProcessingSettings(settingsA);

    // Dispatches under settingsA; process()/processDiff() snapshot
    // d_currentProcessingSettings by value before starting the worker.
    widget.processDiff(fl1,fl2,0,0);

    // Rewrites the shared settings immediately afterward, on the same
    // thread, before the dispatched worker can possibly have completed.
    // A correct implementation is unaffected because the snapshot above
    // already happened; this call also kicks off unrelated reprocessing
    // of the other three ids as a side effect (harmless here).
    widget.updateProcessingSettings(settingsB);

    QCOMPARE(widget.getProcessingSettings().units, FtWorker::FtuV);

    QTRY_VERIFY_WITH_TIMEOUT(!widget.getMainPlotFt().isEmpty(),5000);
    const Ft actual = widget.getMainPlotFt();

    QVERIFY2(ftEquals(actual,expected),
             "diff result does not match the settings in effect at dispatch time");

    // Guards against the comparison above accidentally passing regardless
    // of which settings were used: had settingsB leaked into the
    // in-flight computation, the result would be scaled by 10^6 and this
    // would fail.
    QVERIFY(!ftEquals(actual,wrongIfLeaked));
}

void FtmwViewWorkerStateTest::dispatchIsIndependentPerId()
{
    FtmwViewWidget widget(false);
    widget.show();
    QVERIFY(QTest::qWaitForWindowExposed(&widget));

    auto *liveFtPlot = widget.findChild<FtPlot*>(u"liveFtPlot"_s);
    auto *ftPlot1 = widget.findChild<FtPlot*>(u"ftPlot1"_s);
    auto *ftPlot2 = widget.findChild<FtPlot*>(u"ftPlot2"_s);
    QVERIFY(liveFtPlot);
    QVERIFY(ftPlot1);
    QVERIFY(ftPlot2);
    QVERIFY(liveFtPlot->currentFt().isEmpty());
    QVERIFY(ftPlot1->currentFt().isEmpty());
    QVERIFY(ftPlot2->currentFt().isEmpty());

    // Deliberately not calling updateProcessingSettings() here: it also
    // triggers reprocess(), which independently dispatches process() for
    // live/plot1/plot2 (with their -- still default-empty -- cached
    // FidLists) as a side effect. Racing that internal dispatch against
    // this test's own process() calls below would make it a coin flip
    // whether a given id's "not busy" branch belongs to this test or to
    // reprocess(); reading the settings already in place from
    // construction sidesteps that race entirely.
    const auto settings = widget.getProcessingSettings();

    // All three FIDs are distinct (probe frequency and amplitude both
    // differ), so a swapped or dropped id is directly observable in the
    // settled result. FtWorker::doFT() serializes its actual GSL work
    // per FtWorker instance (see FtWorker's class comment: concurrent
    // calls on one instance are only safe across different code paths),
    // and all four of FtmwViewWidget's ids share the same FtWorker
    // instance -- so this checks dispatch/bookkeeping independence
    // (marking one id busy must not coalesce or misroute another id's
    // request into it), not wall-clock computation overlap, which this
    // shared instance does not provide.
    auto plot1Fl = makeFidList(2048,11000.0,900.0);
    auto liveFl = makeFidList(2048,12000.0,500.0);
    auto plot2Fl = makeFidList(2048,13000.0,300.0);

    // Computed before any dispatch -- see the comment on the analogous
    // computation in dispatchSnapshotsProcessingSettingsAtCallTime()
    // about why a reference FtWorker's synchronous work must not overlap
    // the widget's own dispatched worker.
    const Ft expectedPlot1 = referenceFt(plot1Fl,0,settings);
    const Ft expectedLive = referenceFt(liveFl,-1,settings);
    const Ft expectedPlot2 = referenceFt(plot2Fl,0,settings);
    QVERIFY(!expectedPlot1.isEmpty());
    QVERIFY(!expectedLive.isEmpty());
    QVERIFY(!expectedPlot2.isEmpty());

    // Dispatched back-to-back before the event loop ever runs: at the
    // moment each process() call is made, nothing else has had a chance
    // to run, so each id's own busy flag is still false and each call
    // must take the "dispatch a new worker" branch rather than the
    // "coalesce into whichever id is currently busy" branch.
    widget.process(kPlot1Id,plot1Fl,0);
    widget.process(kLiveId,liveFl,-1);
    widget.process(kPlot2Id,plot2Fl,0);

    QTRY_VERIFY_WITH_TIMEOUT(!ftPlot1->currentFt().isEmpty(),8000);
    QTRY_VERIFY_WITH_TIMEOUT(!liveFtPlot->currentFt().isEmpty(),8000);
    QTRY_VERIFY_WITH_TIMEOUT(!ftPlot2->currentFt().isEmpty(),8000);

    // Correct per-id routing: each plot reflects only its own id's
    // result, not a neighbor's, and none were silently dropped.
    QVERIFY(ftEquals(ftPlot1->currentFt(),expectedPlot1));
    QVERIFY(ftEquals(liveFtPlot->currentFt(),expectedLive));
    QVERIFY(ftEquals(ftPlot2->currentFt(),expectedPlot2));

    // Re-dispatch plot1 with fresh data: its busy flag must have been
    // cleared by its own completion alone, undisturbed by live/plot2
    // having completed in the meantime.
    auto plot1Fl2 = makeFidList(2048,14000.0,111.0);
    const Ft expectedPlot1b = referenceFt(plot1Fl2,0,settings);
    widget.process(kPlot1Id,plot1Fl2,0);
    QTRY_VERIFY_WITH_TIMEOUT(ftEquals(ftPlot1->currentFt(),expectedPlot1b),5000);
}

void FtmwViewWorkerStateTest::coalescingBoundsRepeatedRequests()
{
    FtmwViewWidget widget(false);
    widget.show();
    QVERIFY(QTest::qWaitForWindowExposed(&widget));

    auto *ftPlot2 = widget.findChild<FtPlot*>(u"ftPlot2"_s);
    QVERIFY(ftPlot2);
    QVERIFY(ftPlot2->currentFt().isEmpty());

    // The widget's FtWorker is one of its QObject children, so every
    // completed doFT() is directly countable. Counting dispatches is the
    // only way to tell coalescing apart from a fan-out: the settled plot
    // contents look the same either way.
    auto *worker = widget.findChild<FtWorker*>();
    QVERIFY(worker);
    QSignalSpy ftDoneSpy(worker,&FtWorker::ftDone);
    QVERIFY(ftDoneSpy.isValid());

    // Deliberately not calling updateProcessingSettings(): see the note in
    // dispatchIsIndependentPerId() -- it would race its own reprocess()
    // side effect against the dispatch below.

    // First dispatch: the id is not busy, so this one starts a real
    // (deliberately non-trivial) computation.
    auto firstFl = makeFidList(1<<19,15000.0,500.0);
    widget.process(kPlot2Id,firstFl,0);

    // Hammer the same id many times without ever returning to the event
    // loop. process()'s busy flag can only be cleared by the queued
    // ftProcessingComplete() callback, which needs the event loop to run,
    // so every one of these calls deterministically observes the id
    // still busy and only sets reprocessWhenDone -- never starts a second
    // concurrent worker. Each hammer FidList is distinct and non-empty,
    // so if that guarantee ever broke and some of these were actually
    // dispatched, the settled result would very likely be non-empty and
    // would vary from run to run depending on which of many racing
    // workers happened to finish last.
    for(int i=0; i<300; ++i)
    {
        auto fl = makeFidList(64,16000.0+i,200.0+i);
        widget.process(kPlot2Id,fl,0);
    }

    // A correct implementation coalesces the whole burst into exactly one
    // extra dispatch once the first finishes: 301 requests, two workers.
    // Wait for the first completion, then for the replay, then confirm
    // nothing further arrives.
    QTRY_VERIFY_WITH_TIMEOUT(ftDoneSpy.count() >= 2,15000);

    const int settled = ftDoneSpy.count();
    QTest::qWait(500);
    QCOMPARE(ftDoneSpy.count(),settled);

    QVERIFY2(settled == 2,
             qPrintable(QString("expected the burst to coalesce into one "
                                "replay (2 completions total), saw %1")
                            .arg(settled)));

    // The replay re-processes the id's cached plot FidList rather than
    // any FidList passed to process() directly (see the class comment),
    // and this test never populates that cache -- so the id settles back
    // to an empty Ft. Checked after the dispatch count, which is the
    // assertion carrying the weight here: an empty Ft is also the
    // starting state, so on its own it would hold even if nothing ran.
    QVERIFY(ftPlot2->currentFt().isEmpty());
}

void FtmwViewWorkerStateTest::destructionWhileWorkersInFlightDoesNotCrash()
{
    for(int iter=0; iter<5; ++iter)
    {
        auto widget = std::make_unique<FtmwViewWidget>(false);

        auto liveFl = makeFidList(1<<19,10000.0+iter,400.0);
        auto plot1Fl = makeFidList(1<<19,11000.0+iter,500.0);
        auto plot2Fl = makeFidList(1<<19,12000.0+iter,600.0);
        auto diff1 = makeFidList(1<<17,13000.0+iter,200.0);
        auto diff2 = makeFidList(1<<17,13000.0+iter,250.0);

        widget->process(kLiveId,liveFl,-1);
        widget->process(kPlot1Id,plot1Fl,0);
        widget->process(kPlot2Id,plot2Fl,0);
        widget->processDiff(diff1,diff2,0,0);

        // Destroy immediately, with all four workers very likely still
        // in flight on the thread pool. ~FtmwViewWidget() must wait for
        // them (it calls waitForFinished() on every watcher) rather than
        // let a pool thread go on to write into freed state.
        widget.reset();

        QCoreApplication::processEvents(QEventLoop::AllEvents,50);
    }

    QVERIFY(true);
}

QTEST_MAIN(FtmwViewWorkerStateTest)
#include "tst_ftmwviewworkerstate.moc"
