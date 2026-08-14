#include <QtTest>

#include <atomic>
#include <thread>
#include <vector>

#include <QTemporaryDir>
#include <QVector>

#include <data/lif/lifstorage.h>
#include <data/lif/liftrace.h>

/*!
 * \brief Adversarial coverage for LifStorage's cross-thread accessors.
 *
 * The acquisition path calls addTrace() to accumulate into the current
 * cell while the display widgets read that same cell through
 * currentLifTrace(), getLifTrace(), currentTraceShots(), and
 * completedShots(). Every one of those paths must hold the storage
 * mutex: a LifTrace holds implicitly-shared containers, so an unguarded
 * copy taken during a concurrent mutation corrupts the container
 * refcount rather than merely returning stale numbers.
 *
 * Each test tags every accumulated data point with a value derived
 * from the trace's own grid indices (e.g. one count per shot, or
 * laserIndex()+1 counts per shot) so that a reader can check, from the
 * returned LifTrace alone, that its raw sample data, its shot count,
 * and its (delayIndex, laserIndex) all describe one consistent
 * intermediate state -- never a torn read straddling two addTrace()
 * calls, and never a value stitched from two different grid cells.
 */
class LifStorageConcurrencyTest : public QObject
{
    Q_OBJECT

private slots:
    void concurrentAccumulateVsRead();
    void cellAdvanceBoundary();
    void concurrentDiskReadAfterAcquisitionEnds();

private:
    static constexpr int d_traceSize = 64;

    /// \brief Build a single-shot trace for cell (di,li) whose every sample
    /// equals \a perShot, so accumulated data can be checked against shots().
    static LifTrace makeUnitTrace(int di, int li, qint64 perShot, qint64 perShotRef)
    {
        QVector<qint64> lif(d_traceSize, perShot);
        QVector<qint64> ref(d_traceSize, perShotRef);
        return LifTrace(di, li, lif, ref, 1, 1.0, 1.0, 1.0);
    }
};

void LifStorageConcurrencyTest::concurrentAccumulateVsRead()
{
    // Single grid cell, so shots must be monotonically non-decreasing from
    // any one reader's point of view, and every read (mid-accumulation or
    // not) must show data internally consistent with its own shot count.
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());

    LifStorage storage(1, 1, 1, tmp.path());
    storage.start();

    constexpr int writesPerThread = 4000;
    constexpr int writerThreads = 2;
    constexpr int totalWrites = writesPerThread * writerThreads;

    std::atomic<bool> writersDone{false};
    std::atomic<int> failures{0};
    // Counts reads that landed strictly mid-accumulation. Asserted
    // non-zero below: without it this test would still pass if every
    // reader only ever saw the quiesced final state, which would make
    // the consistency checks above vacuous.
    std::atomic<int> intermediateReads{0};

    auto writer = [&]() {
        for (int i = 0; i < writesPerThread; ++i)
            storage.addTrace(LifStorageConcurrencyTest::makeUnitTrace(0, 0, 1, 2));
    };

    auto reader = [&]() {
        int lastShots = 0;
        // Keep sampling until the writers finish, plus one guaranteed pass
        // afterward so the final state is also exercised by this loop.
        bool sawDone = false;
        while (!sawDone) {
            sawDone = writersDone.load(std::memory_order_acquire);

            auto trace = storage.currentLifTrace();
            const int shots = trace.shots();
            const int size = trace.size();

            // Never-touched or fully-formed: no in-between size.
            if (size != 0 && size != LifStorageConcurrencyTest::d_traceSize) {
                failures.fetch_add(1);
                continue;
            }

            if (shots < lastShots) {
                failures.fetch_add(1); // shots must never appear to regress
            }
            lastShots = shots;

            if (shots > 0 && shots < totalWrites)
                intermediateReads.fetch_add(1);

            if (size == LifStorageConcurrencyTest::d_traceSize) {
                const auto lif = trace.lifRaw();
                const auto ref = trace.refRaw();
                for (int i = 0; i < size; ++i) {
                    if (lif.at(i) != shots || ref.at(i) != 2 * qint64(shots)) {
                        failures.fetch_add(1);
                        break;
                    }
                }
            }

            // currentTraceShots() reads the same monotonically-increasing
            // counter as the trace snapshot just taken, moments later: it
            // can only have advanced further, never fallen behind.
            const int shots2 = storage.currentTraceShots();
            if (shots2 < shots)
                failures.fetch_add(1);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < writerThreads; ++i)
        threads.emplace_back(writer);
    std::vector<std::thread> readers;
    for (int i = 0; i < 3; ++i)
        readers.emplace_back(reader);

    for (auto &t : threads)
        t.join();
    writersDone.store(true, std::memory_order_release);
    for (auto &t : readers)
        t.join();

    QCOMPARE(failures.load(), 0);

    // The readers must have actually raced the writers rather than
    // observing only the settled result.
    QVERIFY2(intermediateReads.load() > 0,
             "readers never observed a partially-accumulated trace; "
             "the consistency checks above proved nothing");

    // Hard check of the final, quiesced state: every shot from every
    // writer landed, with no torn or dropped update.
    auto final = storage.currentLifTrace();
    QCOMPARE(final.shots(), totalWrites);
    QCOMPARE(final.size(), d_traceSize);
    const auto lif = final.lifRaw();
    const auto ref = final.refRaw();
    for (int i = 0; i < final.size(); ++i) {
        QCOMPARE(lif.at(i), qint64(totalWrites));
        QCOMPARE(ref.at(i), qint64(2 * totalWrites));
    }
    QCOMPARE(storage.currentTraceShots(), totalWrites);
    QCOMPARE(storage.completedShots(), totalWrites);
}

void LifStorageConcurrencyTest::cellAdvanceBoundary()
{
    // 1x4 grid. The writer accumulates each cell, then advance()s to the
    // next. Every sample in cell li is tagged li+1 counts/shot (and
    // 10*(li+1) for the reference channel), so a reader can verify -- from
    // the trace's OWN reported (delayIndex, laserIndex) and shots() -- that
    // its data never mixes values from two different cells, even when the
    // read races the advance() boundary.
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());

    constexpr int laserPoints = 4;
    constexpr int shotsPerCell = 1500;

    LifStorage storage(1, laserPoints, 2, tmp.path());
    storage.start();

    std::atomic<bool> writerDone{false};
    std::atomic<int> failures{0};
    // Distinct laser indices seen in the accumulating cell, as a bitmask.
    // Asserted to cover more than one cell below, so a run in which the
    // readers only ever saw the final cell cannot pass silently.
    std::atomic<int> cellsObserved{0};

    auto checkTrace = [&](const LifTrace &trace, bool requireIndex, int expectLi) {
        if (trace.size() == 0 && trace.shots() == 0)
            return; // not yet reached, or default: nothing to check

        if (trace.size() != LifStorageConcurrencyTest::d_traceSize) {
            failures.fetch_add(1);
            return;
        }

        const int li = trace.laserIndex();
        const int di = trace.delayIndex();
        if (di != 0 || li < 0 || li >= laserPoints) {
            failures.fetch_add(1);
            return;
        }
        if (requireIndex && li != expectLi) {
            failures.fetch_add(1);
            return;
        }

        const qint64 shots = trace.shots();
        const auto lif = trace.lifRaw();
        const auto ref = trace.refRaw();
        const qint64 expectLif = shots * (li + 1);
        const qint64 expectRef = shots * 10 * (li + 1);
        for (int i = 0; i < trace.size(); ++i) {
            if (lif.at(i) != expectLif || ref.at(i) != expectRef) {
                failures.fetch_add(1);
                return;
            }
        }
    };

    auto writer = [&]() {
        for (int li = 0; li < laserPoints; ++li) {
            for (int s = 0; s < shotsPerCell; ++s)
                storage.addTrace(LifStorageConcurrencyTest::makeUnitTrace(0, li, li + 1, 10 * (li + 1)));
            storage.advance();
        }
    };

    auto reader = [&]() {
        int cell = 0;
        bool sawDone = false;
        while (!sawDone) {
            sawDone = writerDone.load(std::memory_order_acquire);

            auto current = storage.currentLifTrace();
            checkTrace(current, false, -1);
            if (current.size() > 0)
                cellsObserved.fetch_or(1 << current.laserIndex());

            const int li = cell % laserPoints;
            checkTrace(storage.getLifTrace(0, li), true, li);
            ++cell;
        }
    };

    std::thread w(writer);
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i)
        readers.emplace_back(reader);

    w.join();
    writerDone.store(true, std::memory_order_release);
    for (auto &t : readers)
        t.join();

    QCOMPARE(failures.load(), 0);

    // The readers must have sampled the accumulating cell on both sides
    // of at least one advance(), which is the boundary under test.
    QVERIFY2(qPopulationCount(static_cast<uint>(cellsObserved.load())) > 1,
             "readers only ever saw one grid cell; the advance() boundary "
             "was never exercised");

    storage.finish();

    // Every cell must have landed intact in the completed-cell map.
    for (int li = 0; li < laserPoints; ++li) {
        auto trace = storage.getLifTrace(0, li);
        QCOMPARE(trace.delayIndex(), 0);
        QCOMPARE(trace.laserIndex(), li);
        QCOMPARE(trace.shots(), shotsPerCell);
        const auto lif = trace.lifRaw();
        for (int i = 0; i < trace.size(); ++i)
            QCOMPARE(lif.at(i), qint64(shotsPerCell) * (li + 1));
    }
}

void LifStorageConcurrencyTest::concurrentDiskReadAfterAcquisitionEnds()
{
    // getLifTrace() unlocks pu_mutex before falling through to
    // loadLifTrace() when !d_acquiring (lifstorage.cpp, getLifTrace()).
    // Exercise that path from many threads at once, against a *fresh*
    // LifStorage instance that never called addTrace() -- so every lookup
    // is a genuine cache-miss disk read, including the lazy, once-only
    // parse of lifparams.csv in ensureLifParamsLoaded() racing across
    // threads.
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());

    constexpr int laserPoints = 4;
    constexpr int shots = 777;

    {
        LifStorage writer(1, laserPoints, 5, tmp.path());
        writer.start();
        for (int li = 0; li < laserPoints; ++li) {
            for (int s = 0; s < shots; ++s)
                writer.addTrace(LifStorageConcurrencyTest::makeUnitTrace(0, li, li + 1, 10 * (li + 1)));
            writer.advance();
        }
        writer.finish();
    }

    LifStorage reader(1, laserPoints, 5, tmp.path());
    // d_acquiring is false by construction and no addTrace() has been
    // called, so every getLifTrace() below is forced onto the disk path.

    std::atomic<int> failures{0};
    auto work = [&]() {
        for (int iter = 0; iter < 300; ++iter) {
            const int li = iter % laserPoints;
            auto trace = reader.getLifTrace(0, li);
            if (trace.size() != LifStorageConcurrencyTest::d_traceSize
                || trace.delayIndex() != 0 || trace.laserIndex() != li
                || trace.shots() != shots) {
                failures.fetch_add(1);
                continue;
            }
            const auto lif = trace.lifRaw();
            const auto ref = trace.refRaw();
            const qint64 expectLif = qint64(shots) * (li + 1);
            const qint64 expectRef = qint64(shots) * 10 * (li + 1);
            for (int i = 0; i < trace.size(); ++i) {
                if (lif.at(i) != expectLif || ref.at(i) != expectRef) {
                    failures.fetch_add(1);
                    break;
                }
            }

            // A cell outside the grid has no lifparams.csv entry and must
            // come back as a harmless default rather than stale/garbage data.
            auto missing = reader.getLifTrace(0, laserPoints + 5);
            if (missing.size() != 0 || missing.shots() != 0)
                failures.fetch_add(1);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i)
        threads.emplace_back(work);
    for (auto &t : threads)
        t.join();

    QCOMPARE(failures.load(), 0);
}

QTEST_MAIN(LifStorageConcurrencyTest)
#include "tst_lifstorageconcurrency.moc"
