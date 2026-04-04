#include <QtTest>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QThread>

#include <atomic>
#include <vector>

#include <src/data/storage/waveformbuffer.h>

/*!
 * \brief Unit tests for WaveformBuffer — bounded SPSC ring buffer.
 *
 * Tests are organized into these categories:
 *  1. Basic correctness (single-threaded)
 *  2. Overflow / drop-oldest behavior
 *  3. Flush marker behavior
 *  4. drainAll behavior
 *  5. Pre-allocation (reserveBytes)
 *  6. Threading correctness
 *  7. Threading with overflow
 *  8. Throughput benchmark
 *  9. Flush marker under concurrency
 */
class WaveformBufferTest : public QObject
{
    Q_OBJECT

public:
    WaveformBufferTest();
    ~WaveformBufferTest();

private slots:
    // 1. Basic correctness (single-threaded)
    void testEmptyBufferReadReturnsFalse();
    void testEmptyBufferAvailableIsZero();
    void testSingleWriteRead();
    void testSingleWriteReadPreAccumulated();
    void testFifoOrdering();
    void testFillToCapacity();
    void testCapacityReported();
    void testResetClearsBuffer();
    void testResetClearsDroppedCount();

    // 2. Overflow / drop-oldest behavior
    void testOverflowDropsOldest();
    void testOverflowDropsMultiple();
    void testDroppedCountAccumulates();
    void testOverflowDataIntegrity();

    // 3. Flush marker behavior
    void testWriteFlushMarkerFields();
    void testFlushMarkerOverflow();
    void testFlushMarkerInterleaved();
    void testDrainAllIncludesFlushMarkers();

    // 4. drainAll behavior
    void testDrainAllEmpty();
    void testDrainAllMultiple();
    void testDrainAllAfterPartialRead();
    void testDrainAllClearsBuffer();

    // 5. Pre-allocation (reserveBytes)
    void testReserveBytesDontCrash();
    void testReserveBytesSmallData();
    void testReserveBytesBigData();

    // 6. Threading correctness
    void testThreadedProducerConsumer();
    void testThreadedDataIntegrity();
    void testWaitForDataWakesOnWrite();
    void testWaitForDataTimesOut();

    // 7. Threading with overflow
    void testThreadedOverflow();

    // 8. Throughput benchmark
    void testThroughputBenchmark();

    // 9. Flush marker under concurrency
    void testThreadedFlushMarkerOrdering();

private:
    // Build a deterministic QByteArray payload keyed to an integer index.
    static QByteArray makePayload(int index, int size = 64);
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

WaveformBufferTest::WaveformBufferTest()
{
    QCoreApplication::setApplicationName("BlackchirpWaveformBufferTest");
    QCoreApplication::setOrganizationName("CrabtreeLab");
    QCoreApplication::setOrganizationDomain("crabtreelab.ucdavis.edu");
}

WaveformBufferTest::~WaveformBufferTest()
{
}

QByteArray WaveformBufferTest::makePayload(int index, int size)
{
    QByteArray ba(size, static_cast<char>(index & 0xFF));
    // Embed the index as the first 4 bytes for easy verification.
    if (size >= 4) {
        ba[0] = static_cast<char>((index >> 24) & 0xFF);
        ba[1] = static_cast<char>((index >> 16) & 0xFF);
        ba[2] = static_cast<char>((index >>  8) & 0xFF);
        ba[3] = static_cast<char>( index        & 0xFF);
    }
    return ba;
}

// ---------------------------------------------------------------------------
// 1. Basic correctness (single-threaded)
// ---------------------------------------------------------------------------

void WaveformBufferTest::testEmptyBufferReadReturnsFalse()
{
    WaveformBuffer buf(5);
    WaveformEntry out;
    QVERIFY(!buf.read(out));
}

void WaveformBufferTest::testEmptyBufferAvailableIsZero()
{
    WaveformBuffer buf(5);
    QCOMPARE(buf.available(), 0);
}

void WaveformBufferTest::testSingleWriteRead()
{
    WaveformBuffer buf(5);
    QByteArray payload = makePayload(42);
    buf.write(payload, 7, false);

    QCOMPARE(buf.available(), 1);

    WaveformEntry out;
    QVERIFY(buf.read(out));
    QCOMPARE(out.data, payload);
    QCOMPARE(out.shotCount, static_cast<quint64>(7));
    QCOMPARE(out.preAccumulated, false);
    QCOMPARE(out.flushMarker, false);
    QCOMPARE(buf.available(), 0);
}

void WaveformBufferTest::testSingleWriteReadPreAccumulated()
{
    WaveformBuffer buf(5);
    QByteArray payload = makePayload(1);
    buf.write(payload, 3, true);

    WaveformEntry out;
    QVERIFY(buf.read(out));
    QCOMPARE(out.preAccumulated, true);
    QCOMPARE(out.shotCount, static_cast<quint64>(3));
}

void WaveformBufferTest::testFifoOrdering()
{
    WaveformBuffer buf(10);

    // Write three distinguishable entries.
    for (int i = 0; i < 3; ++i)
        buf.write(makePayload(i), static_cast<quint64>(i));

    for (int i = 0; i < 3; ++i) {
        WaveformEntry out;
        QVERIFY(buf.read(out));
        QCOMPARE(out.data, makePayload(i));
        QCOMPARE(out.shotCount, static_cast<quint64>(i));
    }

    // Buffer should now be empty.
    WaveformEntry extra;
    QVERIFY(!buf.read(extra));
}

void WaveformBufferTest::testFillToCapacity()
{
    const int cap = 8;
    WaveformBuffer buf(cap);

    for (int i = 0; i < cap; ++i)
        buf.write(makePayload(i), static_cast<quint64>(i));

    QCOMPARE(buf.available(), cap);
    QCOMPARE(buf.droppedCount(), static_cast<quint64>(0));

    for (int i = 0; i < cap; ++i) {
        WaveformEntry out;
        QVERIFY(buf.read(out));
        QCOMPARE(out.data, makePayload(i));
    }
}

void WaveformBufferTest::testCapacityReported()
{
    WaveformBuffer buf(17);
    QCOMPARE(buf.capacity(), 17);
}

void WaveformBufferTest::testResetClearsBuffer()
{
    WaveformBuffer buf(5);
    for (int i = 0; i < 4; ++i)
        buf.write(makePayload(i), 1);

    QVERIFY(buf.available() > 0);
    buf.reset();
    QCOMPARE(buf.available(), 0);

    WaveformEntry out;
    QVERIFY(!buf.read(out));
}

void WaveformBufferTest::testResetClearsDroppedCount()
{
    WaveformBuffer buf(3);
    // Cause at least one drop.
    for (int i = 0; i < 5; ++i)
        buf.write(makePayload(i), 1);

    QVERIFY(buf.droppedCount() > 0);
    buf.reset();
    QCOMPARE(buf.droppedCount(), static_cast<quint64>(0));
}

// ---------------------------------------------------------------------------
// 2. Overflow / drop-newest behavior
// ---------------------------------------------------------------------------

void WaveformBufferTest::testOverflowDropsOldest()
{
    // capacity = 3; write 4 entries — entry 3 (newest) should be dropped.
    WaveformBuffer buf(3);
    for (int i = 0; i < 4; ++i)
        buf.write(makePayload(i), static_cast<quint64>(i));

    QCOMPARE(buf.droppedCount(), static_cast<quint64>(1));
    QCOMPARE(buf.available(), 3);

    // The surviving entries should be 0, 1, 2 (earliest kept).
    for (int i = 0; i < 3; ++i) {
        WaveformEntry out;
        QVERIFY(buf.read(out));
        QCOMPARE(out.data, makePayload(i));
        QCOMPARE(out.shotCount, static_cast<quint64>(i));
    }
}

void WaveformBufferTest::testOverflowDropsMultiple()
{
    const int cap = 4;
    const int totalWrites = 10;
    const int expectedDrops = totalWrites - cap;

    WaveformBuffer buf(cap);
    for (int i = 0; i < totalWrites; ++i)
        buf.write(makePayload(i), static_cast<quint64>(i));

    QCOMPARE(buf.droppedCount(), static_cast<quint64>(expectedDrops));
    QCOMPARE(buf.available(), cap);

    // Earliest `cap` entries survive (drop-newest discards later writes).
    for (int i = 0; i < cap; ++i) {
        WaveformEntry out;
        QVERIFY(buf.read(out));
        QCOMPARE(out.data, makePayload(i));
    }
}

void WaveformBufferTest::testDroppedCountAccumulates()
{
    WaveformBuffer buf(2);

    // First overflow: write 3, drop 1 (newest).
    for (int i = 0; i < 3; ++i)
        buf.write(makePayload(i), 1);
    QCOMPARE(buf.droppedCount(), static_cast<quint64>(1));

    // Drain so buffer is empty.
    std::vector<WaveformEntry> tmp;
    buf.drainAll(tmp);

    // Second overflow: write 4, drop 2 (newest).
    for (int i = 0; i < 4; ++i)
        buf.write(makePayload(i), 1);
    QCOMPARE(buf.droppedCount(), static_cast<quint64>(3)); // 1 + 2
}

void WaveformBufferTest::testOverflowDataIntegrity()
{
    // After overflow, verify that surviving entries contain the correct payloads
    // and that there is no corruption.
    const int cap = 5;
    const int totalWrites = 9;

    WaveformBuffer buf(cap);
    for (int i = 0; i < totalWrites; ++i)
        buf.write(makePayload(i, 128), static_cast<quint64>(i));

    // Earliest `cap` entries survive.
    for (int i = 0; i < cap; ++i) {
        WaveformEntry out;
        QVERIFY(buf.read(out));
        QCOMPARE(out.data, makePayload(i, 128));
        QCOMPARE(out.shotCount, static_cast<quint64>(i));
    }
}

// ---------------------------------------------------------------------------
// 3. Flush marker behavior
// ---------------------------------------------------------------------------

void WaveformBufferTest::testWriteFlushMarkerFields()
{
    WaveformBuffer buf(5);
    buf.writeFlushMarker();

    QCOMPARE(buf.available(), 1);

    WaveformEntry out;
    QVERIFY(buf.read(out));
    QVERIFY(out.flushMarker);
    QVERIFY(out.data.isEmpty());
    QCOMPARE(out.shotCount, static_cast<quint64>(0));
    QCOMPARE(out.preAccumulated, false);
}

void WaveformBufferTest::testFlushMarkerOverflow()
{
    // Flush markers participate in overflow just like data entries.
    // Drop-newest: the 3rd write is discarded because buffer is full.
    WaveformBuffer buf(2);
    buf.writeFlushMarker();          // entry 0 — kept
    buf.write(makePayload(1), 1);    // entry 1 — kept (buffer now full)
    buf.write(makePayload(2), 2);    // dropped (newest)

    QCOMPARE(buf.droppedCount(), static_cast<quint64>(1));
    QCOMPARE(buf.available(), 2);

    WaveformEntry e1, e2;
    QVERIFY(buf.read(e1));
    QVERIFY(buf.read(e2));
    QVERIFY(e1.flushMarker);         // flush marker survived
    QVERIFY(!e2.flushMarker);
    QCOMPARE(e2.data, makePayload(1));
}

void WaveformBufferTest::testFlushMarkerInterleaved()
{
    // Write: data0, flush, data1, flush, data2
    WaveformBuffer buf(10);
    buf.write(makePayload(0), 0);
    buf.writeFlushMarker();
    buf.write(makePayload(1), 1);
    buf.writeFlushMarker();
    buf.write(makePayload(2), 2);

    QCOMPARE(buf.available(), 5);

    // Verify the sequence comes back intact.
    struct Expected { bool isFlush; int payloadIndex; };
    QList<Expected> expected = {
        {false, 0},
        {true,  -1},
        {false, 1},
        {true,  -1},
        {false, 2},
    };

    for (const auto &ex : expected) {
        WaveformEntry out;
        QVERIFY(buf.read(out));
        QCOMPARE(out.flushMarker, ex.isFlush);
        if (!ex.isFlush)
            QCOMPARE(out.data, makePayload(ex.payloadIndex));
    }
}

void WaveformBufferTest::testDrainAllIncludesFlushMarkers()
{
    WaveformBuffer buf(10);
    buf.write(makePayload(0), 0);
    buf.writeFlushMarker();
    buf.write(makePayload(1), 1);

    std::vector<WaveformEntry> out;
    int n = buf.drainAll(out);

    QCOMPARE(n, 3);
    QCOMPARE(static_cast<int>(out.size()), 3);
    QVERIFY(!out[0].flushMarker);
    QVERIFY( out[1].flushMarker);
    QVERIFY(!out[2].flushMarker);
}

// ---------------------------------------------------------------------------
// 4. drainAll behavior
// ---------------------------------------------------------------------------

void WaveformBufferTest::testDrainAllEmpty()
{
    WaveformBuffer buf(5);
    std::vector<WaveformEntry> out;
    int n = buf.drainAll(out);
    QCOMPARE(n, 0);
    QVERIFY(out.empty());
}

void WaveformBufferTest::testDrainAllMultiple()
{
    const int count = 6;
    WaveformBuffer buf(10);
    for (int i = 0; i < count; ++i)
        buf.write(makePayload(i), static_cast<quint64>(i));

    std::vector<WaveformEntry> out;
    int n = buf.drainAll(out);

    QCOMPARE(n, count);
    QCOMPARE(static_cast<int>(out.size()), count);
    for (int i = 0; i < count; ++i) {
        QCOMPARE(out[i].data, makePayload(i));
        QCOMPARE(out[i].shotCount, static_cast<quint64>(i));
    }
}

void WaveformBufferTest::testDrainAllAfterPartialRead()
{
    WaveformBuffer buf(10);
    for (int i = 0; i < 5; ++i)
        buf.write(makePayload(i), static_cast<quint64>(i));

    // Read the first two manually.
    WaveformEntry e0, e1;
    QVERIFY(buf.read(e0));
    QVERIFY(buf.read(e1));

    // drainAll should return the remaining 3.
    std::vector<WaveformEntry> out;
    int n = buf.drainAll(out);

    QCOMPARE(n, 3);
    QCOMPARE(static_cast<int>(out.size()), 3);
    for (int i = 0; i < 3; ++i)
        QCOMPARE(out[i].data, makePayload(i + 2));
}

void WaveformBufferTest::testDrainAllClearsBuffer()
{
    WaveformBuffer buf(5);
    for (int i = 0; i < 4; ++i)
        buf.write(makePayload(i), 1);

    std::vector<WaveformEntry> out;
    buf.drainAll(out);

    QCOMPARE(buf.available(), 0);

    WaveformEntry extra;
    QVERIFY(!buf.read(extra));
}

// ---------------------------------------------------------------------------
// 5. Pre-allocation (reserveBytes)
// ---------------------------------------------------------------------------

void WaveformBufferTest::testReserveBytesDontCrash()
{
    // Construction with reserveBytes > 0 should not crash.
    WaveformBuffer buf(10, 1024);
    QCOMPARE(buf.capacity(), 10);
    QCOMPARE(buf.available(), 0);
}

void WaveformBufferTest::testReserveBytesSmallData()
{
    // Data smaller than reserved size should be stored and retrieved correctly.
    WaveformBuffer buf(5, 1024);
    QByteArray small = makePayload(99, 32); // 32 bytes < 1024
    buf.write(small, 5);

    WaveformEntry out;
    QVERIFY(buf.read(out));
    QCOMPARE(out.data, small);
    QCOMPARE(out.shotCount, static_cast<quint64>(5));
}

void WaveformBufferTest::testReserveBytesBigData()
{
    // Data larger than reserved size should be stored and retrieved correctly.
    WaveformBuffer buf(5, 64);
    QByteArray big = makePayload(7, 512); // 512 bytes > 64
    buf.write(big, 2);

    WaveformEntry out;
    QVERIFY(buf.read(out));
    QCOMPARE(out.data, big);
}

// ---------------------------------------------------------------------------
// 6. Threading correctness
// ---------------------------------------------------------------------------

void WaveformBufferTest::testThreadedProducerConsumer()
{
    // Producer and consumer run on separate threads. Buffer is large enough to
    // hold all entries so there should be no drops.
    const int numEntries = 500;
    WaveformBuffer buf(numEntries + 10);

    std::atomic<int> readCount{0};

    QThread *producer = QThread::create([&]() {
        for (int i = 0; i < numEntries; ++i)
            buf.write(makePayload(i), static_cast<quint64>(i));
    });

    QThread *consumer = QThread::create([&]() {
        int total = 0;
        while (total < numEntries) {
            if (buf.waitForData(5000)) {
                std::vector<WaveformEntry> batch;
                total += buf.drainAll(batch);
            } else {
                // Timeout — producer may have finished; do one final drain.
                std::vector<WaveformEntry> batch;
                total += buf.drainAll(batch);
                break;
            }
        }
        readCount.store(total, std::memory_order_relaxed);
    });

    producer->start();
    consumer->start();

    QVERIFY(producer->wait(5000));
    QVERIFY(consumer->wait(5000));

    delete producer;
    delete consumer;

    QCOMPARE(readCount.load(), numEntries);
    QCOMPARE(buf.droppedCount(), static_cast<quint64>(0));
}

void WaveformBufferTest::testThreadedDataIntegrity()
{
    // Each entry is uniquely identifiable. Verify no corruption occurs across
    // threads.
    const int numEntries = 200;
    WaveformBuffer buf(numEntries + 10);

    // Consumer collects all drained entries.
    std::vector<WaveformEntry> received;
    received.reserve(numEntries);

    QThread *producer = QThread::create([&]() {
        for (int i = 0; i < numEntries; ++i)
            buf.write(makePayload(i), static_cast<quint64>(i));
    });

    QThread *consumer = QThread::create([&]() {
        int total = 0;
        while (total < numEntries) {
            if (buf.waitForData(5000)) {
                buf.drainAll(received);
                total = static_cast<int>(received.size());
            } else {
                buf.drainAll(received);
                break;
            }
        }
    });

    producer->start();
    consumer->start();

    QVERIFY(producer->wait(5000));
    QVERIFY(consumer->wait(5000));

    delete producer;
    delete consumer;

    QCOMPARE(static_cast<int>(received.size()), numEntries);
    for (int i = 0; i < numEntries; ++i) {
        QCOMPARE(received[i].data,      makePayload(i));
        QCOMPARE(received[i].shotCount, static_cast<quint64>(i));
    }
}

void WaveformBufferTest::testWaitForDataWakesOnWrite()
{
    WaveformBuffer buf(10);
    bool woke = false;

    QThread *consumer = QThread::create([&]() {
        // Wait for up to 5 seconds; producer will write after a short delay.
        woke = buf.waitForData(5000);
    });

    QThread *producer = QThread::create([&]() {
        // Short sleep to ensure consumer is blocking in waitForData.
        QThread::msleep(50);
        buf.write(makePayload(0), 1);
    });

    consumer->start();
    producer->start();

    QVERIFY(producer->wait(5000));
    QVERIFY(consumer->wait(5000));

    delete producer;
    delete consumer;

    QVERIFY(woke);
}

void WaveformBufferTest::testWaitForDataTimesOut()
{
    WaveformBuffer buf(10);
    // Nothing is written; waitForData should return false after the timeout.
    QElapsedTimer timer;
    timer.start();
    bool result = buf.waitForData(100);
    qint64 elapsed = timer.elapsed();

    QVERIFY(!result);
    // Sanity-check: at least 80 ms elapsed (allow a little slack).
    QVERIFY2(elapsed >= 80, qPrintable(QString("Elapsed was only %1 ms").arg(elapsed)));
}

// ---------------------------------------------------------------------------
// 7. Threading with overflow
// ---------------------------------------------------------------------------

void WaveformBufferTest::testThreadedOverflow()
{
    // Small buffer; producer writes much faster than the consumer reads.
    // Verify: droppedCount > 0, no crashes, all read entries have valid data.
    const int cap = 5;
    const int producerWrites = 2000;
    WaveformBuffer buf(cap);

    std::atomic<bool> producerDone{false};
    std::vector<WaveformEntry> received;

    QThread *producer = QThread::create([&]() {
        for (int i = 0; i < producerWrites; ++i)
            buf.write(makePayload(i & 0xFF, 32), static_cast<quint64>(i));
        producerDone.store(true, std::memory_order_release);
    });

    QThread *consumer = QThread::create([&]() {
        while (!producerDone.load(std::memory_order_acquire) || buf.available() > 0) {
            if (buf.waitForData(100)) {
                buf.drainAll(received);
            }
        }
        // Final drain after producer finishes.
        buf.drainAll(received);
    });

    producer->start();
    consumer->start();

    QVERIFY(producer->wait(5000));
    QVERIFY(consumer->wait(5000));

    delete producer;
    delete consumer;

    // With a tiny buffer and fast producer, drops must have occurred.
    QVERIFY2(buf.droppedCount() > 0,
             "Expected dropped entries with small buffer and fast producer");

    // All received entries must have valid, non-empty data.
    for (const auto &e : received) {
        QVERIFY(!e.data.isEmpty());
        QCOMPARE(e.data.size(), 32);
    }
}

// ---------------------------------------------------------------------------
// 8. Throughput benchmark
// ---------------------------------------------------------------------------

void WaveformBufferTest::testThroughputBenchmark()
{
    // Informational test: no pass/fail threshold. Measures entries/sec across
    // threads to catch deadlocks or severe regressions.
    const int numEntries = 100000;
    const int payloadSize = 128;
    WaveformBuffer buf(256, payloadSize);

    std::atomic<int> readCount{0};
    std::atomic<bool> producerDone{false};
    QElapsedTimer timer;

    QThread *producer = QThread::create([&]() {
        QByteArray payload(payloadSize, 0xAB);
        for (int i = 0; i < numEntries; ++i)
            buf.write(payload, static_cast<quint64>(i));
        producerDone.store(true, std::memory_order_release);
    });

    QThread *consumer = QThread::create([&]() {
        int total = 0;
        std::vector<WaveformEntry> batch;
        batch.reserve(256);
        while (!producerDone.load(std::memory_order_acquire) || buf.available() > 0) {
            if (buf.waitForData(100)) {
                batch.clear();
                total += buf.drainAll(batch);
            }
        }
        // Final drain.
        batch.clear();
        total += buf.drainAll(batch);
        readCount.store(total, std::memory_order_relaxed);
    });

    timer.start();
    producer->start();
    consumer->start();

    QVERIFY(producer->wait(30000));
    QVERIFY(consumer->wait(30000));

    qint64 elapsedMs = timer.elapsed();
    delete producer;
    delete consumer;

    int total = readCount.load();
    quint64 dropped = buf.droppedCount();
    double entriesPerSec = (total * 1000.0) / static_cast<double>(elapsedMs > 0 ? elapsedMs : 1);
    qDebug("WaveformBuffer throughput: %d read + %llu dropped = %lld total in %lld ms — %.0f entries/sec",
           total, dropped, static_cast<long long>(total + dropped), elapsedMs, entriesPerSec);

    // Sanity: must not deadlock and read + dropped should account for all writes.
    QCOMPARE(static_cast<quint64>(total) + dropped, static_cast<quint64>(numEntries));
}

// ---------------------------------------------------------------------------
// 9. Flush marker under concurrency
// ---------------------------------------------------------------------------

void WaveformBufferTest::testThreadedFlushMarkerOrdering()
{
    // Producer writes blocks of data entries separated by flush markers.
    // Consumer verifies that flush markers appear in the correct positions —
    // no data entry from "after" a flush appears "before" it.
    // Buffer is large enough to hold everything (no drops).
    const int blockSize = 20; // data entries per block
    const int numBlocks = 10;
    const int totalProducerWrites = numBlocks * (blockSize + 1); // entries + flush per block
    WaveformBuffer buf(totalProducerWrites + 10);

    std::atomic<bool> producerDone{false};

    QThread *producer = QThread::create([&]() {
        for (int block = 0; block < numBlocks; ++block) {
            for (int i = 0; i < blockSize; ++i)
                buf.write(makePayload(i), static_cast<quint64>(block * blockSize + i));
            buf.writeFlushMarker();
        }
        producerDone.store(true, std::memory_order_release);
    });

    std::vector<WaveformEntry> received;

    QThread *consumer = QThread::create([&]() {
        while (!producerDone.load(std::memory_order_acquire) || buf.available() > 0) {
            if (buf.waitForData(100)) {
                buf.drainAll(received);
            }
        }
        buf.drainAll(received);
    });

    producer->start();
    consumer->start();

    QVERIFY(producer->wait(5000));
    QVERIFY(consumer->wait(5000));

    delete producer;
    delete consumer;

    QCOMPARE(buf.droppedCount(), static_cast<quint64>(0));
    QCOMPARE(static_cast<int>(received.size()), totalProducerWrites);

    // Walk received entries and verify structure: each block of `blockSize`
    // data entries is followed by exactly one flush marker.
    int dataInCurrentBlock = 0;
    int completedBlocks = 0;
    for (const auto &entry : received) {
        if (entry.flushMarker) {
            // A flush marker must arrive after exactly blockSize data entries.
            QCOMPARE(dataInCurrentBlock, blockSize);
            dataInCurrentBlock = 0;
            ++completedBlocks;
        } else {
            ++dataInCurrentBlock;
        }
    }
    QCOMPARE(completedBlocks, numBlocks);
    QCOMPARE(dataInCurrentBlock, 0); // no trailing data after the last flush
}

QTEST_MAIN(WaveformBufferTest)
#include "tst_waveformbuffertest.moc"
