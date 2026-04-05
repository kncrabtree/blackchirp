/*!
 * \file waveformbuffer.cpp
 * \brief Implementation of WaveformBuffer — bounded SPSC ring buffer.
 *
 * See waveformbuffer.h for the full thread safety contract and API documentation.
 *
 * Index arithmetic notes
 * ----------------------
 * d_writeIndex and d_readIndex are monotonically increasing values; slots are
 * addressed as (index % d_capacity).  This avoids the ABA problem that arises
 * when indices are kept in [0, capacity) and wrap around.
 *
 * Semaphore discipline
 * --------------------
 * The semaphore is used only as a wake-up mechanism for the consumer, not as
 * a precise count of readable entries.  One release(1) is performed for every
 * successful write.  Dropped entries (buffer full) do not release the semaphore.
 *
 * waitForData() consumes one permit via tryAcquire, then immediately re-releases
 * it.  drainAll()/read() use the atomic indices to determine actual availability.
 */

#include <data/storage/waveformbuffer.h>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

WaveformBuffer::WaveformBuffer(int capacity, qint64 reserveBytes)
    : d_capacity(capacity > 0 ? capacity : 1),
      d_slots(d_capacity)
{
    if (reserveBytes > 0) {
        for (auto &slot : d_slots)
            slot.data.reserve(static_cast<int>(reserveBytes));
    }
}

// ---------------------------------------------------------------------------
// Producer API
// ---------------------------------------------------------------------------

void WaveformBuffer::write(const QByteArray &data, quint64 shotCount, bool preAccumulated)
{
    WaveformEntry entry;
    entry.shotCount = shotCount;
    entry.preAccumulated = preAccumulated;
    entry.flushMarker = false;

    // Reuse the slot's existing QByteArray allocation when possible to avoid
    // heap churn.  The slot we're about to fill may still hold data from a
    // previous round (its QByteArray capacity is preserved after a std::move
    // in read(), because move leaves the source in a valid-but-unspecified
    // state — Qt's QByteArray move constructor clears the source, so capacity
    // is lost there).  We therefore do the copy into a local entry and let
    // writeEntry move it in; for the reuse optimisation we would need direct
    // slot access before the entry is constructed.  The straightforward
    // approach is used here: assign into the local entry and move.
    entry.data = data;

    writeEntry(std::move(entry));
}

void WaveformBuffer::writeFlushMarker()
{
    WaveformEntry entry;
    entry.shotCount = 0;
    entry.preAccumulated = false;
    entry.flushMarker = true;
    // data stays empty

    writeEntry(std::move(entry));
}

bool WaveformBuffer::isFull() const
{
    int w = d_writeIndex.load(std::memory_order_relaxed);
    int r = d_readIndex.load(std::memory_order_acquire);
    return (w - r) >= d_capacity;
}

void WaveformBuffer::reset()
{
    // Drain any leftover semaphore permits so the consumer will not see stale
    // wake-ups from a previous acquisition run.
    d_semaphore.tryAcquire(d_semaphore.available());

    d_writeIndex.store(0, std::memory_order_relaxed);
    d_readIndex.store(0, std::memory_order_relaxed);
    d_droppedCount.store(0, std::memory_order_relaxed);

    // Clear slot contents but keep QByteArray capacity for reuse.
    for (auto &slot : d_slots) {
        slot.data.clear();      // keeps reserved capacity
        slot.shotCount = 0;
        slot.preAccumulated = false;
        slot.flushMarker = false;
    }
}

// ---------------------------------------------------------------------------
// Consumer API
// ---------------------------------------------------------------------------

bool WaveformBuffer::read(WaveformEntry &out)
{
    int r = d_readIndex.load(std::memory_order_acquire);
    int w = d_writeIndex.load(std::memory_order_acquire);

    if (r == w)
        return false;   // buffer empty

    int slotIndex = r % d_capacity;
    out.shotCount = d_slots[slotIndex].shotCount;
    out.preAccumulated = d_slots[slotIndex].preAccumulated;
    out.flushMarker = d_slots[slotIndex].flushMarker;
    out.data = std::move(d_slots[slotIndex].data);  // avoid copy; slot reused on next write

    // Advance the read index after we have finished reading the slot.
    d_readIndex.store(r + 1, std::memory_order_release);
    return true;
}

int WaveformBuffer::available() const
{
    int w = d_writeIndex.load(std::memory_order_acquire);
    int r = d_readIndex.load(std::memory_order_acquire);
    int diff = w - r;
    return diff > 0 ? diff : 0;
}

bool WaveformBuffer::waitForData(int timeoutMs)
{
    // Fast path: check without touching the semaphore.
    if (available() > 0)
        return true;

    // Block on the semaphore.  If we acquire a permit it means at least one
    // entry was written since the last time we checked.
    if (d_semaphore.tryAcquire(1, timeoutMs)) {
        // Put the permit back so that drainAll() / read() can proceed based
        // solely on the atomic indices rather than needing to account for the
        // consumed permit.  The semaphore is used only for wakeup, not as an
        // accurate count of readable entries.
        d_semaphore.release(1);
        return true;
    }

    // Timeout — do a final index check in case entries arrived between the
    // tryAcquire timeout and now.
    return available() > 0;
}

int WaveformBuffer::drainAll(std::vector<WaveformEntry> &out)
{
    int drained = 0;
    WaveformEntry entry;
    while (read(entry)) {
        out.push_back(std::move(entry));
        ++drained;
    }
    return drained;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

quint64 WaveformBuffer::droppedCount() const
{
    return d_droppedCount.load(std::memory_order_relaxed);
}

int WaveformBuffer::capacity() const
{
    return d_capacity;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void WaveformBuffer::writeEntry(WaveformEntry &&entry)
{
    int w = d_writeIndex.load(std::memory_order_relaxed);
    int r = d_readIndex.load(std::memory_order_acquire);

    // Drop-newest: if the buffer is full, discard the incoming entry.
    // The producer never touches d_readIndex, so there is no race with
    // the consumer.
    if ((w - r) >= d_capacity) {
        d_droppedCount.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    int slotIndex = w % d_capacity;
    d_slots[slotIndex] = std::move(entry);

    // Publish the write by advancing the write index.
    d_writeIndex.store(w + 1, std::memory_order_release);

    // Notify the consumer that new data is available.
    d_semaphore.release(1);
}
