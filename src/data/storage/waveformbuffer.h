#ifndef WAVEFORMBUFFER_H
#define WAVEFORMBUFFER_H

/*!
 * \file waveformbuffer.h
 * \brief Bounded, thread-safe SPSC ring buffer for digitizer waveform data.
 */

#include <atomic>
#include <vector>

#include <QByteArray>
#include <QSemaphore>
#include <QVector>

/*!
 * \brief A single entry in the waveform ring buffer.
 *
 * For normal waveform data, data contains the raw bytes and shotCount holds
 * the number of shots/FIDs represented. For segment boundary sentinels,
 * flushMarker is true and data is empty.
 */
struct WaveformEntry {
    QByteArray data;                ///< Raw waveform bytes (empty for flush markers)
    quint64 shotCount{0};           ///< Number of shots/FIDs represented by this entry
    bool preAccumulated{false};     ///< If true, data contains qint64 values (pre-summed)
    bool flushMarker{false};        ///< If true, this is a segment boundary sentinel
};

/*!
 * \brief Bounded SPSC ring buffer for transferring waveform data
 * between threads.
 *
 * Single-producer/single-consumer: the digitizer hardware thread is
 * the producer (\c write(), \c writeFlushMarker(), \c reset()) and
 * the acquisition manager thread is the consumer (\c read(),
 * \c drainAll(), \c available(), \c waitForData()). Multi-producer
 * or multi-consumer usage is undefined behavior. The producer never
 * blocks; when the buffer is full the incoming entry is silently
 * dropped (\c droppedCount() exposes the count). Slot storage is
 * pre-allocated at construction; pass \c reserveBytes > 0 to also
 * pre-reserve each slot's QByteArray and avoid per-shot heap
 * allocation. Does not inherit from QObject.
 */
class WaveformBuffer
{
public:
    /*!
     * \brief Construct a WaveformBuffer.
     * \param capacity Number of slots in the ring buffer (fixed at creation).
     * \param reserveBytes If > 0, pre-reserve each slot's QByteArray to this
     *        size to avoid per-shot heap allocation during steady-state operation.
     */
    explicit WaveformBuffer(int capacity = 10, qint64 reserveBytes = 0);
    ~WaveformBuffer() = default;

    // Non-copyable, non-movable (contains atomics and QSemaphore)
    WaveformBuffer(const WaveformBuffer &) = delete;
    WaveformBuffer &operator=(const WaveformBuffer &) = delete;
    WaveformBuffer(WaveformBuffer &&) = delete;
    WaveformBuffer &operator=(WaveformBuffer &&) = delete;

    // -----------------------------------------------------------------------
    // Producer API — call from digitizer thread only
    // -----------------------------------------------------------------------

    /*!
     * \brief Write a waveform entry into the buffer.
     *
     * If the buffer is full, the incoming entry is discarded (drop-newest
     * policy) and the dropped counter is incremented. This call never blocks.
     *
     * \param data       Raw waveform bytes.
     * \param shotCount  Number of shots/FIDs represented by this entry.
     * \param preAccumulated If true, data contains qint64 values (pre-summed).
     */
    void write(const QByteArray &data, quint64 shotCount, bool preAccumulated = false);

    /*!
     * \brief Write a segment boundary sentinel into the buffer.
     *
     * The entry has empty data and flushMarker == true. Same drop-newest
     * overflow behavior as write().
     */
    void writeFlushMarker();

    /*!
     * \brief Check if the buffer is full (producer-side query).
     *
     * Uses the same logic as writeEntry's overflow check. This allows
     * the producer to decide whether to begin pre-accumulation before
     * attempting a write.
     *
     * \return true if (writeIndex - readIndex) >= capacity.
     */
    bool isFull() const;

    /*!
     * \brief Clear all entries and reset indices.
     *
     * Call at the start of acquisition. This is NOT thread-safe with concurrent
     * producer/consumer activity — call only when both threads are quiescent.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Consumer API — call from acquisition manager thread only
    // -----------------------------------------------------------------------

    /*!
     * \brief Non-blocking read of a single entry.
     *
     * Moves data out of the ring slot to avoid a copy. Returns false immediately
     * if no entries are available.
     *
     * \param out Receives the entry on success.
     * \return true if an entry was available and written to \a out.
     */
    bool read(WaveformEntry &out);

    /*!
     * \brief Number of entries currently available for reading.
     * \return Count of unread entries.
     */
    int available() const;

    /*!
     * \brief Block until data is available or the timeout expires.
     *
     * Uses QSemaphore::tryAcquire internally. After this returns true, use
     * drainAll() or read() to consume the entries.
     *
     * \param timeoutMs Maximum time to wait in milliseconds (default 100 ms).
     * \return true if at least one entry is available.
     */
    bool waitForData(int timeoutMs = 100);

    /*!
     * \brief Drain all available entries into \a out.
     *
     * More efficient than calling read() in a loop. Appends to \a out rather
     * than clearing it first, so the caller may pre-size the vector if desired.
     * This is the primary consumption API.
     *
     * \param out Vector to append drained entries to.
     * \return Number of entries drained.
     */
    int drainAll(std::vector<WaveformEntry> &out);

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    /*!
     * \brief Number of entries dropped due to overflow since last reset().
     * \return Dropped entry count.
     */
    quint64 droppedCount() const;

    /*!
     * \brief Ring buffer capacity (number of slots).
     * \return Capacity set at construction.
     */
    int capacity() const;

private:
    /*!
     * \brief Internal write implementation shared by write() and writeFlushMarker().
     *
     * Handles slot reuse, overflow detection, index advancement, and semaphore
     * release.
     *
     * \param entry Entry to write (moved into the slot).
     */
    void writeEntry(WaveformEntry &&entry);

    const int d_capacity;                   ///< Fixed ring buffer capacity
    QVector<WaveformEntry> d_slots;         ///< Pre-allocated ring slot storage

    std::atomic<int> d_writeIndex{0};       ///< Next slot to write (producer-owned)
    std::atomic<int> d_readIndex{0};        ///< Next slot to read (consumer-owned)
    std::atomic<quint64> d_droppedCount{0}; ///< Overflow drop counter

    QSemaphore d_semaphore;                 ///< Consumer notification semaphore
};

#endif // WAVEFORMBUFFER_H
