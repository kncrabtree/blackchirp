#ifndef CHIRPCONFIG_H
#define CHIRPCONFIG_H

#include <QtGlobal>
#include <QPointF>
#include <QVector>
#include <QMap>
#include <QVariant>
#include <QString>

#include <data/storage/headerstorage.h>

/// \brief Persistence keys for \c ChirpConfig header fields and loadout serialization.
namespace BC::Store::CC {
inline constexpr QLatin1StringView key{"ChirpConfig"};          ///< Header section key.
inline constexpr QLatin1StringView interval{"ChirpInterval"};   ///< Inter-chirp interval in microseconds.
inline constexpr QLatin1StringView sampleRate{"SampleRate"};    ///< AWG sample rate in MHz (samples per microsecond).
inline constexpr QLatin1StringView sampleInterval{"SampleInterval"}; ///< AWG sample interval in microseconds.

// Loadout serialization keys
inline constexpr QLatin1StringView numChirps{"NumChirps"};       ///< Total number of chirps in the waveform.
inline constexpr QLatin1StringView allIdentical{"AllChirpsIdentical"}; ///< Whether all chirps share the same segment list.
inline constexpr QLatin1StringView chirpIndex{"ChirpIndex"};     ///< Zero-based chirp index in the segment array.
inline constexpr QLatin1StringView segmentIndex{"SegmentIndex"}; ///< Zero-based segment index within a chirp.
inline constexpr QLatin1StringView startFreqMHz{"StartFreqMHz"}; ///< Segment start frequency in MHz.
inline constexpr QLatin1StringView endFreqMHz{"EndFreqMHz"};     ///< Segment end frequency in MHz.
inline constexpr QLatin1StringView durationUs{"DurationUs"};     ///< Segment duration in microseconds.
inline constexpr QLatin1StringView alphaUs{"AlphaUs"};           ///< Chirp rate in MHz per microsecond.
inline constexpr QLatin1StringView segEmpty{"Empty"};            ///< Whether the segment is a silent gap.
inline constexpr QLatin1StringView markerName{"Name"};           ///< Marker channel user label.
inline constexpr QLatin1StringView markerRole{"Role"};           ///< Marker channel role string.
inline constexpr QLatin1StringView timingMode{"TimingMode"};     ///< Marker timing mode string.
inline constexpr QLatin1StringView markerStart{"StartTime"};     ///< Marker start time in microseconds.
inline constexpr QLatin1StringView markerEnd{"EndTime"};         ///< Marker end time in microseconds.
inline constexpr QLatin1StringView markerEnabled{"Enabled"};     ///< Whether the marker channel is active.
}

/// \brief Role classification for an AWG marker channel.
///
/// The role drives safety validation in the experiment wizard:
/// \c Protection and \c Gate roles are checked to ensure the protection
/// pulse encloses the gate and chirp windows, preventing high-power
/// energy from reaching sensitive amplifiers. \c Trigger marks a
/// general digitizer or instrument trigger. \c Custom imposes no
/// safety constraints.
///
/// \sa MarkerChannel, ChirpConfig::findEnabledMarkerByRole()
enum class MarkerRole {
    Protection, ///< Safety interlock: must enclose the Gate window and the chirp. Absence triggers a wizard warning.
    Gate,       ///< Amplifier-enable pulse. Must be enclosed by the Protection marker.
    Trigger,    ///< General digitizer or instrument trigger.
    Custom      ///< Any other use. No safety constraints are enforced.
};

/// \brief Descriptor for a single AWG marker output channel.
///
/// Each \c MarkerChannel maps to one physical marker output on the AWG.
/// The \c timingMode determines how \c startTime and \c endTime are
/// interpreted:
/// - \c ChirpRelative: \c startTime is measured in µs relative to each
///   chirp's start (negative values extend before the chirp); \c endTime
///   is measured relative to each chirp's end (positive values extend
///   after the chirp). The marker repeats for every chirp interval.
/// - \c Absolute: reserved for future use; storage and readback are
///   supported but the \c Absolute branch is not yet honored by the
///   waveform generator.
///
/// The lead time before each chirp interval and the tail time after
/// it are derived from the maximum required \c startTime and \c endTime
/// values across all enabled channels.
///
/// \sa MarkerRole, ChirpConfig::markerChannels(), ChirpConfig::setMarkerChannels()
struct MarkerChannel {
    QString name;                          ///< User-visible label for this marker channel.
    /// \brief Timing reference for \c startTime and \c endTime.
    enum TimingMode {
        Absolute,      ///< Times are measured from the first chirp start (reserved; not yet honored by waveform generation).
        ChirpRelative  ///< Times are measured relative to each chirp's start/end, repeating per interval.
    };
    TimingMode timingMode{ChirpRelative};  ///< Timing mode; \c ChirpRelative by default.
    double startTime{-0.5};  ///< Start offset in µs relative to the chirp start (ChirpRelative) or waveform origin (Absolute). Negative values precede the chirp.
    double endTime{0.5};     ///< End offset in µs relative to the chirp end (ChirpRelative) or waveform origin (Absolute). Positive values follow the chirp.
    bool enabled{true};      ///< Whether this channel contributes to the generated waveform.
    MarkerRole role{MarkerRole::Custom}; ///< Role classification used for safety validation.
};

class BlackchirpCSV;

/// \brief Chirp waveform and marker configuration.
///
/// \c ChirpConfig stores the complete definition of the AWG output waveform:
/// the segment list for each chirp in a multi-chirp sequence, the
/// inter-chirp interval, the AWG sample rate, and the marker channel
/// definitions. All time values are in microseconds; all frequency values
/// are in MHz.
///
/// The chirp waveform is a vector of chirp intervals, each containing an
/// ordered list of \c ChirpSegment records. Segments may be linear-frequency
/// sweeps or silent gaps (\c empty = true). Each chirp's total duration is
/// the sum of its segment durations. If all chirps have identical segments,
/// \c allChirpsIdentical() returns true, which allows \c RfConfig to
/// optimize sweep accounting.
///
/// Marker channels are defined by a \c QVector<MarkerChannel>; the AWG
/// reports its physical marker count via the \c BC::Key::AWG::markerCount
/// setting. The waveform generation methods (\c getMarkerData(),
/// \c getPackedMarkerData()) produce output indexed by the logical channel
/// order; each AWG driver remaps logical bit positions to physical output
/// positions as required by its hardware data format.
///
/// \note The AWG sample rate must be set via \c setAwgSampleRate() before
/// calling any waveform generation method; it is not persisted in the header
/// and is reconstructed from the active AWG profile on load.
///
/// \sa RfConfig, MarkerChannel, MarkerRole
class ChirpConfig : public HeaderStorage
{
public:
    /// \brief Single frequency-sweep or gap segment within a chirp.
    struct ChirpSegment {
        double startFreqMHz{0.0}; ///< Start frequency in MHz; ignored when \c empty is true.
        double endFreqMHz{0.0};   ///< End frequency in MHz; ignored when \c empty is true.
        double durationUs{0.0};   ///< Segment duration in microseconds.
        double alphaUs{0.0};      ///< Chirp rate in MHz/µs, computed as (endFreqMHz - startFreqMHz) / durationUs.
        bool empty{false};        ///< When true the segment generates zero-amplitude samples (a gap).
    };

    /// \brief Constructs a default \c ChirpConfig with no segments and no marker channels.
    ChirpConfig();
    ~ChirpConfig();

    /// \brief Reads the chirp segment CSV file for experiment \a num.
    ///
    /// Populates \c d_chirpList from the \c chirps.csv file located under the
    /// experiment directory for \a num. If \a path is empty the default data
    /// directory is used.
    void readChirpFile(BlackchirpCSV *csv, int num, QString path = QString(""));

    /// \brief Writes the chirp segment CSV file for experiment \a num.
    ///
    /// Returns false if the file cannot be opened for writing.
    bool writeChirpFile(int num) const;

    /// \brief Reads the marker channel CSV file for experiment \a num.
    ///
    /// Replaces the current marker channel list with the contents of
    /// \c markers.csv. If \a path is empty the default data directory is used.
    void readMarkersFile(BlackchirpCSV *csv, int num, QString path = QString(""));

    /// \brief Writes the marker channel CSV file for experiment \a num.
    ///
    /// Returns false if the file cannot be opened for writing.
    bool writeMarkersFile(int num) const;

    /// \brief Returns the waveform lead time in µs before the first chirp start.
    ///
    /// Computed as \c max(0, max(-m.startTime)) over all enabled channels with
    /// \c ChirpRelative timing. This is the pre-roll added before each chirp
    /// interval to accommodate markers that begin before the chirp itself.
    double leadTimeUs() const;

    /// \brief Returns the waveform tail time in µs after each chirp end.
    ///
    /// Computed as \c max(0, max(m.endTime)) over all enabled channels with
    /// \c ChirpRelative timing.
    double tailTimeUs() const;

    /// \brief Returns the number of chirps in the sequence.
    int numChirps() const;

    /// \brief Returns the inter-chirp interval in µs.
    ///
    /// The interval is measured from the start of one chirp window to the start
    /// of the next, including the lead time. A value of -1 indicates the interval
    /// has not been set.
    double chirpInterval() const;

    /// \brief Returns true if every chirp in the sequence has identical segment lists.
    bool allChirpsIdentical() const;

    /// \brief Returns a copy of the full chirp segment list (chirps × segments).
    QVector<QVector<ChirpSegment>> chirpList() const;

    /// \brief Returns the start frequency in MHz for the given \a chirp and \a segment indices.
    ///
    /// Returns -1.0 if either index is out of range.
    double segmentStartFreq(int chirp, int segment) const;

    /// \brief Returns the end frequency in MHz for the given \a chirp and \a segment indices.
    ///
    /// Returns -1.0 if either index is out of range.
    double segmentEndFreq(int chirp, int segment) const;

    /// \brief Returns the duration in µs for the given \a chirp and \a segment indices.
    ///
    /// Returns -1.0 if either index is out of range.
    double segmentDuration(int chirp, int segment) const;

    /// \brief Returns true if the segment at the given indices is a silent gap.
    ///
    /// Returns true if either index is out of range.
    bool segmentEmpty(int chirp, int segment) const;

    /// \brief Returns a SHA-256 hash of the waveform parameters.
    ///
    /// The hash covers the marker channel definitions (name, role, start time,
    /// end time, enabled state) and all chirp segment parameters (start and end
    /// frequencies, duration, empty flag), along with the chirp count and interval.
    /// Two \c ChirpConfig objects with identical hashes will produce identical AWG
    /// waveforms. The sample rate is excluded because it is hardware-derived and not
    /// recorded in the stored configuration.
    QByteArray waveformHash() const;

    /// \brief Returns the total duration of chirp \a chirpNum in µs.
    ///
    /// Sums the durations of all segments in that chirp. Returns 0 if \a chirpNum
    /// is out of range.
    double chirpDurationUs(int chirpNum) const;

    /// \brief Returns the total waveform duration in µs.
    ///
    /// Computed as leadTimeUs() + (numChirps() - 1) * chirpInterval() +
    /// chirpDurationUs(last) + tailTimeUs().
    double totalDuration() const;

    /// \brief Returns the chirp waveform as a vector of (time_µs, amplitude) points spanning the full duration.
    QVector<QPointF> getChirpMicroseconds() const;

    /// \brief Returns a slice of the chirp waveform for the time range [\a t1, \a t2) in µs.
    ///
    /// Used by \c ChirpConfigPlot to render a zoomed view without computing the full waveform.
    QVector<QPointF> getChirpSegmentMicroSeconds(double t1, double t2) const;

    /// \brief Returns per-channel marker data as a 2-D boolean array (channels × samples).
    ///
    /// The outer dimension is indexed by logical marker channel index (matching
    /// \c markerChannels()); the inner dimension is indexed by waveform sample.
    /// Only enabled channels contribute non-false entries. The AWG sample rate
    /// must be set via \c setAwgSampleRate() before calling this method.
    QVector<QVector<bool>> getMarkerData() const;

    /// \brief Returns marker data packed into one \c quint32 per sample (bit 0 = channel 0).
    ///
    /// This is a convenience format for AWG drivers that can consume packed data
    /// directly. Each driver is responsible for remapping the logical bit order to
    /// its hardware's physical output bit positions. Requires the AWG sample rate
    /// to be set via \c setAwgSampleRate().
    QVector<quint32> getPackedMarkerData() const;

    /// \brief Returns a const reference to the marker channel list.
    const QVector<MarkerChannel>& markerChannels() const;

    /// \brief Returns a pointer to the first enabled marker channel with the given \a role.
    ///
    /// Returns \c nullptr if no enabled channel with that role exists. Used by
    /// \c FtmwConfig to obtain trigger timing and by the validation page to check
    /// protection/gate enclosure.
    const MarkerChannel* findEnabledMarkerByRole(MarkerRole role) const;

    /// \brief Sets the AWG sample rate from the hardware value in samples per second.
    ///
    /// Converts to the internal representation (samples per µs and µs per sample).
    /// Must be called before any waveform generation method.
    void setAwgSampleRate(const double samplesPerSecond);

    /// \brief Sets the total number of chirps, extending or truncating the chirp list.
    ///
    /// When growing the list, new entries are copied from the first chirp (or left
    /// empty if the list is currently empty). When shrinking, trailing entries are
    /// removed.
    void setNumChirps(const int n);

    /// \brief Sets the inter-chirp interval in µs.
    void setChirpInterval(const double i);

    /// \brief Replaces the entire marker channel list.
    void setMarkerChannels(const QVector<MarkerChannel>& channels);

    /// \brief Appends a frequency-sweep segment to the specified chirp (or all chirps when \a chirpNum < 0).
    ///
    /// Negative values for \a startMHz, \a endMHz, or \a durationUs are ignored.
    /// \param startMHz   Segment start frequency in MHz.
    /// \param endMHz     Segment end frequency in MHz.
    /// \param durationUs Segment duration in µs.
    /// \param chirpNum   Chirp index to append to; -1 appends to all chirps.
    void addSegment(const double startMHz, const double endMHz, const double durationUs, const int chirpNum = -1);

    /// \brief Appends a silent-gap segment of \a durationUs to the specified chirp (or all chirps when \a chirpNum < 0).
    void addEmptySegment(const double durationUs, const int chirpNum = -1);

    /// \brief Replaces the entire chirp segment list with \a l.
    void setChirpList(const QVector<QVector<ChirpSegment> > l);

private:
    QVector<MarkerChannel> d_markerChannels;
    double d_chirpInterval{-1.0}; //units: us


    //working data to improve efficiency; do not record to disk!
    double d_sampleRateSperUS; //awg rate, samples per microsecond
    double d_sampleIntervalUS; //awg sample interval in microseconds

    int getFirstSample(double time) const;
    int getLastSample(double time) const;
    double getSampleTime(const int sample) const;
    double calculateChirp(const ChirpSegment segment, const double t, const double phase) const;
    double calculateEndingPhaseRadians(const ChirpSegment segment, const double endingTime, const double startingPhase) const;

    QVector<QVector<ChirpSegment>> d_chirpList;


    // HeaderStorage interface
protected:
    /// \brief Writes chirp interval and sample rate to the header storage tree.
    void storeValues() override;

    /// \brief Reads chirp interval and sample rate back from the header storage tree.
    void retrieveValues() override;
};

Q_DECLARE_TYPEINFO(ChirpConfig::ChirpSegment,Q_PRIMITIVE_TYPE);

#endif // CHIRPCONFIG_H
