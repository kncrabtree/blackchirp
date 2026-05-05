#ifndef RFCONFIG_H
#define RFCONFIG_H

#include <QSharedDataPointer>

#include <data/storage/headerstorage.h>
#include <data/experiment/chirpconfig.h>

/// \brief Persistence keys for \c RfConfig header fields.
namespace BC::Store::RFC {
inline constexpr QLatin1StringView key{"RfConfig"};              ///< Header section key.
inline constexpr QLatin1StringView commonLO{"CommonUpDownLO"};   ///< Whether a single LO drives both up- and downconversion.
inline constexpr QLatin1StringView targetSweeps{"TargetSweeps"}; ///< Number of complete LO/DR sweeps to acquire.
inline constexpr QLatin1StringView shots{"ShotsPerClockConfig"}; ///< Shots co-averaged at each clock configuration step.
inline constexpr QLatin1StringView awgM{"AwgMult"};              ///< AWG frequency multiplication factor.
inline constexpr QLatin1StringView upSB{"UpconversionSideband"}; ///< Upconversion mixer sideband selection.
inline constexpr QLatin1StringView chirpM{"ChirpMult"};          ///< Post-upconversion frequency multiplication factor.
inline constexpr QLatin1StringView downSB{"DownconversionSideband"}; ///< Downconversion mixer sideband selection.
}

/// \brief RF/microwave chain configuration bridging \c FtmwConfig and \c ChirpConfig.
///
/// \c RfConfig encapsulates the complete RF frequency chain for an FTMW
/// experiment: the upconversion path (AWG output → mixer → optional multiplier
/// → sample), the downconversion path (received signal → mixer), the clock
/// assignments for up-LO, down-LO, AWG reference, digital-receiver reference,
/// common reference, and double-resonance (DR) sources, the target sweep count,
/// and the embedded \c ChirpConfig.
///
/// Clock frequencies are stored in two layers:
///
/// - A *clock template* (\c d_clockTemplate) holds the single-point frequency
///   configuration used for standard (non-scan) acquisitions and serves as the
///   base from which scan steps are derived.
/// - A *clock step list* (\c d_clockConfigs) holds one hash per acquisition
///   segment; for LO and DR scans each element differs only in the scanned
///   clock's desired frequency. The list is empty for non-scan acquisitions,
///   in which case \c prepareForAcquisition() initializes it from the template.
///
/// \c clockFrequency() and \c getClocks() always return from the active step
/// (indexed by \c d_currentClockIndex). \c rawClockFrequency() converts the
/// desired output frequency back to the hardware oscillator frequency via the
/// multiplication or division factor stored in each \c ClockFreq record.
///
/// The chirp frequency at the sample for a given AWG output frequency is:
///
///   \verbatim
///   chirpFreq = (awgFreq × awgMult ± upLO) × chirpMult
///   \endverbatim
///
/// where the sign depends on \c d_upMixSideband (upper = +, lower = −).
/// The received signal is mixed with the down-LO and the digitizer captures the
/// IF; \c d_downMixSideband selects which mixing product is the signal.
/// \c calculateChirpFreq() and \c calculateAwgFreq() perform these conversions.
///
/// \sa ChirpConfig, FtmwConfig
class RfConfig : public HeaderStorage
{
    Q_GADGET
public:
    /// \brief Whether a clock frequency is multiplied or divided to reach the desired output.
    enum MultOperation {
        Multiply, ///< Output frequency = oscillator frequency × factor.
        Divide    ///< Output frequency = oscillator frequency / factor.
    };
    Q_ENUM(MultOperation)

    /// \brief Mixer sideband selection for up- and downconversion.
    enum Sideband {
        UpperSideband, ///< Sum product: IF = RF + LO.
        LowerSideband  ///< Difference product: IF = LO − RF.
    };
    Q_ENUM(Sideband)

    /// \brief Logical role for a clock source in the RF chain.
    enum ClockType {
        UpLO,    ///< Local oscillator for the upconversion mixer.
        DownLO,  ///< Local oscillator for the downconversion mixer.
        AwgRef,  ///< Reference clock input to the AWG.
        DRClock, ///< Double-resonance pump source.
        DigRef,  ///< Reference clock input to the digitizer.
        ComRef   ///< Common 10 MHz (or other) reference distributed to all instruments.
    };
    Q_ENUM(ClockType)

    /// \brief Frequency assignment for one clock role in one acquisition step.
    struct ClockFreq {
        double desiredFreqMHz{0.0}; ///< Target output frequency in MHz.
        MultOperation op{Multiply}; ///< Whether \c factor multiplies or divides to reach \c desiredFreqMHz.
        double factor{1.0};         ///< Multiplier or divisor relating oscillator frequency to output frequency.
        QString hwKey{""};          ///< Hardware key of the clock source (\c "<Type>.<label>").
        int output{0};              ///< Output port index on the clock hardware.
    };

    /// \brief Constructs an \c RfConfig with default single-step, single-sweep settings.
    RfConfig();
    ~RfConfig();

    bool d_commonUpDownLO{false};     ///< When true, \c UpLO and \c DownLO share one hardware source.
    int d_completedSweeps{0};         ///< Number of completed LO/DR sweeps (updated by \c advanceClockStep()).
    int d_targetSweeps{1};            ///< Number of sweeps to complete before the experiment ends.
    int d_shotsPerClockConfig{0};     ///< Shots co-averaged at each clock step before advancing.

    //Upconversion chain
    double d_awgMult{1.0};            ///< Multiplication factor applied to the AWG output frequency.
    Sideband d_upMixSideband{UpperSideband}; ///< Sideband selected by the upconversion mixer.
    double d_chirpMult{1.0};          ///< Multiplication factor applied after upconversion.

    //downconversion chain
    Sideband d_downMixSideband{UpperSideband}; ///< Sideband selected by the downconversion mixer.

    //chirp
    ChirpConfig d_chirpConfig; ///< Embedded chirp waveform configuration.

    /// \brief Validates and initializes the clock step list for acquisition.
    ///
    /// If no clock steps have been added (non-scan case) the template is copied
    /// to produce a single-step list. Resets \c d_currentClockIndex to 0 and
    /// \c d_completedSweeps to 0. Returns false if the chirp list is empty.
    bool prepareForAcquisition();

    /// \brief Replaces the active clock step with \a clocks.
    ///
    /// If clock steps have been populated, updates the entry at \c d_currentClockIndex;
    /// otherwise updates the template.
    void setCurrentClocks(const QHash<ClockType,ClockFreq> clocks);

    /// \brief Sets the desired frequency for clock type \a t in the active step.
    void setClockDesiredFreq(ClockType t, double f);

    /// \brief Sets the full \c ClockFreq record for type \a t in the clock template.
    ///
    /// If \a cf has an empty hardware key the entry is removed. When
    /// \c d_commonUpDownLO is true, setting \c UpLO also updates \c DownLO
    /// and vice versa.
    void setClockFreqInfo(ClockType t, const ClockFreq &cf);

    /// \brief Appends a complete clock configuration hash as one acquisition step.
    void addClockStep(QHash<ClockType,ClockFreq> h);

    /// \brief Appends an LO-scan step with the specified up- and down-LO frequencies.
    ///
    /// Copies the clock template, replaces the \c UpLO and \c DownLO desired
    /// frequencies with \a upLoMHz and \a downLoMHz, and appends the result.
    void addLoScanClockStep(double upLoMHz, double downLoMHz);

    /// \brief Appends a DR-scan step with the DR clock set to \a drFreqMHz.
    ///
    /// Copies the clock template, replaces the \c DRClock desired frequency, and
    /// appends the result.
    void addDrScanClockStep(double drFreqMHz);

    /// \brief Removes all clock steps and resets \c d_currentClockIndex to 0.
    void clearClockSteps();

    /// \brief Replaces the embedded chirp configuration.
    void setChirpConfig(const ChirpConfig &cc);

    /// \brief Advances to the next clock step; wraps to step 0 and increments the sweep count.
    ///
    /// Returns the new \c d_currentClockIndex.
    int advanceClockStep();

    /// \brief Returns the total shot count for a complete run (shots × steps × sweeps).
    quint64 totalShots() const;

    /// \brief Returns the number of shots accumulated in completed steps of the current sweep.
    quint64 completedSegmentShots() const;

    /// \brief Returns true when \a shots meets or exceeds the per-sweep shot target.
    bool canAdvance(qint64 shots) const;

    /// \brief Returns the number of clock steps; returns 1 for non-scan acquisitions.
    int numSegments() const;

    /// \brief Returns all clock steps; if the step list is empty, returns the template as a single-element vector.
    QVector<QHash<ClockType,RfConfig::ClockFreq>> clockSteps() const;

    /// \brief Returns the clock hash for the active step, or the template if no steps are defined.
    QHash<ClockType,ClockFreq> getClocks() const;

    /// \brief Returns the desired output frequency in MHz for clock type \a t in the active step.
    double clockFrequency(ClockType t) const;

    /// \brief Returns the hardware oscillator frequency in MHz for clock type \a t in the active step.
    ///
    /// Applies the inverse of the \c MultOperation to convert from the desired output
    /// frequency to the raw oscillator frequency.
    double rawClockFrequency(ClockType t) const;

    /// \brief Returns the [min, max] output frequency range in MHz for clock type \a t.
    ///
    /// Reads the hardware's \c minFreq and \c maxFreq settings and converts them to
    /// output frequencies via the clock's \c MultOperation and \c factor.
    std::pair<double,double> clockRange(ClockType t) const;

    /// \brief Returns the hardware key of the clock assigned to type \a t in the template.
    QString clockHardware(ClockType t) const;

    /// \brief Returns true when \c d_completedSweeps >= \c d_targetSweeps.
    bool isComplete() const;

    /// \brief Converts an AWG output frequency in MHz to the final chirp frequency at the sample.
    ///
    /// Applies \c d_awgMult, the upconversion sideband, the up-LO, and \c d_chirpMult.
    double calculateChirpFreq(double awgFreq) const;

    /// \brief Converts a chirp frequency in MHz to the required AWG output frequency.
    ///
    /// The inverse of \c calculateChirpFreq().
    double calculateAwgFreq(double chirpFreq) const;

    /// \brief Returns the absolute frequency offset in MHz between the chirp frequency and the down-LO.
    double calculateChirpAbsOffset(double awgFreq) const;

    /// \brief Returns the [min, max] chirp absolute-offset range in MHz across all non-identical chirps.
    QPair<double,double> calculateChirpAbsOffsetRange() const;

    /// \brief Writes the clock step CSV file for experiment \a num.
    ///
    /// Returns false if the file cannot be opened for writing.
    bool writeClockFile(int num) const;

    /// \brief Reads and populates clock steps from the clock CSV file for experiment \a num.
    void loadClockSteps(BlackchirpCSV *csv, int num, QString path);

    /// \brief Converts a \c ClockFreq's desired output frequency to the hardware oscillator frequency.
    static double getRawFrequency(const ClockFreq f);

    /// \brief Converts a hardware oscillator frequency to the output frequency given a \c ClockFreq record.
    static double rawToOutputFrequency(const ClockFreq &f, double rawFreq);

private:
    //clocks
    QHash<ClockType,ClockFreq> d_clockTemplate;
    QVector<QHash<ClockType,RfConfig::ClockFreq>> d_clockConfigs;
    int d_currentClockIndex{-1};


    // HeaderStorage interface
protected:
    /// \brief Writes RF chain scalar fields to the header storage tree.
    void storeValues() override;

    /// \brief Reads RF chain scalar fields back from the header storage tree.
    void retrieveValues() override;

    /// \brief Registers \c d_chirpConfig as a child \c HeaderStorage node.
    void prepareChildren() override;
};

Q_DECLARE_METATYPE(RfConfig)
Q_DECLARE_METATYPE(RfConfig::MultOperation)
Q_DECLARE_METATYPE(RfConfig::ClockType)
Q_DECLARE_METATYPE(RfConfig::ClockFreq)

#endif // RFCONFIG_H
