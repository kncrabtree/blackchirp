#ifndef LIFCONFIG_H
#define LIFCONFIG_H

#include <QVector>
#include <QPointF>
#include <QMap>
#include <QVariant>
#include <memory>

#include <data/storage/headerstorage.h>
#include <data/experiment/experimentobjective.h>
#include <data/lif/lifstorage.h>
#include <data/lif/lifdigitizerconfig.h>
#include <data/lif/lifunits.h>
#include <data/lif/lifconversion.h>

#include <vector>

/// \brief Storage keys used to persist LifConfig fields via HeaderStorage.
namespace BC::Store::LIF {
inline constexpr QLatin1StringView key("LifConfig");          ///< Root group key.
inline constexpr QLatin1StringView order("ScanOrder");        ///< LifScanOrder enumerator.
inline constexpr QLatin1StringView completeMode("CompleteMode"); ///< LifCompleteMode enumerator.
inline constexpr QLatin1StringView dStart("DelayStart");      ///< Delay scan start time in microseconds.
inline constexpr QLatin1StringView dStep("DelayStep");        ///< Delay scan step size in microseconds.
inline constexpr QLatin1StringView dPoints("DelayPoints");    ///< Number of delay scan points.
inline constexpr QLatin1StringView dRandom("DelayRandom");    ///< Whether to randomize delay point order each sweep.
inline constexpr QLatin1StringView lStart("LaserStart");      ///< Laser scan start position.
inline constexpr QLatin1StringView lStep("LaserStep");        ///< Laser scan step size.
inline constexpr QLatin1StringView lPoints("LaserPoints");    ///< Number of laser scan points.
inline constexpr QLatin1StringView shotsPerPoint("ShotsPerPoint"); ///< Target shots per scan point.
}

/// \brief Config key identifying the LIF type stored in an Experiment.
namespace BC::Config::Exp {
inline constexpr QLatin1StringView lifType{"LifType"};
}

/*!
 * \brief Configuration and runtime state for a LIF (Laser-Induced Fluorescence) acquisition.
 *
 * LifConfig defines a two-dimensional scan over delay time and laser position,
 * accumulating a user-specified number of shots at each grid point.  The scan
 * can traverse points in delay-first or laser-first order, and the delay axis
 * can optionally be randomized each sweep to reduce systematic errors.
 *
 * After completion LifConfig can continue averaging (ContinueAveraging mode)
 * or stop immediately (StopWhenComplete mode).  The class owns a LifStorage
 * instance that persists raw LIF traces and processing-gate settings to disk.
 *
 * \sa LifDigitizerConfig, LifStorage, LifTrace
 */
class LifConfig : public ExperimentObjective, public HeaderStorage
{
    Q_GADGET
public:
    /*!
     * \brief Controls the order in which the 2-D scan grid is traversed.
     */
    enum LifScanOrder {
        DelayFirst, ///< Scan all delay points before advancing the laser position.
        LaserFirst  ///< Scan all laser positions before advancing the delay.
    };
    Q_ENUM(LifScanOrder)

    /*!
     * \brief Controls behavior when the scan grid has been fully covered once.
     */
    enum LifCompleteMode {
        StopWhenComplete,   ///< Stop acquisition after the first complete sweep.
        ContinueAveraging   ///< Continue accumulating additional sweeps indefinitely.
    };
    Q_ENUM(LifCompleteMode)

    /*!
     * \brief Construct with the hardware key of the LIF digitizer.
     * \param digitizerHwKey Hardware key string identifying the LIF digitizer.
     */
    LifConfig(const QString& digitizerHwKey);
    ~LifConfig() = default;

    bool d_complete{false};                         ///< Set to \c true once the first full sweep is complete.
    LifScanOrder d_order{DelayFirst};               ///< Scan traversal order.
    LifCompleteMode d_completeMode{ContinueAveraging}; ///< Behavior on completion.
    bool d_disableFlashlamp{true};                  ///< Disable the laser flashlamp between scan points when \c true.

    double d_delayStartUs{-1.0};    ///< Delay scan start time in microseconds.
    double d_delayStepUs{0.0};      ///< Delay scan step size in microseconds.
    int d_delayPoints{0};           ///< Number of delay scan points.
    bool d_delayRandom{false};      ///< Randomize delay point order each sweep when \c true.
    QVector<int> d_delayIndices;    ///< Permuted index array for randomized delay scanning.
    int d_delayScanIndex{0};        ///< Current position within d_delayIndices.

    double d_laserPosStart{-1.0};   ///< Laser scan start position, in the display LaserUnit.
    double d_laserPosStep{0.0};     ///< Laser scan step size, in the display LaserUnit.
    int d_laserPosPoints{0};        ///< Number of laser scan points.

    LifTrace::LifProcSettings d_procSettings; ///< Gate positions and processing parameters for LIF traces.

    int d_shotsPerPoint{0}; ///< Target number of shots to accumulate at each scan point.

    /*!
     * \brief Return a mutable reference to the LIF digitizer configuration.
     */
    LifDigitizerConfig &digitizerConfig() { return *ps_digitizerConfig; }

    /*!
     * \brief Return a const reference to the LIF digitizer configuration.
     */
    const LifDigitizerConfig &digitizerConfig() const { return std::as_const(*ps_digitizerConfig); }

    /*!
     * \brief Return \c true when d_complete is set and (if ContinueAveraging) once at 1000 per-mille.
     */
    bool isComplete() const override;

    /*!
     * \brief Return the current delay time in microseconds.
     */
    double currentDelay() const;

    /*!
     * \brief Return the current laser position as an output-beam
     *        wavenumber (cm⁻¹), converted from the display-unit scan
     *        grid at this dispatch boundary.
     */
    double currentLaserPos() const;

    /*!
     * \brief Return the (start, end) delay range in microseconds.
     */
    QPair<double,double> delayRange() const;

    /*!
     * \brief Return the (start, end) laser position range, in the
     *        display LaserUnit.
     */
    QPair<double,double> laserRange() const;

    /*!
     * \brief Return the total number of shots across the full scan grid.
     */
    int targetShots() const;

    /*!
     * \brief Return the number of shots accumulated so far.
     */
    int completedShots() const;

    /*!
     * \brief Return the (start, end) sample indices of the LIF signal gate.
     */
    QPair<int,int> lifGate() const;

    /*!
     * \brief Return the (start, end) sample indices of the reference gate.
     */
    QPair<int,int> refGate() const;

    /*!
     * \brief Return the shared LifStorage object managing on-disk data.
     */
    std::shared_ptr<LifStorage> storage() { return ps_storage; }

    /*!
     * \brief Accept a raw LIF waveform and add it to storage at the current scan point.
     *
     * Constructs a LifTrace and forwards it to LifStorage. If d_completeMode
     * is StopWhenComplete and d_complete is \c true, the waveform is discarded.
     * \param d Raw waveform bytes from the LIF digitizer.
     */
    void addWaveform(const QVector<qint8> d);

    /*!
     * \brief Reconstruct LifStorage from disk for post-acquisition loading.
     */
    void loadLifData();

    /*!
     * \brief Set the display unit used for the laser position axis.
     * \param units Display unit (BC::LifConv::LaserUnit).
     */
    void setLaserUnits(BC::LifConv::LaserUnit units);

    /*!
     * \brief Set the decimal-precision hint used when serializing the laser position axis.
     *
     * Controls how LaserStart/LaserStep are formatted in header.csv so the
     * column-width of fractional digits is preserved on disk. Callers
     * normally seed this from the LIF laser hardware's display-decimals
     * setting at acquisition time; on load it is inferred from the
     * on-disk formatting of LaserStart/LaserStep.
     */
    void setLaserDecimals(int decimals);

    /*!
     * \brief Return the laser position display unit.
     *
     * Populated from the column-6 unit cell of the LaserStart header
     * row on load, or from the laser hardware setting at acquisition.
     * The laser scan axis (d_laserPosStart/Step) is uniform in this
     * unit; see currentLaserPos() for the display->output-cm⁻¹
     * conversion at the hardware-dispatch boundary.
     */
    BC::LifConv::LaserUnit laserUnits() const { return d_laserUnits; }

    /*!
     * \brief Return the laser-position display precision in fractional digits.
     *
     * Inferred at load time from the on-disk strings of LaserStart and
     * LaserStep (max fractional digits across both), and seeded from
     * the laser hardware setting at acquisition time. Used by display
     * widgets for axis-label formatting; not persisted as its own row.
     */
    int laserDecimals() const { return d_laserDecimals; }

    /*!
     * \brief Record the per-experiment conversion-topology node list and
     *        rebuild the cached assembled conversion from it.
     *
     * \a nodes is the authoritative, already-joined node-descriptor list
     * (op/n from each stage's hardware, inputs/isFinal from the
     * per-experiment wiring); empty for the identity / bare-laser case,
     * where the output beam is the grating fundamental and no topology
     * file is written. \a laserKey is the active LifLaser's hwKey, used to
     * serialize a tunable-source (RefType::Laser) input by its real hwKey
     * rather than a sentinel.
     *
     * Rebuilds the cached conversion() via LifConversion::assemble(); on
     * assembly failure the cache falls back to the identity conversion
     * (mirroring the tolerant fallback used elsewhere when a topology
     * cannot yet be assembled, e.g. mid-edit).
     */
    void setConversionNodes(std::vector<BC::LifConv::Node> nodes, const QString &laserKey);

    /*!
     * \brief Return the current conversion-topology node list (empty =
     *        identity/no stages).
     */
    const std::vector<BC::LifConv::Node> &conversionNodes() const { return d_conversionNodes; }

    /*!
     * \brief Return the conversion assembled from conversionNodes() by the
     *        most recent setConversionNodes() call (identity if never set,
     *        or if assembly failed).
     */
    const LifConversion &conversion() const { return d_conversion; }

    /*!
     * \brief Return \c true when conversionNodes() is non-empty.
     */
    bool hasConversion() const { return !d_conversionNodes.empty(); }

    /*!
     * \brief Deprecated shim for setConversionNodes(): drops \a conv (the
     *        cached conversion is now rebuilt internally via assemble())
     *        and forwards \a nodes/\a laserKey.
     *
     * \deprecated Retained only so the existing HardwareManager prep call
     * site continues to link; callers should switch to setConversionNodes()
     * directly.
     */
    void setConversionTopology(const std::vector<BC::LifConv::Node> &nodes,
                               const LifConversion &conv,
                               const QString &laserKey);

    /*!
     * \brief Write liftopology.csv — one row per conversion node — into the
     *        experiment directory.
     *
     * Records the raw DAG (op, harmonic order, input wiring, FINAL marker)
     * alongside each node's resolved output-beam affine mapping
     * (\c output = A·fundamental + B, cm⁻¹). The FINAL row's coefficients are
     * the output-axis ↔ fundamental relation. Does nothing and returns
     * \c true for the identity case (no conversion stages): header.csv
     * already carries the full display-unit axis. Returns \c false only on a
     * file-write failure.
     */
    bool writeTopologyFile() const;

    /*!
     * \brief Read liftopology.csv, if present, and reconstruct the
     *        conversion-topology node list via setConversionNodes().
     *
     * Mirrors RfConfig::loadClockSteps(): resolves the file from this
     * config's own d_number/d_path. A missing file means the identity case
     * (writeTopologyFile() skips writing one) and is not an error. Input
     * tokens are classified as \c Fixed:<cm1> -> Fixed, a token matching
     * another row's StageKey -> Stage, and anything else -> Laser (that
     * token is the laser hwKey, captured as the config's conversion laser
     * key). OutCoeffA/B are derived data recomputed via assembly, not read.
     * Returns \c false only when the file exists but cannot be opened.
     */
    bool readTopologyFile();

private:
    std::shared_ptr<LifStorage> ps_storage;
    std::shared_ptr<LifDigitizerConfig> ps_digitizerConfig;
    BC::LifConv::LaserUnit d_laserUnits{BC::LifConv::LaserUnit::Nm};
    int d_laserDecimals{2};
    std::vector<BC::LifConv::Node> d_conversionNodes; ///< Conversion-topology node descriptors (empty = identity/no stages).
    LifConversion d_conversion;                       ///< Assembled conversion, for resolved per-node output coefficients.
    QString d_conversionLaserKey;                     ///< Active LifLaser hwKey, for serializing tunable-source inputs.
    int d_currentDelayIndex{0};
    int d_currentLaserIndex{0};
    int d_completedSweeps{0};

    // HeaderStorage interface
protected:
    /*!
     * \brief Serialize LifConfig fields into HeaderStorage.
     */
    void storeValues() override;
    /*!
     * \brief Deserialize LifConfig fields from HeaderStorage.
     */
    void retrieveValues() override;

public:
    /*!
     * \brief Register child HeaderStorage objects (LifDigitizerConfig).
     */
    void prepareChildren() override;

    // ExperimentObjective interface
public:
    /*!
     * \brief Initialize storage, build the delay index permutation, and start acquisition.
     * \return Always returns \c true for LifConfig.
     */
    bool initialize() override;

    /*!
     * \brief Advance the scan to the next point; return \c true if the point was incremented.
     *
     * Checks whether the current point has reached its shot target, shuffles
     * the delay order if randomization is enabled and a full delay sweep just
     * completed, and calls LifStorage::advance().
     */
    bool advance() override;

    /*!
     * \brief Clear the processing-paused flag when hardware reports readiness.
     */
    void hwReady() override;

    /*!
     * \brief Return progress in per-mille (0–1000) across all scan points.
     */
    int perMilComplete() const override;

    /*!
     * \brief Return \c true when ContinueAveraging mode is active and a full sweep is done.
     */
    bool indefinite() const override;

    /*!
     * \brief Abort the LIF acquisition; always returns \c false (no-op).
     */
    bool abort() override;

    /*!
     * \brief Return the experiment-config key used to identify the LIF type.
     */
    QString objectiveKey() const override;

    /*!
     * \brief Finalize LifStorage and flush data to disk.
     */
    void cleanupAndSave() override;
};


#endif // LIFCONFIG_H
