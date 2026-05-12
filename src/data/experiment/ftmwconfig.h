#ifndef FTMWCONFIG_H
#define FTMWCONFIG_H

#include <vector>

#include <QDateTime>
#include <QVariant>
#include <QMetaType>
#include <memory>

#include <data/storage/fidstoragebase.h>
#include <data/experiment/fid.h>
#include <data/experiment/rfconfig.h>
#include <data/experiment/hardware/core/ftmwdigitizerconfig.h>
#include <data/experiment/experimentobjective.h>

#ifdef BC_CUDA
#include <modules/cuda/gpuaverager.h>
#endif

/// \brief Storage keys used to persist FtmwConfig fields via HeaderStorage.
namespace BC::Store::FTMW {
inline constexpr QLatin1StringView key{"FtmwConfig"};           ///< Root group key.
inline constexpr QLatin1StringView duration{"Duration"};        ///< Target duration value.
inline constexpr QLatin1StringView enabled{"Enabled"};          ///< Whether FTMW is enabled.
inline constexpr QLatin1StringView phase{"PhaseCorrectionEnabled"};   ///< Phase-correction flag.
inline constexpr QLatin1StringView chirp{"ChirpScoringEnabled"};      ///< Chirp-scoring flag.
inline constexpr QLatin1StringView chirpThresh{"ChirpRMSThreshold"};  ///< Chirp RMS rejection threshold.
inline constexpr QLatin1StringView chirpOffset{"ChirpOffset"};        ///< Manual chirp offset in microseconds.
inline constexpr QLatin1StringView ftType{"Type"};              ///< FtmwType enumerator.
inline constexpr QLatin1StringView tShots{"TargetShots"};       ///< Target shot count.
inline constexpr QLatin1StringView objective{"Objective"};      ///< Serialized objective value.
}

/// \brief Config key identifying the FtmwType stored in an Experiment.
namespace BC::Config::Exp {
inline constexpr QLatin1StringView ftmwType{"FtmwType"};
}

struct WaveformEntry;
class BlackchirpCSV;
class WaveformBuffer;

/*!
 * \brief Abstract base class for FTMW experiment configurations.
 *
 * FtmwConfig brings together the digitizer configuration, RF/chirp
 * configuration, and FID storage for a single CP-FTMW acquisition.
 * Concrete subclasses (FtmwConfigSingle, FtmwConfigForever, etc.) implement
 * the completion and storage logic appropriate for each acquisition mode.
 *
 * The class also drives the chirp-quality pipeline: phase-correction
 * cross-correlates each incoming FID against the running average to find the
 * optimal sample shift, and chirp scoring rejects shots whose chirp RMS falls
 * below a configurable threshold.
 *
 * FtmwConfig holds a non-owning pointer to the experiment's WaveformBuffer.
 * The buffer is created and owned by the FtmwDigitizer hardware object; FtmwConfig
 * receives the pointer at acquisition setup time and must not free it.
 *
 * \sa RfConfig, FtmwDigitizerConfig, FidStorageBase
 */
class FtmwConfig : public ExperimentObjective, public HeaderStorage
{
    Q_GADGET
public:
    /*!
     * \brief Selects the acquisition termination criterion.
     */
    enum FtmwType
    {
        Target_Shots,   ///< Stop after accumulating a fixed number of shots.
        Target_Duration,///< Stop after a fixed wall-clock duration.
        Forever,        ///< Accumulate indefinitely until manually stopped.
        Peak_Up,        ///< Rolling-average peak-up mode; does not terminate automatically.
        LO_Scan,        ///< Step the local oscillator through a frequency range.
        DR_Scan         ///< Step a double-resonance drive frequency through a range.
    };
    Q_ENUM(FtmwType)

    /*!
     * \brief Construct with the hardware key of the FtmwDigitizer to use.
     * \param digitizerHwKey Hardware key string identifying the digitizer.
     */
    FtmwConfig(const QString& digitizerHwKey);
    FtmwConfig(const FtmwConfig &) =default;
    FtmwConfig &operator=(const FtmwConfig &) =default;
    virtual ~FtmwConfig();

    bool d_phaseCorrectionEnabled{false};  ///< Enable per-shot phase correction (cross-correlation shift).
    bool d_chirpScoringEnabled{false};     ///< Enable chirp quality scoring; shots below threshold are rejected.
    double d_chirpRMSThreshold{0.0};       ///< Minimum chirp RMS relative to running average (used when d_chirpScoringEnabled is true).
    double d_chirpOffsetUs{-1.0};          ///< Manual chirp start offset in microseconds; negative means auto-detect from marker.
    FtmwType d_type{Forever};              ///< Acquisition termination mode for this configuration.
    quint64 d_objective{0};                ///< Numeric objective (target shots or duration ticks, depending on d_type).

    /*!
     * \brief Return a mutable reference to the digitizer configuration.
     */
    FtmwDigitizerConfig &digitizerConfig() { return *ps_digitizerConfig; }
    /*!
     * \brief Return a const reference to the digitizer configuration.
     */
    const FtmwDigitizerConfig &digitizerConfig() const { return std::as_const(*ps_digitizerConfig); }

    RfConfig d_rfConfig; ///< RF chain and chirp configuration for this acquisition.

    /*!
     * \brief Initialize FID storage, the FID template, and RF configuration.
     * \return \c true on success; sets d_errorString and returns \c false on failure.
     */
    bool initialize() override;

    /*!
     * \brief Advance multi-segment acquisitions (LO/DR scans) and trigger autosave.
     * \return \c true if a segment boundary was crossed, \c false otherwise.
     */
    bool advance() override;

    /*!
     * \brief Update the FID template probe frequency when hardware is ready.
     *
     * Called after all hardware objects report readiness. Clears the
     * processing-paused flag so incoming FIDs are processed.
     */
    void hwReady() override;

    /*!
     * \brief Abort the acquisition; always returns \c false (no-op for FTMW).
     */
    bool abort() override;

    /*!
     * \brief Finalize FID storage and flush data to disk.
     */
    virtual void cleanupAndSave() override;

    /*!
     * \brief Return \c false; FTMW objectives are not indefinite by default.
     *
     * Subclasses that implement forever or duration-with-continue modes
     * override this to return \c true under the appropriate conditions.
     */
    virtual bool indefinite() const override { return false; }

    /*!
     * \brief Return the total number of shots accumulated so far.
     *
     * Must be implemented by each concrete subclass; the value drives
     * progress reporting and completion checks.
     */
    virtual quint64 completedShots() const =0;

    /*!
     * \brief Return the number of shots represented by one waveform transfer.
     *
     * For block-average modes this equals d_numAverages; otherwise 1.
     */
    quint64 shotIncrement() const;

    /*!
     * \brief Parse a raw waveform byte array into a FidList.
     * \param b Raw waveform bytes from the digitizer.
     * \return Parsed FidList ready for accumulation.
     */
    FidList parseWaveform(const QByteArray b) const;

    /*!
     * \brief Parse and combine a batch of WaveformEntry objects into a FidList.
     *
     * Uses parallel processing when the record is large enough to benefit.
     * \param entries Batch of raw-waveform entries from the WaveformBuffer drain cycle.
     * \return FidList with shot count equal to the sum of all entry shot counts.
     * \sa addBatchFids
     */
    FidList parseBatchFids(const std::vector<WaveformEntry> &entries) const;

    /*!
     * \brief Parse, optionally preprocess, and accumulate a batch of waveform entries.
     *
     * Calls parseBatchFids(), runs chirp scoring and phase correction if
     * enabled, then forwards the combined result to FID storage.
     * \param entries Batch of raw-waveform entries.
     * \return \c true if the FIDs were accepted and added; \c false on rejection or error.
     */
    bool addBatchFids(const std::vector<WaveformEntry> &entries);

    /*!
     * \brief Return the minimum frequency of the FTMW spectral window in MHz.
     */
    double ftMinMHz() const;

    /*!
     * \brief Return the maximum frequency of the FTMW spectral window in MHz.
     */
    double ftMaxMHz() const;

    /*!
     * \brief Return the Nyquist frequency of the digitizer in MHz (half the sample rate).
     */
    double ftNyquistMHz() const;

    /*!
     * \brief Return the FID record duration in microseconds.
     */
    double fidDurationUs() const;

    /*!
     * \brief Return the chirp figure of merit from the most recently processed shot.
     */
    double chirpFOM() const { return d_lastFom; }

    /*!
     * \brief Return the phase-correction sample shift applied to the most recent shot.
     */
    double chirpShift() const { return d_currentShift; }

    /*!
     * \brief Return the chirp RMS of the running average from the most recent preprocessing pass.
     */
    double chirpRMS() const { return d_lastRMS; }

    /*!
     * \brief Calculate the start sample index and sample count of the chirp window within the FID record.
     *
     * Uses the first chirp segment duration and the digitizer sample rate.
     * The chirp start is taken from d_chirpOffsetUs when non-negative; otherwise
     * it is derived from the Trigger marker time or the protection/gate lead time.
     *
     * \return QPair where the first element is the start sample index and the
     *         second is the number of samples spanning the chirp.
     */
    QPair<int,int> chirpRange() const;

    /*!
     * \brief Replace the accumulated FID data for all records.
     *
     * Constructs a FidList from \p newList using the current FID template
     * and forwards it to FID storage.
     * \param newList New raw data vectors, one per record channel.
     * \return \c true if storage accepted the data.
     */
    bool setFidsData(const QVector<QVector<qint64> > newList);

    /*!
     * \brief Parse, optionally preprocess, and accumulate a single raw waveform.
     * \param rawData Raw waveform bytes from the digitizer.
     * \return \c true if the FID was accepted.
     */
    bool addFids(const QByteArray rawData);

    /*!
     * \brief Accept and accumulate a pre-parsed array of qint64 FID values.
     *
     * Used when the WaveformBuffer producer has already accumulated raw bytes
     * into qint64 format (backpressure pre-accumulation path). Chirp scoring
     * and phase correction are still applied if enabled.
     * \param data   Raw bytes encoding a flat array of qint64 samples.
     * \param shotCount Number of shots represented by \p data.
     * \return \c true on success.
     */
    bool addPreAccumulatedFids(const QByteArray &data, quint64 shotCount);

    /*!
     * \brief Return the shared FID storage object for this configuration.
     */
    std::shared_ptr<FidStorageBase> storage() const;

    /*!
     * \brief Set the non-owning pointer to the experiment's WaveformBuffer.
     *
     * The buffer is created and owned by the FtmwDigitizer hardware object.
     * FtmwConfig stores the pointer so that AcquisitionManager can retrieve
     * it via waveformBuffer() without accessing the hardware object directly.
     * FtmwConfig does \b not take ownership and must not delete the pointer.
     * \param buf Pointer to the buffer, or \c nullptr to clear it.
     * \sa waveformBuffer
     */
    void setWaveformBuffer(WaveformBuffer *buf);

    /*!
     * \brief Return the non-owning pointer to the experiment's WaveformBuffer.
     *
     * May be \c nullptr before acquisition setup or after cleanup.
     * The pointed-to object is owned by the FtmwDigitizer hardware object.
     * \sa setWaveformBuffer
     */
    WaveformBuffer* waveformBuffer() const;

    /*!
     * \brief Reconstruct FID storage from disk for post-acquisition loading.
     *
     * Creates the appropriate FidStorageBase subclass using the experiment
     * number and path stored in d_number / d_path.
     */
    void loadFids();

private:
    WaveformBuffer *p_waveformBuffer{nullptr};
    std::shared_ptr<FtmwDigitizerConfig> ps_digitizerConfig;
    std::shared_ptr<FidStorageBase> p_fidStorage;
    Fid d_fidTemplate;
    QDateTime d_lastAutosaveTime;
    int d_currentShift{0};
    float d_lastFom{0.0};
    double d_lastRMS{0.0};

    bool preprocessChirp(const FidList l);
    float calculateFom(const QVector<qint64> vec, const Fid fid, QPair<int,int> range, int trialShift);
    double calculateChirpRMS(const QVector<qint64> chirp, quint64 shots = 1);

#ifdef BC_CUDA
    std::shared_ptr<GpuAverager> ps_gpu;
#endif

    // HeaderStorage interface
protected:
    /*!
     * \brief Serialize FTMW fields into HeaderStorage.
     */
    void storeValues() override;
    /*!
     * \brief Deserialize FTMW fields from HeaderStorage.
     */
    void retrieveValues() override;
    /*!
     * \brief Register child HeaderStorage objects (RfConfig, FtmwDigitizerConfig).
     */
    void prepareChildren() override;
    /*!
     * \brief Return the experiment-config key used to identify the FtmwType.
     */
    QString objectiveKey() const override;
    /*!
     * \brief Return the FtmwType enumerator as a QVariant for experiment config storage.
     */
    QVariant objectiveData() const override;

    /*!
     * \brief Return the bit-shift applied to waveform sample values.
     *
     * The default implementation returns 0. Subclasses may override this to
     * account for extra padding bits inserted by certain acquisition modes
     * (e.g. peak-up rolling average). The shift is used to scale the FID
     * voltage multiplier and to right-shift samples during waveform parsing.
     */
    virtual quint8 bitShift() const { return 0; }

public:
    /*!
     * \brief Public accessor for the protected bitShift() virtual.
     */
    quint8 getBitShift() const { return bitShift(); }

    /*!
     * \brief Subclass hook called at the end of initialize().
     * \return \c true on success; sets d_errorString and returns \c false on failure.
     */
    virtual bool _init() =0;

    /*!
     * \brief Subclass hook called during HeaderStorage serialization to store mode-specific fields.
     */
    virtual void _prepareToSave() =0;

    /*!
     * \brief Subclass hook called during HeaderStorage deserialization to restore mode-specific state.
     */
    virtual void _loadComplete() =0;

    /*!
     * \brief Factory method: create the FID storage object for this acquisition mode.
     * \param num Experiment number used to construct the storage path.
     * \param path Optional explicit base path; empty uses the default data directory.
     * \return Shared pointer to the newly created FidStorageBase subclass.
     */
    virtual std::shared_ptr<FidStorageBase> createStorage(int num, QString path="") =0;
};



#endif // FTMWCONFIG_H
