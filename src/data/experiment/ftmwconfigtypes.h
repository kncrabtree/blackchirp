#ifndef FTMWCONFIGTYPES_H
#define FTMWCONFIGTYPES_H

#include <data/experiment/ftmwconfig.h>
#include <QDateTime>

/*!
 * \brief FtmwConfig subclass for target-shots and target-duration modes.
 *
 * Implements the completion logic for acquisitions that stop after a
 * fixed number of shots (FtmwType::Target_Shots).  It is also the base
 * used when constructing a typed FtmwConfig from a deserialized FtmwConfig
 * value object.
 */
class FtmwConfigSingle : public FtmwConfig
{
public:
    /*!
     * \brief Construct with the hardware key of the FtmwDigitizer.
     * \param digitizerHwKey Hardware key string identifying the digitizer.
     */
    FtmwConfigSingle(const QString& digitizerHwKey);

    /*!
     * \brief Construct from a deserialized FtmwConfig value object.
     * \param other Source FtmwConfig.
     */
    FtmwConfigSingle(const FtmwConfig &other);
    ~FtmwConfigSingle() {}

    // ExperimentObjective interface
    /// \brief Return progress in per-mille (0–1000) toward the target shot count.
    int perMilComplete() const override;
    /// \brief Return \c true when the accumulated shot count reaches d_objective.
    bool isComplete() const override;

    // FtmwConfig interface
    /// \brief Return the total accumulated shot count from FID storage.
    quint64 completedShots() const override;
protected:
    /// \brief Initialize mode-specific state at acquisition start.
    bool _init() override;
    /// \brief Store mode-specific fields during HeaderStorage serialization.
    void _prepareToSave() override;
    /// \brief Restore mode-specific state during HeaderStorage deserialization.
    void _loadComplete() override;
    /// \brief Create the FID storage object for single-shot mode.
    std::shared_ptr<FidStorageBase> createStorage(int num, QString path="") override;
};


/*!
 * \brief FtmwConfig subclass for peak-up (rolling-average) mode.
 *
 * Maintains a rolling average of the most recent shots rather than an
 * accumulating sum.  Once the target shot count is reached, a non-zero
 * bitShift() supplies additional effective bits of precision for the
 * rolling average.
 */
class FtmwConfigPeakUp : public FtmwConfig
{
public:
    /*!
     * \brief Construct with the hardware key of the FtmwDigitizer.
     * \param digitizerHwKey Hardware key string identifying the digitizer.
     */
    FtmwConfigPeakUp(const QString& digitizerHwKey);

    /*!
     * \brief Construct from a deserialized FtmwConfig value object.
     * \param other Source FtmwConfig.
     */
    FtmwConfigPeakUp(const FtmwConfig &other);
    ~FtmwConfigPeakUp() {}

    // ExperimentObjective interface
    /// \brief Return progress in per-mille; peak-up never completes automatically.
    int perMilComplete() const override;
    /// \brief Always returns \c false; peak-up runs indefinitely.
    bool isComplete() const override;

    // FtmwConfig interface
    /// \brief Return the number of shots accumulated since last reset.
    quint64 completedShots() const override;
protected:
    /*!
     * \brief Return the bit-shift that supplies additional effective bits of precision
     *        for the rolling average once the target shot count is reached.
     */
    quint8 bitShift() const override;
    /// \brief Initialize mode-specific state at acquisition start.
    bool _init() override;
    /// \brief Store mode-specific fields during HeaderStorage serialization.
    void _prepareToSave() override;
    /// \brief Restore mode-specific state during HeaderStorage deserialization.
    void _loadComplete() override;
    /// \brief Create the peak-up FID storage object.
    std::shared_ptr<FidStorageBase> createStorage(int num, QString path="") override;
};


/*!
 * \brief FtmwConfig subclass for target-duration mode.
 *
 * Terminates acquisition after a wall-clock duration specified by d_objective.
 */
class FtmwConfigDuration : public FtmwConfig
{
public:
    /*!
     * \brief Construct with the hardware key of the FtmwDigitizer.
     * \param digitizerHwKey Hardware key string identifying the digitizer.
     */
    FtmwConfigDuration(const QString& digitizerHwKey);

    /*!
     * \brief Construct from a deserialized FtmwConfig value object.
     * \param other Source FtmwConfig.
     */
    FtmwConfigDuration(const FtmwConfig &other);
    ~FtmwConfigDuration() {}

    // ExperimentObjective interface
    /// \brief Return progress in per-mille toward the target duration.
    int perMilComplete() const override;
    /// \brief Return \c true when the wall-clock target time has been reached.
    bool isComplete() const override;

    // FtmwConfig interface
    /// \brief Return the total accumulated shot count from FID storage.
    quint64 completedShots() const override;
protected:
    /// \brief Initialize mode-specific state and record the start and target times.
    bool _init() override;
    /// \brief Store mode-specific fields during HeaderStorage serialization.
    void _prepareToSave() override;
    /// \brief Restore mode-specific state during HeaderStorage deserialization.
    void _loadComplete() override;
    /// \brief Create the FID storage object for duration mode.
    std::shared_ptr<FidStorageBase> createStorage(int num, QString path="") override;

private:
    QDateTime d_startTime, d_targetTime; ///< Wall-clock start and target end times.
};


/*!
 * \brief FtmwConfig subclass for indefinite (forever) averaging mode.
 *
 * Acquisition continues until manually stopped.  indefinite() returns
 * \c true once the shot counter first reaches the completion threshold,
 * preventing normal completion logic from triggering.
 */
class FtmwConfigForever : public FtmwConfig
{
public:
    /*!
     * \brief Construct with the hardware key of the FtmwDigitizer.
     * \param digitizerHwKey Hardware key string identifying the digitizer.
     */
    FtmwConfigForever(const QString& digitizerHwKey);

    /*!
     * \brief Construct from a deserialized FtmwConfig value object.
     * \param other Source FtmwConfig.
     */
    FtmwConfigForever(const FtmwConfig &other);
    ~FtmwConfigForever() {}

    // ExperimentObjective interface
    /// \brief Return progress in per-mille; always returns 0 for forever mode.
    int perMilComplete() const override;
    /// \brief Return \c true to prevent the normal completion check from firing.
    bool indefinite() const override;
    /// \brief Always returns \c false; the user must stop acquisition manually.
    bool isComplete() const override;

    // FtmwConfig interface
    /// \brief Return the total accumulated shot count from FID storage.
    quint64 completedShots() const override;
protected:
    /// \brief Initialize mode-specific state at acquisition start.
    bool _init() override;
    /// \brief Store mode-specific fields during HeaderStorage serialization.
    void _prepareToSave() override;
    /// \brief Restore mode-specific state during HeaderStorage deserialization.
    void _loadComplete() override;
    /// \brief Create the FID storage object for forever mode.
    std::shared_ptr<FidStorageBase> createStorage(int num, QString path="") override;
};

/// \brief Storage keys for FtmwConfigLOScan scan-range parameters.
namespace BC::Store::FtmwLO {
inline constexpr QLatin1StringView upStart{"UpLOBegin"};   ///< Upward sweep LO start frequency (MHz).
inline constexpr QLatin1StringView upEnd{"UpLOEnd"};       ///< Upward sweep LO end frequency (MHz).
inline constexpr QLatin1StringView upMin{"UpMinorSteps"};  ///< Number of minor LO steps per major step on the upward sweep.
inline constexpr QLatin1StringView upMaj{"UpMajorSteps"};  ///< Number of major LO steps on the upward sweep.
inline constexpr QLatin1StringView downStart{"DownLOBegin"};///< Downward sweep LO start frequency (MHz).
inline constexpr QLatin1StringView downEnd{"DownLOEnd"};   ///< Downward sweep LO end frequency (MHz).
inline constexpr QLatin1StringView downMin{"DownMinorSteps"};///< Number of minor LO steps per major step on the downward sweep.
inline constexpr QLatin1StringView downMaj{"DownMajorSteps"};///< Number of major LO steps on the downward sweep.
}

/*!
 * \brief FtmwConfig subclass for LO-scan (frequency-sweep) mode.
 *
 * Steps the local oscillator through an upward and optional downward
 * frequency sweep, accumulating a fixed number of shots at each step.
 * Each pair of (major, minor) step counts defines the scan grid; the
 * minor step size is derived from the LO start/end and major step count.
 */
class FtmwConfigLOScan : public FtmwConfig
{
public:
    /*!
     * \brief Construct with the hardware key of the FtmwDigitizer.
     * \param digitizerHwKey Hardware key string identifying the digitizer.
     */
    FtmwConfigLOScan(const QString& digitizerHwKey);

    /*!
     * \brief Construct from a deserialized FtmwConfig value object.
     * \param other Source FtmwConfig.
     */
    FtmwConfigLOScan(const FtmwConfig &other);
    ~FtmwConfigLOScan() {}

    double d_upStart{0.0};   ///< Upward sweep LO start frequency in MHz.
    double d_upEnd{0.0};     ///< Upward sweep LO end frequency in MHz.
    double d_downStart{0.0}; ///< Downward sweep LO start frequency in MHz.
    double d_downEnd{0.0};   ///< Downward sweep LO end frequency in MHz.
    int d_upMaj{0};          ///< Number of major steps in the upward sweep.
    int d_upMin{0};          ///< Number of minor steps per major step in the upward sweep.
    int d_downMaj{0};        ///< Number of major steps in the downward sweep.
    int d_downMin{0};        ///< Number of minor steps per major step in the downward sweep.

    // ExperimentObjective interface
    /// \brief Return progress in per-mille toward completing all LO scan steps.
    int perMilComplete() const override;
    /// \brief Return \c true when all scan steps and their shot targets are complete.
    bool isComplete() const override;

    // FtmwConfig interface
    /// \brief Return the total accumulated shot count across all segments.
    quint64 completedShots() const override;
protected:
    /// \brief Initialize mode-specific state and configure LO scan segments in RfConfig.
    bool _init() override;
    /// \brief Store LO scan parameters during HeaderStorage serialization.
    void _prepareToSave() override;
    /// \brief Restore LO scan parameters during HeaderStorage deserialization.
    void _loadComplete() override;
    /// \brief Create the FID storage object for LO scan mode.
    std::shared_ptr<FidStorageBase> createStorage(int num, QString path) override;
};

/// \brief Storage keys for FtmwConfigDRScan scan-range parameters.
namespace BC::Store::FtmwDR {
inline constexpr QLatin1StringView drStart{"DRBegin"};        ///< Double-resonance drive start frequency (MHz).
inline constexpr QLatin1StringView drStep{"DRStep"};          ///< Step size between DR drive frequencies (MHz).
inline constexpr QLatin1StringView drEnd{"DREnd"};            ///< Double-resonance drive end frequency (MHz).
inline constexpr QLatin1StringView drNumSteps{"DRNumSteps"};  ///< Total number of DR frequency steps.
}

/*!
 * \brief FtmwConfig subclass for double-resonance (DR) scan mode.
 *
 * Steps a second microwave source (the DR drive) through an evenly spaced
 * frequency range while collecting a fixed number of shots per step.
 */
class FtmwConfigDRScan : public FtmwConfig
{
public:
    /*!
     * \brief Construct with the hardware key of the FtmwDigitizer.
     * \param digitizerHwKey Hardware key string identifying the digitizer.
     */
    FtmwConfigDRScan(const QString& digitizerHwKey);

    /*!
     * \brief Construct from a deserialized FtmwConfig value object.
     * \param other Source FtmwConfig.
     */
    FtmwConfigDRScan(const FtmwConfig &other);
    ~FtmwConfigDRScan() {};

    double d_start{0.0};  ///< DR drive start frequency in MHz.
    double d_step{1.0};   ///< DR drive step size in MHz.
    int d_numSteps{2};    ///< Total number of DR frequency steps.

    // ExperimentObjective interface
    /// \brief Return progress in per-mille toward completing all DR scan steps.
    int perMilComplete() const override;
    /// \brief Return \c true when all DR scan steps and their shot targets are complete.
    bool isComplete() const override;

    // FtmwConfig interface
    /// \brief Return the total accumulated shot count across all DR scan segments.
    quint64 completedShots() const override;

protected:
    /// \brief Initialize mode-specific state and configure DR scan segments in RfConfig.
    bool _init() override;
    /// \brief Store DR scan parameters during HeaderStorage serialization.
    void _prepareToSave() override;
    /// \brief Restore DR scan parameters during HeaderStorage deserialization.
    void _loadComplete() override;
    /// \brief Create the FID storage object for DR scan mode.
    std::shared_ptr<FidStorageBase> createStorage(int num, QString path) override;
};

#endif // FTMWCONFIGTYPES_H
