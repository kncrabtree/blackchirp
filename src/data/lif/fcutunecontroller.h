#ifndef FCUTUNECONTROLLER_H
#define FCUTUNECONTROLLER_H

#include <memory>
#include <optional>

#include <QObject>
#include <QVector>

#include <data/lif/fcutunesweep.h>
#include <data/lif/lifdigitizerconfig.h>
#include <data/lif/liftrace.h>

Q_DECLARE_METATYPE(BC::FcuTune::Result)

/*!
 * \brief Drives one FcuTuneSweep against live hardware through signals.
 *
 * Hardware-agnostic: it requests trims with requestTrim() and learns the
 * outcome through trimUpdated(), and it consumes raw LIF digitizer
 * waveforms through processWaveform(). The owner connects these to
 * HardwareManager::setLifStageTrim(), HardwareManager::lifStageTrimUpdate
 * and HardwareManager::lifDigitizerShotAcquired (directly or via a
 * forwarding widget).
 *
 * Sequence: start() requests the first sweep trim; once the stage confirms
 * it, waveforms are integrated over the reference gate and fed to the
 * sweep until the point is complete, then the next trim is requested.
 * After the last point the sweep is fit. If the maximum lies at the edge
 * of the window, a new sweep centered on that edge point is run, up to
 * \c Settings::maxRecenters times. A successful fit's center
 * (rounded to a whole native unit) is requested as the final trim;
 * otherwise, or on abort() or a failed move, the trim held at start() is
 * restored. finished() is emitted once that final move is confirmed.
 * A reference waveform counts as clipped when any sample within the
 * reference gate reaches the digitizer's full scale or, when
 * \c Settings::saturationVolts is positive, that level in either
 * polarity (a photodiode can saturate well below the digitizer range).
 * Waveforms arriving while a move is pending are ignored, as is a
 * successful trim update that does not carry the pending trim (one
 * triggered by some other requester of the same stage).
 */
class FcuTuneController : public QObject
{
    Q_OBJECT
public:
    explicit FcuTuneController(QObject *parent = nullptr);
    ~FcuTuneController() override;

    bool isRunning() const { return d_state != State::Idle; }
    const QString &stageKey() const { return d_stageKey; }

    /*!
     * \brief Begin a sweep of \a stageKey centered on its current trim
     *        \a startTrim.
     *
     * \a digiConfig describes the waveforms that processWaveform() will
     * receive, and \a procSettings supplies the reference gate.
     *
     * \return \c false (and emits nothing) if a sweep is already running or
     * the reference channel is not enabled in \a digiConfig.
     */
    bool start(const QString &stageKey, double startTrim, const BC::FcuTune::Settings &settings,
               const LifDigitizerConfig &digiConfig, const LifTrace::LifProcSettings &procSettings);

    //! Stop a running sweep and restore the starting trim.
    void abort();

public slots:
    void processWaveform(const QVector<qint8> b);
    void trimUpdated(const QString &stageKey, double trim, int direction, bool success);

signals:
    void requestTrim(QString stageKey, double trim);
    void pointComplete(double trim, double mean, double stdErr);
    void progress(int perMil);
    //! The maximum fell at the window edge; a new sweep centered on \a newCenter is starting.
    void recentered(double newCenter, int attempt);
    /*!
     * \brief The sweep is over and the final trim move has completed.
     * \param result Fit outcome (status Aborted or MoveFailed when the sweep did not finish).
     * \param finalTrim Trim the stage holds afterwards.
     * \param moveOk Whether the final move succeeded.
     */
    void finished(BC::FcuTune::Result result, double finalTrim, bool moveOk);

private:
    enum class State { Idle, Moving, Acquiring, Finishing };

    void finish(BC::FcuTune::Result result, double finalTrim);
    void request(double trim);

    State d_state{State::Idle};
    QString d_stageKey;
    double d_startTrim{0.0};
    double d_pendingTrim{0.0};
    int d_recenters{0};
    std::unique_ptr<FcuTuneSweep> pu_sweep;
    std::unique_ptr<LifDigitizerConfig> pu_digiConfig;
    LifTrace::LifProcSettings d_procSettings;
    BC::FcuTune::Result d_result;
};

#endif // FCUTUNECONTROLLER_H
