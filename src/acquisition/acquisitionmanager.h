#ifndef ACQUISITIONMANAGER_H
#define ACQUISITIONMANAGER_H

#include <atomic>

#include <QObject>
#include <QDateTime>
#include <QTime>
#include <QTimer>
#include <QThread>
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>

#include <data/loghandler.h>
#include <data/experiment/experiment.h>

/// \brief Result of a worker-thread waveform-processing batch.
///
/// Returned by the QtConcurrent worker to the AcquisitionManager event loop
/// so that side-effect operations (advance, signals, abort) remain on the AM
/// thread. The struct is the sole cross-thread data contract between the
/// waveform-processing worker and the manager.
struct FtmwProcessingResult {
    int entriesProcessed{0};  ///< Number of waveform entries processed in this batch.
    bool success{true};       ///< \c false if a fatal processing error occurred.
    QString errorString;      ///< Non-empty when \c success is \c false; describes the error.
    QString warningString;    ///< Non-empty when a non-fatal anomaly was detected.
};

/// \brief Drives a single Experiment from hardware initialization through
/// acquisition-loop completion.
///
/// AcquisitionManager owns the in-progress Experiment shared pointer,
/// accumulates FID data and auxiliary sensor readings into the per-experiment
/// storage tree, and emits the signals that notify the main window,
/// HardwareManager, and BatchManager of lifecycle events.
///
/// The manager lives on a dedicated QThread ("AcquisitionManagerThread")
/// created in MainWindow. All public slots and signal emissions execute on
/// that thread. Callers on other threads must use QMetaObject::invokeMethod
/// or queued signal/slot connections — direct calls from the GUI thread are
/// not safe.
///
/// Waveform processing is dispatched to the Qt thread pool via
/// QtConcurrent::run; the result is returned to the AM thread through a
/// QFutureWatcher so that all state mutations remain thread-confined.
///
/// \sa BatchManager, HardwareManager, FtmwProcessingResult
class AcquisitionManager : public QObject
{
    Q_OBJECT
public:
    /// \brief Constructs an AcquisitionManager with the given parent.
    explicit AcquisitionManager(QObject *parent = nullptr);

    /// \brief Destroys the manager, waiting for any in-flight waveform worker to finish.
    ~AcquisitionManager();

    /// \brief Describes the current phase of the acquisition loop.
    enum AcquisitionState
    {
        Idle,      ///< No experiment is running; the manager accepts a new beginExperiment call.
        Acquiring, ///< An experiment is active and data are being collected.
        Paused     ///< Acquisition is temporarily suspended; no new data are processed.
    };

signals:
    /// \brief Requests that a message be appended to the application log.
    /// \param msg The text to log.
    /// \param code Severity classification; defaults to LogHandler::Normal.
    void logMessage(QString msg, LogHandler::MessageCode code = LogHandler::Normal);

    /// \brief Requests that a transient status string be displayed in the status bar.
    /// \param msg The status text.
    /// \param timeout Display duration in milliseconds; 0 means indefinite.
    void statusMessage(QString msg, int timeout = 0);

    /// \brief Emitted when the acquisition loop for the current experiment has ended.
    ///
    /// Emitted after endAcquisition(), after the experiment is saved to disk, and before
    /// the experiment shared pointer is released. BatchManager::experimentComplete() is
    /// connected to this signal so the batch can decide whether to advance.
    void experimentComplete();

    /// \brief Reports FTMW acquisition progress to the progress bar.
    /// \param perMil Progress in units of 1/1000 of completion (0–1000).
    void ftmwUpdateProgress(int perMil);

    /// \brief Notifies HardwareManager that clock frequencies should be applied.
    ///
    /// Emitted at experiment start and whenever an FTMW segment boundary is crossed.
    /// \param clocks Map from clock type to target frequency.
    void newClockSettings(QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks);

    /// \brief Notifies HardwareManager that data collection should begin.
    ///
    /// Connected to HardwareManager::beginAcquisition. Emitted after clock
    /// settings have been dispatched at the start of beginExperiment().
    void beginAcquisition();

    /// \brief Notifies HardwareManager that data collection has ended.
    ///
    /// Emitted inside finishAcquisition() before the state transitions to Idle.
    /// HardwareManager uses this to stop hardware triggers.
    void endAcquisition();

    /// \brief Requests that HardwareManager read and return aux sensor data.
    ///
    /// Emitted on each aux-data timer tick. HardwareManager responds by
    /// emitting its auxData signal, which is connected to processAuxData().
    void auxDataSignal();

    /// \brief Delivers a completed aux-data point to observers such as the plot widget.
    /// \param data Map of sensor key to measured value.
    /// \param timestamp Acquisition time of the point.
    void auxData(AuxDataStorage::AuxDataMap data, QDateTime timestamp);

    /// \brief Emitted when a concurrent backup operation launched via QtConcurrent has finished.
    ///
    /// Connected to the FtmwViewWidget so that the view can refresh its backup list.
    void backupComplete();

    /// \brief Notifies the LIF display that the current LIF point has new data.
    void lifPointUpdate();

    /// \brief Requests that HardwareManager configure LIF hardware for the next scan point.
    /// \param delay Delay value in seconds for the next LIF point.
    /// \param frequency Laser frequency in wavenumbers for the next LIF point.
    void nextLifPoint(double delay, double frequency);

    /// \brief Reports LIF acquisition progress to the LIF progress bar.
    /// \param perMil Progress in units of 1/1000 of completion (0–1000).
    void lifShotAcquired(int perMil);

public slots:
    /// \brief Begins acquiring data for the given experiment.
    ///
    /// Called by MainWindow::experimentInitialized() via QMetaObject::invokeMethod
    /// after hardware initialization succeeds. Sets the state to Acquiring,
    /// emits beginAcquisition(), starts the aux-data timer if configured, and
    /// starts the FTMW drain timer for buffered waveform modes.
    ///
    /// \param exp Shared pointer to the fully initialized experiment. The manager
    ///            holds this pointer for the duration of the acquisition loop.
    void beginExperiment(std::shared_ptr<Experiment> exp);

    /// \brief Incorporates a map of aux-sensor readings into the current experiment.
    ///
    /// Stores the readings in the experiment's AuxDataStorage and forwards the
    /// data map via the auxData() signal. If storage fails, abort() is called.
    /// No-op when the state is not Acquiring.
    ///
    /// \param m Map of sensor key to measured value returned by HardwareManager.
    void processAuxData(AuxDataStorage::AuxDataMap m);

    /// \brief Validates a map of sensor readings against the experiment's limit set.
    ///
    /// Calls Experiment::validateItem() for each entry. If any value exceeds its
    /// configured limit, abort() is called. No-op when the state is not Acquiring.
    ///
    /// \param m Map of sensor key to measured value returned by HardwareManager.
    void processValidationData(AuxDataStorage::AuxDataMap m);

    /// \brief Records the actual clock frequencies applied by hardware.
    ///
    /// Called when HardwareManager emits allClocksReady. Stores the confirmed
    /// frequencies in the FTMW configuration and marks hardware as ready so
    /// waveform acquisition can proceed.
    ///
    /// \param clocks Map from clock type to the frequency actually programmed by hardware.
    void clockSettingsComplete(const QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks);

    /// \brief Suspends data processing without stopping hardware triggers.
    ///
    /// Transitions from Acquiring to Paused. Incoming waveform data and aux readings
    /// are dropped until resume() is called. No-op if the state is not Acquiring.
    void pause();

    /// \brief Resumes data processing after a pause.
    ///
    /// Transitions from Paused back to Acquiring. No-op if the state is not Paused.
    void resume();

    /// \brief Aborts the current experiment and ends the acquisition loop.
    ///
    /// Marks the experiment as aborted, then calls finishAcquisition().
    /// Safe to call from Acquiring or Paused. No-op from Idle.
    void abort();

    /// \brief Accumulates a raw LIF waveform shot into the current LIF scan point.
    ///
    /// Adds the waveform to the LIF configuration, emits lifPointUpdate(),
    /// and advances to the next scan point if the current point is complete.
    /// Calls checkComplete() to detect overall experiment completion.
    ///
    /// \param b Raw 8-bit waveform bytes from the LIF digitizer.
    void processLifScopeShot(const QVector<qint8> b);

    /// \brief Notifies the manager of the outcome of a LIF hardware configuration step.
    ///
    /// If \a success is \c false, the experiment is aborted. If \c true, the LIF
    /// configuration is marked hardware-ready so shot acquisition can proceed.
    ///
    /// \param success \c true if the LIF delay and frequency were set successfully.
    void lifHardwareReady(bool success);

private:
    std::unique_ptr<QFutureWatcher<void>> pu_fw;
    std::unique_ptr<QFutureWatcher<FtmwProcessingResult>> pu_processingWatcher;
    std::shared_ptr<Experiment> ps_currentExperiment;
    AcquisitionState d_state;
    int d_auxTimerId;
    QTimer *p_drainTimer{nullptr};
    std::atomic<bool> d_abortProcessing{false};

    void auxDataTick();
    void checkComplete();
    void finishAcquisition();
    void drainFtmwBuffer();
    void onProcessingComplete();

    // QObject interface
protected:
    /// \brief Handles the periodic aux-data timer tick.
    ///
    /// On each firing of the aux-data interval timer, reads FTMW shot counts and
    /// optional phase/chirp metrics, then calls processAuxData() and emits auxDataSignal().
    /// \param event The timer event; accepted if it matches the aux-data timer ID.
    void timerEvent(QTimerEvent *event) override;
};

#endif // ACQUISITIONMANAGER_H
