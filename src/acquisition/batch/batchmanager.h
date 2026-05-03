#ifndef BATCHMANAGER_H
#define BATCHMANAGER_H

#include <QObject>

#include <data/experiment/experiment.h>

/// \brief Abstract base class that iterates one or more experiments through
/// AcquisitionManager and signals completion to the main window.
///
/// BatchManager coordinates the lifecycle of an acquisition run, whether it
/// consists of a single experiment (BatchSingle) or a timed sequence
/// (BatchSequence). The main window constructs a concrete subclass and passes
/// it to MainWindow::startBatch(), which connects the batch's signals to
/// AcquisitionManager and the UI, then triggers the first experiment via
/// HardwareManager::initializeExperiment().
///
/// BatchManager lives on the main (GUI) thread. Its signals cross into the
/// AcquisitionManager thread via queued connections; its slots are called
/// from the GUI thread by the main window or by AcquisitionManager's
/// experimentComplete() signal.
///
/// Subclass authors must implement the five pure-virtual hooks described
/// below. The optional beginNextExperiment() override is available when the
/// default behavior — emit beginExperiment() immediately — is not appropriate
/// (for example, BatchSequence delays the next experiment by a configurable
/// interval).
///
/// \sa AcquisitionManager, BatchSingle, BatchSequence
class BatchManager : public QObject
{
    Q_OBJECT
public:
    /// \brief Identifies the concrete batch type in use.
    enum BatchType
    {
        SingleExperiment, ///< A single experiment managed by BatchSingle.
        Sequence          ///< A timed multi-experiment sequence managed by BatchSequence.
    };

    /// \brief Returns the experiment that is in progress or about to be acquired.
    ///
    /// Called by the main window inside the beginExperiment() signal handler to
    /// pass the experiment to HardwareManager::initializeExperiment(). Also called
    /// by experimentComplete() to inspect the result of the last acquisition.
    ///
    /// \return Shared pointer to the active experiment. Must never be null while
    ///         the batch is in progress.
    virtual std::shared_ptr<Experiment> currentExperiment() = 0;

    /// \brief Constructs the base with the given batch type.
    /// \param b The BatchType that identifies the concrete subclass.
    explicit BatchManager(BatchType b);

    /// \brief Destroys the batch manager.
    virtual ~BatchManager();

    /// \brief Returns \c true when no further experiments remain to be acquired.
    ///
    /// Evaluated inside experimentComplete() after processExperiment() returns.
    /// If \c true, the batch writes its report and emits batchComplete().
    ///
    /// \return \c true if the batch sequence has finished all configured experiments.
    virtual bool isComplete() = 0;

signals:
    /// \brief Requests that a transient status string be displayed in the status bar.
    /// \param msg The status text.
    /// \param timeout Display duration in milliseconds; 0 means indefinite.
    void statusMessage(QString msg, int timeout = 0);

    /// \brief Requests that a message be appended to the application log.
    /// \param msg The text to log.
    /// \param code Severity classification; defaults to LogHandler::Normal.
    void logMessage(QString msg, LogHandler::MessageCode code = LogHandler::Normal);

    /// \brief Signals that the next experiment in the batch should begin.
    ///
    /// The main window connects this signal to a lambda that calls
    /// HardwareManager::initializeExperiment(currentExperiment()). Emitted by
    /// beginNextExperiment() (or by an overriding subclass after a delay).
    void beginExperiment();

    /// \brief Emitted when the batch has finished, either normally or by abort.
    ///
    /// Connected to MainWindow::batchComplete(). After this signal, the batch
    /// manager is no longer active and will be deleted on the next batch start.
    ///
    /// \param aborted \c true if the last experiment was aborted; \c false for normal completion.
    void batchComplete(bool aborted);

public slots:
    /// \brief Responds to AcquisitionManager::experimentComplete().
    ///
    /// Logs the experiment result, calls processExperiment() if initialization
    /// succeeded, then either advances to the next experiment via
    /// beginNextExperiment() or ends the batch by calling writeReport() and
    /// emitting batchComplete().
    ///
    /// The decision tree is:
    /// - If init succeeded and batch is not complete: call beginNextExperiment().
    /// - If init succeeded and batch is complete: write report, emit batchComplete(false).
    /// - If the experiment was aborted or init failed: call abort(), write report,
    ///   emit batchComplete(true).
    ///
    /// \note This slot and AcquisitionManager::experimentComplete() share the same
    ///       name but are different entities: the AM signal triggers this BM slot.
    void experimentComplete();

    /// \brief Advances to the next experiment in the batch.
    ///
    /// The default implementation emits beginExperiment() immediately.
    /// Subclasses may override to insert a delay or other inter-experiment logic
    /// (see BatchSequence).
    ///
    /// \note Called by experimentComplete() when the batch is not yet complete
    ///       and initialization of the previous experiment succeeded.
    virtual void beginNextExperiment();

    /// \brief Aborts the batch unconditionally.
    ///
    /// Implementations must mark the batch as complete so that isComplete()
    /// returns \c true and no further experiments are started. They must also
    /// ensure that any pending timers or resources are released.
    ///
    /// \note Called by experimentComplete() when the experiment was aborted or
    ///       hardware initialization failed, and by the main window abort button.
    virtual void abort() = 0;

protected:
    BatchType d_type; ///< The concrete batch type set at construction.

    /// \brief Writes a summary report for the completed batch.
    ///
    /// Called by experimentComplete() after the last experiment finishes (either
    /// normally or by abort) and before batchComplete() is emitted. Implementations
    /// may write a file, log a summary, or take no action.
    virtual void writeReport() = 0;

    /// \brief Performs any post-acquisition processing for the most recently completed experiment.
    ///
    /// Called by experimentComplete() after a successful initialization/acquisition
    /// cycle, before isComplete() is evaluated. Implementations may re-analyze
    /// data, update state, or do nothing.
    virtual void processExperiment() = 0;
};

#endif //BATCHMANAGER_H
