#include <QtTest>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include <src/data/processing/overlayoperation.h>
#include <src/data/processing/overlayprocessmanager.h>

using namespace Qt::Literals::StringLiterals;

namespace {

/// \brief A minimal operation whose execute() body never actually runs
///        in this test: it is cancelled while still Queued, before
///        OverlayProcessManager ever dequeues it.
class NoOpOperation : public OverlayOperation
{
    Q_OBJECT
public:
    explicit NoOpOperation(QObject *parent = nullptr)
        : OverlayOperation(Type::Immediate, Priority::Normal, parent) {}

    std::shared_ptr<OverlayBase> execute() override { return nullptr; }
    void cancel() override { d_cancelled = true; }
    bool canCancel() const override { return true; }
    bool producesOverlay() const override { return false; }
    QString getDescription() const override { return "No-op reentrancy test operation"_L1; }
    QString getOperationName() const override { return "NoOp"_L1; }
};

/// \brief Receiver whose slot re-enters OverlayProcessManager from
///        inside a direct-connected operationCancelled() emission --
///        the same shape as a widget's manager-signal handler calling
///        back into the manager while
///        OverlayProcessManager::cancelOperation() is still on the
///        call stack (e.g. a destructor that has not yet disconnected).
class ReentrantCancelReceiver : public QObject
{
    Q_OBJECT
public:
    ReentrantCancelReceiver(OverlayProcessManager &manager, QString operationId)
        : d_manager(manager), d_operationId(std::move(operationId)) {}

    std::atomic<bool> slotRan{false};
    std::atomic<bool> callReturned{false};

public slots:
    void onOperationCancelled(const QString &operationId)
    {
        if (operationId != d_operationId)
            return;
        slotRan = true;
        // Re-entrant calls into the manager while cancelOperation() is
        // still on the call stack. With d_mutex released before the
        // emit, these return normally. If the manager ever regresses
        // to emitting operationCancelled() while still holding
        // d_mutex, these calls deadlock the calling thread against
        // itself (QMutex is non-recursive).
        d_manager.queueSize();
        d_manager.operation(d_operationId);
        callReturned = true;
    }

private:
    OverlayProcessManager &d_manager;
    QString d_operationId;
};

} // namespace

/// \brief Regression coverage for the OverlayProcessManager::cancelOperation()
///        re-entrancy contract: a slot connected to operationCancelled()
///        must be able to call back into the manager without deadlocking.
///
/// This is the manager-side half of the use-after-free fix for closing
/// an overlay dialog while a background operation is queued/running: a
/// widget's destructor disconnects from the manager and then cancels
/// its own operations, but other still-connected objects (e.g. an
/// owning dialog) receive operationCancelled() synchronously and may
/// call back into the manager from their handler. That re-entry must
/// not deadlock against OverlayProcessManager::d_mutex.
class OverlayProcessManagerReentrancyTest : public QObject
{
    Q_OBJECT

private slots:
    void cancelOperationDoesNotDeadlockOnReentrantCallback();
};

void OverlayProcessManagerReentrancyTest::cancelOperationDoesNotDeadlockOnReentrantCallback()
{
    auto &manager = OverlayProcessManager::instance();

    // The queue/connect/cancel sequence runs on a dedicated thread and
    // the wait for it is bounded: if
    // OverlayProcessManager::cancelOperation() ever regresses to
    // emitting operationCancelled() while still holding d_mutex, the
    // reentrant queueSize()/operation() calls inside the receiver's
    // slot deadlock that thread against itself. A regression there
    // must fail this test, not hang it (and CI with it), so success is
    // observed through a std::future with a bounded wait rather than
    // an unconditional thread join.
    auto testBody = [&manager]() -> bool {
        auto operation = std::make_shared<NoOpOperation>();
        QString operationId = manager.queueOperation(operation, OverlayProcessManager::Priority::Normal);
        if (operationId.isEmpty())
            return false;

        ReentrantCancelReceiver receiver(manager, operationId);
        QObject::connect(&manager, &OverlayProcessManager::operationCancelled,
                          &receiver, &ReentrantCancelReceiver::onOperationCancelled,
                          Qt::DirectConnection);

        // Cancelled while still Queued: queueOperation() only schedules
        // processQueue() via a queued invocation, so with no event loop
        // pumped in between, the operation is still sitting in the
        // queue here -- the same state a widget's own queued
        // convolution/parse is in when a dialog is closed before that
        // operation starts running.
        bool cancelled = manager.cancelOperation(operationId);

        QObject::disconnect(&manager, &OverlayProcessManager::operationCancelled,
                             &receiver, &ReentrantCancelReceiver::onOperationCancelled);

        return cancelled && receiver.slotRan.load() && receiver.callReturned.load();
    };

    std::packaged_task<bool()> task(testBody);
    std::future<bool> result = task.get_future();
    std::thread worker(std::move(task));

    constexpr auto timeout = std::chrono::seconds(5);
    if (result.wait_for(timeout) != std::future_status::ready) {
        // The worker is presumably deadlocked on d_mutex; joining it
        // would hang this test right along with it. Detach instead so
        // the process can still exit and report the failure.
        worker.detach();
        QFAIL("OverlayProcessManager::cancelOperation() deadlocked on a "
              "reentrant callback from operationCancelled()");
    }

    worker.join();
    QVERIFY(result.get());
}

// An end-to-end variant of this test -- constructing a real
// UnifiedOverlayDialog/CatalogOverlayWidget tree, queuing an operation,
// and destroying the dialog while it is still Queued -- was attempted
// and dropped. See the accompanying report for why: it is a build-graph
// blocker, not a logic one. Summary: any test binary that instantiates
// a Q_OBJECT class from the blackchirp-gui static library links in that
// library's single combined AUTOMOC translation unit
// (blackchirp-gui_autogen/mocs_compilation.cpp), which carries
// MainWindow's moc output alongside every other widget's; that in turn
// requires mainwindow.cpp.o, which references AcquisitionManager and
// BatchManager -- symbols that exist only in the main blackchirp
// executable's own private sources, not in blackchirp-data,
// blackchirp-gui, or blackchirp-hardware. There is no dialog-specific
// dependency (no FtmwViewWidget ancestor is required); the blocker is
// structural to how this target's AUTOMOC output is linked.

QTEST_MAIN(OverlayProcessManagerReentrancyTest)
#include "tst_overlayprocessmanager_reentrancy.moc"
