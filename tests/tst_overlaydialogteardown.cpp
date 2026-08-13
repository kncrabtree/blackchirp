#include <QtTest>

#include <QSignalSpy>
#include <memory>

#include <data/analysis/ft.h>
#include <data/experiment/overlaybase.h>
#include <data/processing/overlayprocessmanager.h>
#include <data/storage/overlaystorage.h>
#include <gui/overlay/catalogoverlaywidget.h>
#include <gui/overlay/unifiedoverlaydialog.h>

using namespace Qt::Literals::StringLiterals;

/*!
 * \brief Regression coverage for the ~UnifiedOverlayDialog() destructor
 *        ordering invariant: disconnect before cancel.
 *
 * Child widgets (UnifiedOverlayWidget, and the type-specific widget
 * beneath it) are destroyed during this dialog's base-class teardown --
 * after ~UnifiedOverlayDialog()'s own body has run and its own members
 * are gone, but while its signal/slot connections to
 * OverlayProcessManager are still live unless that body severs them
 * first. CatalogOverlayWidget's own destructor cancels any operation it
 * still owns, and OverlayProcessManager::cancelOperation() emits
 * operationCancelled() synchronously; if the dialog is still connected
 * to that signal at that point, the resulting slot call runs against
 * this dialog's already-destructed members.
 *
 * This is the destructor-ordering half of the fix in "Make overlay
 * background operations thread-safe and cancellation-correct"; the
 * manager-side re-entrancy half is covered separately by
 * tst_overlayprocessmanager_reentrancy.cpp.
 */
class OverlayDialogTeardownTest : public QObject
{
    Q_OBJECT

private slots:
    void destroyingDialogWithQueuedOperationDoesNotCrash();
};

void OverlayDialogTeardownTest::destroyingDialogWithQueuedOperationDoesNotCrash()
{
    // number < 1 short-circuits OverlayStorage/DataStorageBase before any
    // filesystem access, so this is safe to construct headless without a
    // real experiment directory.
    auto storage = std::make_shared<OverlayStorage>(0, QString());

    auto dialog = std::make_unique<UnifiedOverlayDialog>(
        OverlayBase::Catalog, QStringList(), Ft(), storage, nullptr);

    // No FtmwViewWidget ancestor is constructed -- only
    // OverlayManagerWidget::findFtmwView() needs one, not the dialog
    // itself -- so this dialog can be built headless.
    auto *catalogWidget = dialog->findChild<CatalogOverlayWidget *>();
    QVERIFY(catalogWidget != nullptr);

    QSignalSpy queuedSpy(catalogWidget, &OverlayTypeSpecificWidget::operationQueued);

    // Point the widget at a file. It does not need to exist: parsing
    // happens on a background thread that this test never lets run (no
    // event loop is pumped between here and the dialog's destruction),
    // so only the queueing side of the path is exercised -- the
    // operation is still sitting in OverlayProcessManager's queue, never
    // dequeued, exactly like a parse or convolution queued from a
    // real-time settings edit an instant before the dialog is closed.
    //
    // setSourceFilePath() queues more than once: updatePathDisplayAndTooltip()
    // sets the line edit's text, which fires the textChanged connection to
    // onFilePathChanged() (queuing a parse), and then setSourceFilePath()
    // calls onFilePathChanged() again directly (cancelling that parse and
    // queuing a replacement). Only the last id is still Queued afterward;
    // take that one rather than assume a fixed signal count.
    catalogWidget->setSourceFilePath(u"/nonexistent/regression-teardown.cat"_s);

    QVERIFY(queuedSpy.count() >= 1);
    const QString operationId = queuedSpy.constLast().at(0).toString();
    QVERIFY(!operationId.isEmpty());

    auto &manager = OverlayProcessManager::instance();
    QCOMPARE(manager.getOperationState(operationId),
              OverlayProcessManager::OperationState::Queued);

    // Destroy the dialog while the operation is still queued. If
    // ~UnifiedOverlayDialog() does not disconnect from
    // OverlayProcessManager before its base-class teardown deletes
    // p_widget (and, beneath it, catalogWidget), catalogWidget's own
    // destructor cancelling this operation reenters this dialog's
    // already-destructed onOperationCancelled() -- reliably a crash or
    // corruption, not merely a logic error, since it runs against a
    // std::set whose destructor has already completed.
    dialog.reset();

    // The dialog is gone, but OverlayProcessManager -- a process-wide
    // singleton that outlives every dialog -- must be unharmed by its
    // teardown: the operation actually reached Cancelled, and the
    // manager keeps answering queries normally afterward.
    QCOMPARE(manager.getOperationState(operationId),
              OverlayProcessManager::OperationState::Cancelled);
    QCOMPARE(manager.queueSize(), 0);
}

QTEST_MAIN(OverlayDialogTeardownTest)
#include "tst_overlaydialogteardown.moc"
