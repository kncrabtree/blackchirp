#include <QtTest>
#include <QSignalSpy>

#include <data/storage/overlaystorage.h>
#include <data/experiment/overlaytypes.h>

/*!
 * \brief Covers OverlayStorage's preview collection being keyed by object
 * identity rather than by label.
 *
 * Before the fix, d_previewOverlays was a std::map<QString, shared_ptr>
 * keyed by getLabel(). Two consequences followed, both reachable on the
 * cancel path: a preview renamed after registration (CatalogOverlayWidget
 * sets the label from the parsed molecule name only after the preview is
 * already registered) would leave a stale map entry that removePreviewOverlay()
 * could never reach by its new label; and two previews -- or a preview and
 * an unrelated persistent overlay -- sharing a label would silently
 * collide, with the second addPreviewOverlay() overwriting the first's
 * entry and a later removal call affecting the wrong object.
 *
 * This covers only the OverlayStorage half of defect B. The FtPlot half
 * (addOverlay()/removeOverlay()/updateOverlay()/hasOverlay() also switched
 * from label to identity comparison) is not covered here: constructing an
 * FtPlot in a test requires linking blackchirp-gui, whose combined AUTOMOC
 * translation unit pulls in MainWindow and thence AcquisitionManager, which
 * tst_zoompanplotthreadsafety.cpp works around by compiling the plot-layer
 * sources directly rather than linking the library. Doing the same here
 * for a single identity-comparison check was judged not worth the
 * duplicated build surface; see the test plan discussion for this defect.
 */
class OverlayStoragePreviewIdentityTest : public QObject
{
    Q_OBJECT

private slots:
    void twoPreviewsSharingALabelAreIndependent();
    void removePreviewOverlayMatchesByIdentityNotLabel();
    void detachPreviewOverlayDoesNotEmitRemoved();
    void addingSameOverlayTwiceDoesNotDuplicate();

private:
    static std::shared_ptr<GenericXYOverlay> makeOverlay(const QString &label);
};

std::shared_ptr<GenericXYOverlay> OverlayStoragePreviewIdentityTest::makeOverlay(const QString &label)
{
    auto overlay = std::make_shared<GenericXYOverlay>();
    overlay->setLabel(label);
    overlay->setPreview(true);
    return overlay;
}

// number = -1 marks a transient instance: DataStorageBase performs no disk
// I/O for it, which keeps this a pure in-memory test of the preview
// collection.

void OverlayStoragePreviewIdentityTest::twoPreviewsSharingALabelAreIndependent()
{
    OverlayStorage storage(-1, QString());

    auto first = makeOverlay("Same Label");
    auto second = makeOverlay("Same Label");
    QVERIFY(first != second);

    QVERIFY(storage.addPreviewOverlay(first));
    QVERIFY(storage.addPreviewOverlay(second));

    // A label-keyed map would have let the second insertion silently
    // overwrite the first's entry; both must be present.
    auto previews = storage.getAllPreviewOverlays();
    QCOMPARE(previews.size(), 2);
    QVERIFY(previews.contains(first));
    QVERIFY(previews.contains(second));

    // Removing one by identity must not affect the other, even though
    // they share a label.
    QVERIFY(storage.removePreviewOverlay(first));
    previews = storage.getAllPreviewOverlays();
    QCOMPARE(previews.size(), 1);
    QVERIFY(!previews.contains(first));
    QVERIFY(previews.contains(second));

    QVERIFY(storage.removePreviewOverlay(second));
    QCOMPARE(storage.getAllPreviewOverlays().size(), 0);
}

void OverlayStoragePreviewIdentityTest::removePreviewOverlayMatchesByIdentityNotLabel()
{
    OverlayStorage storage(-1, QString());

    // Simulate CatalogOverlayWidget renaming a preview from its
    // auto-detected molecule name after the preview was already
    // registered under the original (creation-time) label.
    auto overlay = makeOverlay("Untitled");
    QVERIFY(storage.addPreviewOverlay(overlay));

    overlay->setLabel("Renamed Molecule");

    // A label-keyed removePreviewOverlay(QString) would look up
    // "Renamed Molecule" and find nothing, since the map entry is still
    // keyed by "Untitled". The identity-keyed overload has no such
    // dependency on the label staying put.
    QVERIFY(storage.removePreviewOverlay(overlay));
    QCOMPARE(storage.getAllPreviewOverlays().size(), 0);

    // Removing again (already gone) must fail cleanly, not crash.
    QVERIFY(!storage.removePreviewOverlay(overlay));
}

void OverlayStoragePreviewIdentityTest::detachPreviewOverlayDoesNotEmitRemoved()
{
    OverlayStorage storage(-1, QString());

    auto overlay = makeOverlay("Promoted");
    QVERIFY(storage.addPreviewOverlay(overlay));

    QSignalSpy removedSpy(&storage, &OverlayStorage::overlayRemoved);

    // Promotion path: detach must remove the preview entry without
    // signalling overlayRemoved, so the curve already on the plot survives
    // the transition to permanent storage (see OverlayManagerWidget::addOverlay()).
    QVERIFY(storage.detachPreviewOverlay(overlay));
    QCOMPARE(storage.getAllPreviewOverlays().size(), 0);
    QCOMPARE(removedSpy.count(), 0);
}

void OverlayStoragePreviewIdentityTest::addingSameOverlayTwiceDoesNotDuplicate()
{
    OverlayStorage storage(-1, QString());

    auto overlay = makeOverlay("Dup");
    QVERIFY(storage.addPreviewOverlay(overlay));
    QVERIFY(storage.addPreviewOverlay(overlay));

    QCOMPARE(storage.getAllPreviewOverlays().size(), 1);

    // A single removal is then enough to clear it.
    QVERIFY(storage.removePreviewOverlay(overlay));
    QCOMPARE(storage.getAllPreviewOverlays().size(), 0);
}

QTEST_MAIN(OverlayStoragePreviewIdentityTest)
#include "tst_overlaystoragepreviewidentity.moc"
