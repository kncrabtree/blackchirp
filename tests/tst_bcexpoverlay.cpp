#include <QtTest>
#include <QTemporaryDir>

#include <data/experiment/overlaytypes.h>
#include <data/storage/overlaystorage.h>
#include <data/analysis/ft.h>

/*!
 * \brief Covers the VScale ignore band that BCExpOverlay carries with its FT.
 *
 * An FT imported as an overlay keeps the half-width around the LO that the
 * plot it came from suppressed. Autoscaling against the DC spike inside that
 * band collapses the overlay to a sliver, so yMax() must exclude it, and the
 * band has to survive the trip through the destination file.
 */
class BCExpOverlayTest : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void yMaxExcludesIgnoredBand();
    void yMaxWithoutIgnoreBandSeesEverything();
    void ignoreBandSurvivesStorageRoundTrip();
    void yMaxFallsBackWhenBandCoversData();
    void displayYRangeExcludesIgnoredBand();
    void displayYRangeFollowsScaleAndOffset();

private:
    /*!
     * \brief Build an FT with a large spike at the LO and a smaller real peak away from it.
     *
     * The axis runs from \a lo to \a lo + 999 MHz in 1 MHz bins: the spike
     * spans the first 50 MHz and peaks at +5 MHz, the real peak sits at
     * +600 MHz. The spike peak is offset from the LO bin itself because that
     * bin is excluded even by a zero-width ignore band.
     */
    Ft makeSpikedFt(double lo = 10000.0) const
    {
        Ft ft(1000, lo, 1.0, lo);
        for (int i = 0; i < 1000; ++i)
            ft.setPoint(i, 1.0);

        for (int i = 0; i < 50; ++i)
            ft.setPoint(i, d_spike * (1.0 - qAbs(i - 5) / 50.0));

        ft.setPoint(600, d_realPeak);
        return ft;
    }

    static constexpr double d_spike{500.0};
    static constexpr double d_realPeak{20.0};
};

void BCExpOverlayTest::initTestCase()
{
    QCoreApplication::setOrganizationDomain("crabtreelab.ucdavis.edu");
    QCoreApplication::setApplicationName("BlackchirpTest");
}

void BCExpOverlayTest::yMaxExcludesIgnoredBand()
{
    BCExpOverlay o;
    o.setAutoScaleIgnoreMHz(250.0);
    o.setFtData(makeSpikedFt());

    // The spike lies inside the ignored band, so the real peak is the maximum.
    QCOMPARE(o.yMax(), d_realPeak);

    // The Ft's own extrema are restated under the same band.
    QCOMPARE(o.getFtData().yMax(), d_realPeak);
}

void BCExpOverlayTest::yMaxWithoutIgnoreBandSeesEverything()
{
    BCExpOverlay o;
    o.setFtData(makeSpikedFt());

    QCOMPARE(o.getAutoScaleIgnoreMHz(), 0.0);
    QCOMPARE(o.yMax(), d_spike);
}

void BCExpOverlayTest::ignoreBandSurvivesStorageRoundTrip()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    auto src = std::make_shared<BCExpOverlay>();
    src->setLabel("roundtrip");
    src->setAutoScaleIgnoreMHz(250.0);
    src->setFtData(makeSpikedFt());

    OverlayStorage writer(1, dir.path());
    QVERIFY(writer.addOverlay(src));
    writer.waitForPendingWrites();
    writer.save();

    // Reload through the same path the experiment loader uses: the settings
    // CSV, then the magnitude column from the destination file.
    OverlayStorage reader(1, dir.path());
    QVERIFY(reader.loadOverlay("roundtrip", OverlayBase::BCExperiment));

    auto overlays = reader.getAllOverlays();
    QCOMPARE(overlays.size(), 1);

    auto dest = std::dynamic_pointer_cast<BCExpOverlay>(overlays.constFirst());
    QVERIFY(dest != nullptr);

    QCOMPARE(dest->getAutoScaleIgnoreMHz(), 250.0);
    QCOMPARE(dest->getFtData().size(), src->getFtData().size());

    // The destination file holds magnitudes only, so this is the case that
    // regressed before: the band, not a cached extremum, is what restores it.
    QCOMPARE(dest->yMax(), d_realPeak);
}

void BCExpOverlayTest::yMaxFallsBackWhenBandCoversData()
{
    BCExpOverlay o;
    o.setAutoScaleIgnoreMHz(5000.0); // wider than the 1000 MHz span
    o.setFtData(makeSpikedFt());

    // Nothing survives the exclusion; reporting zero would disable autoscaling
    // altogether, so the full extent stands in.
    QCOMPARE(o.yMax(), d_spike);
}

void BCExpOverlayTest::displayYRangeExcludesIgnoredBand()
{
    BCExpOverlay o;
    o.setAutoScaleIgnoreMHz(250.0);
    o.setFtData(makeSpikedFt());

    // Scaling the overlay so its real peak is visible multiplies the spike by
    // the same factor; the plot must not stretch its axis to accommodate it.
    o.setYScale(10.0);

    const auto [lo, hi] = o.displayYRange();
    QCOMPARE(hi, d_realPeak * 10.0);
    QCOMPARE(lo, 1.0 * 10.0); // the baseline outside the band
}

void BCExpOverlayTest::displayYRangeFollowsScaleAndOffset()
{
    BCExpOverlay o;
    o.setAutoScaleIgnoreMHz(250.0);
    o.setFtData(makeSpikedFt());
    o.setYScale(-2.0);
    o.setYOffset(5.0);

    // Inverted: the real peak becomes the minimum.
    const auto [lo, hi] = o.displayYRange();
    QCOMPARE(lo, d_realPeak * -2.0 + 5.0);
    QCOMPARE(hi, 1.0 * -2.0 + 5.0);
}

QTEST_MAIN(BCExpOverlayTest)
#include "tst_bcexpoverlay.moc"
