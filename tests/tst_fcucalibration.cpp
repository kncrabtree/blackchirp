#include <QtTest>

#include <algorithm>
#include <cmath>
#include <limits>

#include <data/lif/fcucalibration.h>

using namespace Qt::Literals::StringLiterals;
using namespace BC::FcuCal;

class FcuCalibrationTest : public QObject
{
    Q_OBJECT

private slots:
    // Physical scheme
    void testPhysicalPhaseMatchAngleBbo();
    void testPhysicalPhaseMatchAngleKdp();
    void testPhysicalMonotoneBbo();
    void testPhysicalMonotoneKdp();
    void testPhysicalRoundTripBbo();
    void testPhysicalRoundTripKdp();
    void testPhysicalInvert();
    void testPhysicalRejectNonFinite();
    void testPhysicalRejectZeroScrewPitch();

    // Polynomial scheme
    void testPolynomialHornerEvaluation();
    void testPolynomialRoundTrip();
    void testPolynomialRejectEmptyCoeffs();
    void testPolynomialNormalizedEvaluation();
    void testPolynomialNormalizedRoundTrip();
    void testPolynomialRejectBadNormalization();

    // Spline scheme
    void testSplinePassesThroughPoints();
    void testSplineRoundTrip();
    void testSplineRejectTooFewPoints();
    void testSplineRejectNonMonotone();
    void testSplineCopyIsIndependentOfLifetime();

private:
    static bool close(double a, double b, double eps) { return std::abs(a - b) < eps; }
};

void FcuCalibrationTest::testPhysicalPhaseMatchAngleBbo()
{
    // Table 6-1: BBO cut angle 57.4 degrees. Scan the 430-560 nm fundamental
    // band (already validated in the plan's Python prototype) and require
    // the computed phase-match angle to come within a few degrees of the
    // tabulated cut angle somewhere in that band.
    double minDiff = std::numeric_limits<double>::max();
    for (double lam = 430.0; lam <= 560.0; lam += 5.0) {
        const double angle = FcuCalibration::phaseMatchAngleDeg(CrystalType::BBO, lam, 293.0);
        minDiff = std::min(minDiff, std::abs(angle - 57.4));
    }
    QVERIFY(minDiff < 3.0);
}

void FcuCalibrationTest::testPhysicalPhaseMatchAngleKdp()
{
    // Table 6-1: KDP cut angle 63 degrees (SHG output 260-380 nm, i.e.
    // fundamental 520-760 nm).
    double minDiff = std::numeric_limits<double>::max();
    for (double lam = 520.0; lam <= 760.0; lam += 5.0) {
        const double angle = FcuCalibration::phaseMatchAngleDeg(CrystalType::KDP, lam, 293.0);
        minDiff = std::min(minDiff, std::abs(angle - 63.0));
    }
    QVERIFY(minDiff < 3.0);
}

void FcuCalibrationTest::testPhysicalMonotoneBbo()
{
    auto cal = FcuCalibration::physical(CrystalType::BBO, 57.4, 293.0, 50.0, 10.0, 2.0, 30.0, 5000.0, false);
    QVERIFY2(cal.isValid(), qPrintable(cal.errorString()));

    double prev = cal.wavelengthToPos(430.0);
    for (double lam = 440.0; lam <= 560.0; lam += 10.0) {
        const double pos = cal.wavelengthToPos(lam);
        QVERIFY(pos < prev); // strictly decreasing across the whole band
        prev = pos;
    }
}

void FcuCalibrationTest::testPhysicalMonotoneKdp()
{
    auto cal = FcuCalibration::physical(CrystalType::KDP, 63.0, 293.0, 50.0, 10.0, 2.0, 30.0, 5000.0, false);
    QVERIFY2(cal.isValid(), qPrintable(cal.errorString()));

    double prev = cal.wavelengthToPos(520.0);
    for (double lam = 530.0; lam <= 760.0; lam += 10.0) {
        const double pos = cal.wavelengthToPos(lam);
        QVERIFY(pos < prev);
        prev = pos;
    }
}

void FcuCalibrationTest::testPhysicalRoundTripBbo()
{
    auto cal = FcuCalibration::physical(CrystalType::BBO, 57.4, 293.0, 50.0, 10.0, 2.0, 30.0, 5000.0, false);
    QVERIFY2(cal.isValid(), qPrintable(cal.errorString()));

    for (double lam : {440.0, 470.0, 500.0, 530.0, 555.0}) {
        const double pos = cal.wavelengthToPos(lam);
        const double back = cal.posToWavelength(pos);
        QVERIFY(!std::isnan(back));
        QVERIFY2(close(back, lam, 1e-4), qPrintable(u"lam=%1 back=%2"_s.arg(lam).arg(back)));
    }
}

void FcuCalibrationTest::testPhysicalRoundTripKdp()
{
    auto cal = FcuCalibration::physical(CrystalType::KDP, 63.0, 293.0, 50.0, 10.0, 2.0, 30.0, 5000.0, false);
    QVERIFY2(cal.isValid(), qPrintable(cal.errorString()));

    for (double lam : {525.0, 570.0, 620.0, 680.0, 750.0}) {
        const double pos = cal.wavelengthToPos(lam);
        const double back = cal.posToWavelength(pos);
        QVERIFY(!std::isnan(back));
        QVERIFY2(close(back, lam, 1e-4), qPrintable(u"lam=%1 back=%2"_s.arg(lam).arg(back)));
    }
}

void FcuCalibrationTest::testPhysicalInvert()
{
    // The invert flag flips the sign of (theta_pm - cutAngle) in the forward
    // map; with everything else held fixed, this changes the sine-bar
    // travel direction, so the two calibrations must disagree while each
    // remains internally round-trip-consistent.
    auto normal = FcuCalibration::physical(CrystalType::BBO, 57.4, 293.0, 50.0, 10.0, 2.0, 30.0, 5000.0, false);
    auto inverted = FcuCalibration::physical(CrystalType::BBO, 57.4, 293.0, 50.0, 10.0, 2.0, 30.0, 5000.0, true);
    QVERIFY2(normal.isValid(), qPrintable(normal.errorString()));
    QVERIFY2(inverted.isValid(), qPrintable(inverted.errorString()));

    const double lam = 495.0;
    QVERIFY(!close(normal.wavelengthToPos(lam), inverted.wavelengthToPos(lam), 1.0));

    const double pos = inverted.wavelengthToPos(lam);
    QVERIFY(close(inverted.posToWavelength(pos), lam, 1e-4));
}

void FcuCalibrationTest::testPhysicalRejectNonFinite()
{
    auto cal = FcuCalibration::physical(CrystalType::BBO, std::numeric_limits<double>::quiet_NaN(),
                                         293.0, 50.0, 10.0, 2.0, 30.0, 5000.0, false);
    QVERIFY(!cal.isValid());
    QVERIFY(!cal.errorString().isEmpty());
}

void FcuCalibrationTest::testPhysicalRejectZeroScrewPitch()
{
    auto cal = FcuCalibration::physical(CrystalType::BBO, 57.4, 293.0, 50.0, 10.0, 0.0, 30.0, 5000.0, false);
    QVERIFY(!cal.isValid());
    QVERIFY(!cal.errorString().isEmpty());
}

void FcuCalibrationTest::testPolynomialHornerEvaluation()
{
    // c0=1, c1=2, c2=3 -> 1 + 2x + 3x^2; at x=2: 1 + 4 + 12 = 17.
    auto cal = FcuCalibration::polynomial({1.0, 2.0, 3.0}, {0.0, 1.0});
    QVERIFY2(cal.isValid(), qPrintable(cal.errorString()));
    QCOMPARE(cal.wavelengthToPos(2.0), 17.0);
}

void FcuCalibrationTest::testPolynomialRoundTrip()
{
    // pos = 100 + 50*lam is exactly inverted by lam = -2 + 0.02*pos.
    auto cal = FcuCalibration::polynomial({100.0, 50.0}, {-2.0, 0.02});
    QVERIFY2(cal.isValid(), qPrintable(cal.errorString()));

    for (double lam : {400.0, 500.0, 600.0}) {
        const double pos = cal.wavelengthToPos(lam);
        QCOMPARE(pos, 100.0 + 50.0*lam);
        const double back = cal.posToWavelength(pos);
        QVERIFY(close(back, lam, 1e-9));
    }
}

void FcuCalibrationTest::testPolynomialRejectEmptyCoeffs()
{
    auto cal1 = FcuCalibration::polynomial({}, {1.0, 2.0});
    QVERIFY(!cal1.isValid());
    QVERIFY(!cal1.errorString().isEmpty());

    auto cal2 = FcuCalibration::polynomial({1.0, 2.0}, {});
    QVERIFY(!cal2.isValid());
    QVERIFY(!cal2.errorString().isEmpty());

    auto cal3 = FcuCalibration::polynomial({}, {});
    QVERIFY(!cal3.isValid());
}

void FcuCalibrationTest::testPolynomialNormalizedEvaluation()
{
    // Forward evaluated at u = (lam - 560)/4: at lam = 568, u = 2 and
    // 1 + 2u + 3u^2 = 17. Inverse at u = (pos - 1000)/10: at pos = 1030,
    // u = 3 and 5 + 2u = 11.
    auto cal = FcuCalibration::polynomial({1.0, 2.0, 3.0}, {5.0, 2.0}, 560.0, 4.0, 1000.0, 10.0);
    QVERIFY2(cal.isValid(), qPrintable(cal.errorString()));
    QCOMPARE(cal.wavelengthToPos(568.0), 17.0);
    QCOMPARE(cal.posToWavelength(1030.0), 11.0);
}

void FcuCalibrationTest::testPolynomialNormalizedRoundTrip()
{
    // A quadratic tuning curve near 563 nm and 1.64e7 steps, the regime
    // where raw-unit coefficients cancel heavily. Normalized, a quintic
    // least-squares inverse of the forward map reproduces the wavelength
    // to well under a step's worth.
    const double lc = 563.0, ls = 3.0, pc = 16412000.0, ps = 15000.0;
    const std::vector<double> fwd{16412000.0, 14300.0, 120.0};
    auto forward = [&](double lam){
        auto u = (lam - lc)/ls;
        return fwd[0] + fwd[1]*u + fwd[2]*u*u;
    };

    // Inverse coefficients from a least-squares quintic in normalized
    // position, sampled densely across the band.
    const int n = 61, deg = 5;
    std::vector<double> us, lams;
    for(int i=0; i<n; i++)
    {
        auto lam = lc - ls + 2.0*ls*i/(n-1);
        us.push_back((forward(lam) - pc)/ps);
        lams.push_back(lam);
    }
    std::vector<std::vector<double>> ata(deg+1, std::vector<double>(deg+2, 0.0));
    for(int k=0; k<n; k++)
        for(int r=0; r<=deg; r++)
        {
            for(int c=0; c<=deg; c++)
                ata[r][c] += std::pow(us[k],r+c);
            ata[r][deg+1] += std::pow(us[k],r)*lams[k];
        }
    for(int c=0; c<=deg; c++)
        for(int r=c+1; r<=deg; r++)
        {
            auto f = ata[r][c]/ata[c][c];
            for(int j=c; j<=deg+1; j++)
                ata[r][j] -= f*ata[c][j];
        }
    std::vector<double> inv(deg+1);
    for(int r=deg; r>=0; r--)
    {
        auto v = ata[r][deg+1];
        for(int j=r+1; j<=deg; j++)
            v -= ata[r][j]*inv[j];
        inv[r] = v/ata[r][r];
    }

    auto cal = FcuCalibration::polynomial(fwd, inv, lc, ls, pc, ps);
    QVERIFY2(cal.isValid(), qPrintable(cal.errorString()));
    for(double lam : {560.5, 562.36, 563.0, 565.9})
    {
        QCOMPARE(cal.wavelengthToPos(lam), forward(lam));
        // 1e-4 nm is about half a step on this curve.
        QVERIFY(close(cal.posToWavelength(cal.wavelengthToPos(lam)), lam, 1e-4));
    }
}

void FcuCalibrationTest::testPolynomialRejectBadNormalization()
{
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    QVERIFY(!FcuCalibration::polynomial({1.0}, {1.0}, 560.0, 0.0, 0.0, 1.0).isValid());
    QVERIFY(!FcuCalibration::polynomial({1.0}, {1.0}, 0.0, 1.0, 0.0, 0.0).isValid());
    QVERIFY(!FcuCalibration::polynomial({1.0}, {1.0}, nan, 1.0, 0.0, 1.0).isValid());
    auto cal = FcuCalibration::polynomial({1.0}, {1.0}, 0.0, 1.0, 0.0, nan);
    QVERIFY(!cal.isValid());
    QVERIFY(!cal.errorString().isEmpty());
}

void FcuCalibrationTest::testSplinePassesThroughPoints()
{
    const std::vector<std::pair<double,double>> points = {
        {400.0, 1000.0}, {450.0, 1200.0}, {500.0, 1500.0}, {550.0, 1900.0}, {600.0, 2400.0}
    };
    auto cal = FcuCalibration::spline(points);
    QVERIFY2(cal.isValid(), qPrintable(cal.errorString()));

    for (const auto &p : points) {
        QVERIFY2(close(cal.wavelengthToPos(p.first), p.second, 1e-6),
                  qPrintable(u"lam=%1"_s.arg(p.first)));
        QVERIFY2(close(cal.posToWavelength(p.second), p.first, 1e-6),
                  qPrintable(u"pos=%1"_s.arg(p.second)));
    }
}

void FcuCalibrationTest::testSplineRoundTrip()
{
    // Unsorted input; the factory must sort internally.
    const std::vector<std::pair<double,double>> points = {
        {550.0, 1900.0}, {400.0, 1000.0}, {600.0, 2400.0}, {450.0, 1200.0}, {500.0, 1500.0}
    };
    auto cal = FcuCalibration::spline(points);
    QVERIFY2(cal.isValid(), qPrintable(cal.errorString()));

    // The forward and inverse splines are fit independently (distinct knot
    // placement in each direction), so a non-grid-point round trip is a
    // close approximate inverse rather than exact; 1 nm is comfortably
    // above the interpolation error for this smooth, widely-spaced table.
    for (double lam : {420.0, 475.0, 525.0, 580.0}) {
        const double pos = cal.wavelengthToPos(lam);
        const double back = cal.posToWavelength(pos);
        QVERIFY(!std::isnan(back));
        QVERIFY2(close(back, lam, 1.0), qPrintable(u"lam=%1 back=%2"_s.arg(lam).arg(back)));
    }

    // Outside the point table's domain: NaN by convention, not extrapolation.
    QVERIFY(std::isnan(cal.wavelengthToPos(399.0)));
    QVERIFY(std::isnan(cal.wavelengthToPos(601.0)));
    QVERIFY(std::isnan(cal.posToWavelength(999.0)));
    QVERIFY(std::isnan(cal.posToWavelength(2401.0)));
}

void FcuCalibrationTest::testSplineRejectTooFewPoints()
{
    auto cal0 = FcuCalibration::spline({});
    QVERIFY(!cal0.isValid());
    QVERIFY(!cal0.errorString().isEmpty());

    auto cal1 = FcuCalibration::spline({{400.0, 1000.0}});
    QVERIFY(!cal1.isValid());
    QVERIFY(!cal1.errorString().isEmpty());
}

void FcuCalibrationTest::testSplineRejectNonMonotone()
{
    // Positions go up, then down: not invertible.
    const std::vector<std::pair<double,double>> points = {
        {400.0, 1000.0}, {450.0, 1200.0}, {500.0, 900.0}, {550.0, 1900.0}
    };
    auto cal = FcuCalibration::spline(points);
    QVERIFY(!cal.isValid());
    QVERIFY(!cal.errorString().isEmpty());
}

void FcuCalibrationTest::testSplineCopyIsIndependentOfLifetime()
{
    // The value type must be safely copyable: a copy must remain usable
    // after the original goes out of scope (exercises the shared_ptr +
    // custom-deleter GSL ownership strategy).
    FcuCalibration copy;
    {
        const std::vector<std::pair<double,double>> points = {
            {400.0, 1000.0}, {450.0, 1200.0}, {500.0, 1500.0}
        };
        auto original = FcuCalibration::spline(points);
        QVERIFY2(original.isValid(), qPrintable(original.errorString()));
        copy = original;
    }
    QVERIFY(copy.isValid());
    QVERIFY(close(copy.wavelengthToPos(450.0), 1200.0, 1e-6));
}

QTEST_APPLESS_MAIN(FcuCalibrationTest)

#include "tst_fcucalibration.moc"
