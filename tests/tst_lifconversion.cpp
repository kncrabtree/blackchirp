#include <QtTest>

#include <data/lif/lifconversion.h>
#include <data/lif/lifunits.h>

using namespace BC::LifConv;

class LifConversionTest : public QObject
{
    Q_OBJECT

private slots:
    // BC::LifConv::toCm1/fromCm1 (lifunits.h)
    void testUnitRoundTrips();
    void testUnitGuards();
    void testUnitLabels();

    // LifConversion (lifconversion.h)
    void testIdentity();
    void testNhgRoundTrip();
    void testSfgRoundTrip();
    void testDfgRoundTrip();
    void testOutputRange();
    void testOutputRangeInversion();
    void testTriplerWithLaserSecondInput();
    void testTriplerWithFixedSecondInput();
    void testStageOutput();

    void testRejectUnresolvedStageRef();
    void testRejectWrongArity();
    void testRejectNhgOrderBelowOne();
    void testRejectNoFinal();
    void testRejectMultipleFinal();
    void testRejectCycle();
    void testRejectDuplicateStageKey();
    void testRejectNoNetTunableDependence();

private:
    static bool close(double a, double b, double eps = 1e-6) { return std::abs(a - b) < eps; }
};

void LifConversionTest::testUnitRoundTrips()
{
    // Cm1 is the identity mapping.
    QCOMPARE(toCm1(1234.5, LaserUnit::Cm1), 1234.5);
    QCOMPARE(fromCm1(1234.5, LaserUnit::Cm1), 1234.5);

    // Nm is reciprocal in wavelength: lambda_nm = 1e7/cm1.
    const double cm1 = 20000.0; // -> 500 nm
    const double nm = fromCm1(cm1, LaserUnit::Nm);
    QVERIFY(close(nm, 500.0));
    QVERIFY(close(toCm1(nm, LaserUnit::Nm), cm1));

    // GHz: f_GHz = 29.9792458 * cm1 (affine, exact both ways).
    const double ghz = fromCm1(cm1, LaserUnit::GHz);
    QVERIFY(close(ghz, 29.9792458 * cm1));
    QVERIFY(close(toCm1(ghz, LaserUnit::GHz), cm1));

    // eV: E_eV = 1.239841984e-4 * cm1 (affine, exact both ways).
    const double ev = fromCm1(cm1, LaserUnit::eV);
    QVERIFY(close(ev, 1.239841984e-4 * cm1));
    QVERIFY(close(toCm1(ev, LaserUnit::eV), cm1, 1e-3));

    // Round trip starting from each unit's own natural value.
    for (auto u : {LaserUnit::Cm1, LaserUnit::Nm, LaserUnit::GHz, LaserUnit::eV}) {
        const double v = fromCm1(cm1, u);
        QVERIFY(close(toCm1(v, u), cm1, 1e-3));
    }
}

void LifConversionTest::testUnitGuards()
{
    // Nm is reciprocal in wavelength; non-positive input must not divide by
    // zero and must return a negative sentinel instead.
    QVERIFY(toCm1(0.0, LaserUnit::Nm) < 0.0);
    QVERIFY(toCm1(-5.0, LaserUnit::Nm) < 0.0);
    QVERIFY(fromCm1(0.0, LaserUnit::Nm) < 0.0);
    QVERIFY(fromCm1(-5.0, LaserUnit::Nm) < 0.0);

    // GHz and eV are affine (not reciprocal) in both directions, so they
    // never divide by a caller-supplied value and need no guard.
    QVERIFY(fromCm1(-5.0, LaserUnit::GHz) < 0.0); // ordinary linear result, not a sentinel path
    QVERIFY(close(fromCm1(-5.0, LaserUnit::GHz), 29.9792458 * -5.0));
    QVERIFY(close(toCm1(-5.0, LaserUnit::GHz), -5.0 / 29.9792458));
}

void LifConversionTest::testUnitLabels()
{
    QCOMPARE(unitLabel(LaserUnit::Cm1), QString::fromUtf8("cm⁻¹"));
    QCOMPARE(unitLabel(LaserUnit::Nm), QStringLiteral("nm"));
    QCOMPARE(unitLabel(LaserUnit::GHz), QStringLiteral("GHz"));
    QCOMPARE(unitLabel(LaserUnit::eV), QStringLiteral("eV"));
}

void LifConversionTest::testIdentity()
{
    auto res = LifConversion::assemble({});
    QVERIFY(res.ok);
    QVERIFY(res.conversion.isIdentity());
    QCOMPARE(res.conversion.laserToOutput(12345.6), 12345.6);
    QCOMPARE(res.conversion.outputToLaser(12345.6), 12345.6);
    QVERIFY(res.conversion.stageInput(QStringLiteral("anything"), 100.0) < 0.0);

    auto range = res.conversion.outputRange(100.0, 200.0);
    QCOMPARE(range.first, 100.0);
    QCOMPARE(range.second, 200.0);

    // Default construction matches the assembled identity.
    LifConversion def;
    QVERIFY(def.isIdentity());
    QCOMPARE(def.laserToOutput(42.0), 42.0);
    QCOMPARE(def.outputToLaser(42.0), 42.0);
}

void LifConversionTest::testNhgRoundTrip()
{
    Node doubler;
    doubler.stageKey = QStringLiteral("doubler");
    doubler.op = Op::NHG;
    doubler.n = 3;
    doubler.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    doubler.isFinal = true;

    auto res = LifConversion::assemble({doubler});
    QVERIFY2(res.ok, qPrintable(res.errorString));
    QVERIFY(!res.conversion.isIdentity());

    QCOMPARE(res.conversion.laserToOutput(100.0), 300.0);
    QCOMPARE(res.conversion.outputToLaser(300.0), 100.0);
    QCOMPARE(res.conversion.stageInput(QStringLiteral("doubler"), 100.0), 100.0);
    QVERIFY(res.conversion.stageInput(QStringLiteral("nonexistent"), 100.0) < 0.0);
}

void LifConversionTest::testSfgRoundTrip()
{
    Node sfg;
    sfg.stageKey = QStringLiteral("sfg");
    sfg.op = Op::SFG;
    sfg.inputs = {InputRef{RefType::Laser, {}, 0.0}, InputRef{RefType::Fixed, {}, 500.0}};
    sfg.isFinal = true;

    auto res = LifConversion::assemble({sfg});
    QVERIFY2(res.ok, qPrintable(res.errorString));

    QCOMPARE(res.conversion.laserToOutput(1000.0), 1500.0);
    QCOMPARE(res.conversion.outputToLaser(1500.0), 1000.0);
    QCOMPARE(res.conversion.stageInput(QStringLiteral("sfg"), 1000.0), 1000.0);
}

void LifConversionTest::testDfgRoundTrip()
{
    Node dfg;
    dfg.stageKey = QStringLiteral("dfg");
    dfg.op = Op::DFG;
    dfg.inputs = {InputRef{RefType::Laser, {}, 0.0}, InputRef{RefType::Fixed, {}, 200.0}};
    dfg.isFinal = true;

    auto res = LifConversion::assemble({dfg});
    QVERIFY2(res.ok, qPrintable(res.errorString));

    QCOMPARE(res.conversion.laserToOutput(1000.0), 800.0);
    QCOMPARE(res.conversion.outputToLaser(800.0), 1000.0);
    QCOMPARE(res.conversion.stageInput(QStringLiteral("dfg"), 1000.0), 1000.0);
}

void LifConversionTest::testOutputRange()
{
    Node doubler;
    doubler.stageKey = QStringLiteral("doubler");
    doubler.op = Op::NHG;
    doubler.n = 2;
    doubler.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    doubler.isFinal = true;

    auto res = LifConversion::assemble({doubler});
    QVERIFY2(res.ok, qPrintable(res.errorString));

    auto range = res.conversion.outputRange(100.0, 200.0);
    QCOMPARE(range.first, 200.0);
    QCOMPARE(range.second, 400.0);
    QVERIFY(range.first <= range.second);
}

void LifConversionTest::testOutputRangeInversion()
{
    // DFG(Fixed, Laser): output = fixed - fundamental, a direction-reversing
    // (negative-slope) topology. A larger fundamental yields a SMALLER
    // output, so outputRange must still return its bounds sorted ascending.
    Node dfg;
    dfg.stageKey = QStringLiteral("invert");
    dfg.op = Op::DFG;
    dfg.inputs = {InputRef{RefType::Fixed, {}, 1000.0}, InputRef{RefType::Laser, {}, 0.0}};
    dfg.isFinal = true;

    auto res = LifConversion::assemble({dfg});
    QVERIFY2(res.ok, qPrintable(res.errorString));

    QCOMPARE(res.conversion.laserToOutput(100.0), 900.0);
    QCOMPARE(res.conversion.laserToOutput(200.0), 800.0);

    auto range = res.conversion.outputRange(100.0, 200.0);
    QVERIFY(range.first <= range.second);
    QCOMPARE(range.first, 800.0);
    QCOMPARE(range.second, 900.0);
}

void LifConversionTest::testTriplerWithLaserSecondInput()
{
    // NHG(n=2) doubler feeding an SFG whose second input references the
    // fundamental directly: output = 2f + f = 3f.
    Node doubler;
    doubler.stageKey = QStringLiteral("doubler");
    doubler.op = Op::NHG;
    doubler.n = 2;
    doubler.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    doubler.isFinal = false;

    Node tripler;
    tripler.stageKey = QStringLiteral("tripler");
    tripler.op = Op::SFG;
    tripler.inputs = {InputRef{RefType::Stage, QStringLiteral("doubler"), 0.0},
                       InputRef{RefType::Laser, {}, 0.0}};
    tripler.isFinal = true;

    auto res = LifConversion::assemble({doubler, tripler});
    QVERIFY2(res.ok, qPrintable(res.errorString));

    QCOMPARE(res.conversion.laserToOutput(100.0), 300.0);
    QCOMPARE(res.conversion.outputToLaser(300.0), 100.0);
    QCOMPARE(res.conversion.stageInput(QStringLiteral("doubler"), 100.0), 100.0);
    QCOMPARE(res.conversion.stageInput(QStringLiteral("tripler"), 100.0), 200.0);
}

void LifConversionTest::testTriplerWithFixedSecondInput()
{
    // Same doubler, but the SFG's second input is a Fixed mixing beam
    // instead of the fundamental: output = 2f + fixed.
    Node doubler;
    doubler.stageKey = QStringLiteral("doubler");
    doubler.op = Op::NHG;
    doubler.n = 2;
    doubler.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    doubler.isFinal = false;

    Node sfg;
    sfg.stageKey = QStringLiteral("sfg");
    sfg.op = Op::SFG;
    sfg.inputs = {InputRef{RefType::Stage, QStringLiteral("doubler"), 0.0},
                  InputRef{RefType::Fixed, {}, 50.0}};
    sfg.isFinal = true;

    auto res = LifConversion::assemble({doubler, sfg});
    QVERIFY2(res.ok, qPrintable(res.errorString));

    QCOMPARE(res.conversion.laserToOutput(100.0), 250.0);
    QCOMPARE(res.conversion.outputToLaser(250.0), 100.0);
    QCOMPARE(res.conversion.stageInput(QStringLiteral("sfg"), 100.0), 200.0);
}

void LifConversionTest::testStageOutput()
{
    // Each node's OUTPUT beam is its own conversion applied to its inputs.
    // For the FINAL node stageOutput must equal laserToOutput; for an
    // intermediate node it is that beam's absolute wavenumber, which the
    // topology snapshot records as per-node affine coefficients.
    Node doubler;
    doubler.stageKey = QStringLiteral("doubler");
    doubler.op = Op::NHG;
    doubler.n = 2;
    doubler.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    doubler.isFinal = false;

    Node tripler;
    tripler.stageKey = QStringLiteral("tripler");
    tripler.op = Op::SFG;
    tripler.inputs = {InputRef{RefType::Stage, QStringLiteral("doubler"), 0.0},
                       InputRef{RefType::Laser, {}, 0.0}};
    tripler.isFinal = true;

    auto res = LifConversion::assemble({doubler, tripler});
    QVERIFY2(res.ok, qPrintable(res.errorString));
    const auto &c = res.conversion;

    // doubler output = 2f; tripler output = 2f + f = 3f (== FINAL).
    QCOMPARE(c.stageOutput(QStringLiteral("doubler"), 100.0), 200.0);
    QCOMPARE(c.stageOutput(QStringLiteral("tripler"), 100.0), 300.0);
    QCOMPARE(c.stageOutput(QStringLiteral("tripler"), 100.0), c.laserToOutput(100.0));

    // Slope/intercept recovery (as the topology writer performs it).
    const double b = c.stageOutput(QStringLiteral("doubler"), 0.0);
    const double a = c.stageOutput(QStringLiteral("doubler"), 1.0) - b;
    QVERIFY(close(a, 2.0));
    QVERIFY(close(b, 0.0));

    // Unknown stage key yields a negative sentinel.
    QVERIFY(c.stageOutput(QStringLiteral("nonexistent"), 100.0) < 0.0);
}

void LifConversionTest::testRejectUnresolvedStageRef()
{
    Node a;
    a.stageKey = QStringLiteral("a");
    a.op = Op::NHG;
    a.n = 2;
    a.inputs = {InputRef{RefType::Stage, QStringLiteral("missing"), 0.0}};
    a.isFinal = true;

    auto res = LifConversion::assemble({a});
    QVERIFY(!res.ok);
    QVERIFY(!res.errorString.isEmpty());
}

void LifConversionTest::testRejectWrongArity()
{
    // SFG requires exactly 2 inputs; give it 1.
    Node a;
    a.stageKey = QStringLiteral("a");
    a.op = Op::SFG;
    a.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    a.isFinal = true;

    auto res = LifConversion::assemble({a});
    QVERIFY(!res.ok);
    QVERIFY(!res.errorString.isEmpty());

    // NHG requires exactly 1 input; give it 2.
    Node b;
    b.stageKey = QStringLiteral("b");
    b.op = Op::NHG;
    b.n = 2;
    b.inputs = {InputRef{RefType::Laser, {}, 0.0}, InputRef{RefType::Fixed, {}, 1.0}};
    b.isFinal = true;

    auto res2 = LifConversion::assemble({b});
    QVERIFY(!res2.ok);
    QVERIFY(!res2.errorString.isEmpty());
}

void LifConversionTest::testRejectNhgOrderBelowOne()
{
    Node a;
    a.stageKey = QStringLiteral("a");
    a.op = Op::NHG;
    a.n = 0;
    a.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    a.isFinal = true;

    auto res = LifConversion::assemble({a});
    QVERIFY(!res.ok);
    QVERIFY(!res.errorString.isEmpty());
}

void LifConversionTest::testRejectNoFinal()
{
    Node a;
    a.stageKey = QStringLiteral("a");
    a.op = Op::NHG;
    a.n = 2;
    a.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    a.isFinal = false;

    auto res = LifConversion::assemble({a});
    QVERIFY(!res.ok);
    QVERIFY(!res.errorString.isEmpty());
}

void LifConversionTest::testRejectMultipleFinal()
{
    Node a;
    a.stageKey = QStringLiteral("a");
    a.op = Op::NHG;
    a.n = 2;
    a.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    a.isFinal = true;

    Node b;
    b.stageKey = QStringLiteral("b");
    b.op = Op::NHG;
    b.n = 3;
    b.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    b.isFinal = true;

    auto res = LifConversion::assemble({a, b});
    QVERIFY(!res.ok);
    QVERIFY(!res.errorString.isEmpty());
}

void LifConversionTest::testRejectCycle()
{
    Node a;
    a.stageKey = QStringLiteral("a");
    a.op = Op::NHG;
    a.n = 2;
    a.inputs = {InputRef{RefType::Stage, QStringLiteral("b"), 0.0}};
    a.isFinal = true;

    Node b;
    b.stageKey = QStringLiteral("b");
    b.op = Op::NHG;
    b.n = 2;
    b.inputs = {InputRef{RefType::Stage, QStringLiteral("a"), 0.0}};
    b.isFinal = false;

    auto res = LifConversion::assemble({a, b});
    QVERIFY(!res.ok);
    QVERIFY(!res.errorString.isEmpty());
}

void LifConversionTest::testRejectDuplicateStageKey()
{
    Node a;
    a.stageKey = QStringLiteral("dup");
    a.op = Op::NHG;
    a.n = 2;
    a.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    a.isFinal = true;

    Node b;
    b.stageKey = QStringLiteral("dup");
    b.op = Op::NHG;
    b.n = 3;
    b.inputs = {InputRef{RefType::Laser, {}, 0.0}};
    b.isFinal = false;

    auto res = LifConversion::assemble({a, b});
    QVERIFY(!res.ok);
    QVERIFY(!res.errorString.isEmpty());
}

void LifConversionTest::testRejectNoNetTunableDependence()
{
    // DFG(Laser, Laser): output = f - f = 0, i.e. no net dependence on the
    // tunable source. This is the currently-representable proxy for the
    // deferred "more than one tunable source" scope boundary (plan §1):
    // the schema cannot yet name a second, independent tunable source, but
    // it can produce a FINAL beam whose coefficient on the one fundamental
    // is zero, which is equally non-invertible and must be rejected.
    Node cancel;
    cancel.stageKey = QStringLiteral("cancel");
    cancel.op = Op::DFG;
    cancel.inputs = {InputRef{RefType::Laser, {}, 0.0}, InputRef{RefType::Laser, {}, 0.0}};
    cancel.isFinal = true;

    auto res = LifConversion::assemble({cancel});
    QVERIFY(!res.ok);
    QVERIFY(!res.errorString.isEmpty());
}

QTEST_APPLESS_MAIN(LifConversionTest)

#include "tst_lifconversion.moc"
