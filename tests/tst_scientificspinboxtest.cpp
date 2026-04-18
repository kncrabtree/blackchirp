#include <QtTest>
#include <QApplication>
#include <QClipboard>
#include <QLineEdit>
#include <QSignalSpy>
#include <QWheelEvent>
#include <limits>

#include <src/gui/widget/scientificspinbox.h>

using namespace Qt::Literals::StringLiterals;

// Multiplication sign as used by applySuperscript
static const QString MUL = u" \u00d7 10^"_s;
static const QString SUFFIX = u" MHz"_s;

class ScientificSpinBoxTest : public QObject
{
    Q_OBJECT

private slots:
    void testInitialState();

    // Auto-mode display: fixed range [1e-6, 1e6)
    void testAutoModeFixed_data();
    void testAutoModeFixed();

    // Auto-mode display: scientific outside [1e-6, 1e6)
    void testAutoModeScientific_data();
    void testAutoModeScientific();

    // Forced display modes
    void testDisplayModeFixed_data();
    void testDisplayModeFixed();
    void testDisplayModeScientific_data();
    void testDisplayModeScientific();

    // Explicit precision override
    void testExplicitPrecision_data();
    void testExplicitPrecision();
    void testPrecisionClamp();
    void testPrecisionAutoResetOnSetValue();

    // Suffix
    void testSuffixDisplay_data();
    void testSuffixDisplay();
    void testSuffixValueParsing();
    void testSuffixValidation();

    // Stepping
    void testStepFromZero();
    void testStepSize_data();
    void testStepSize();
    void testStepReversibility_data();
    void testStepReversibility();
    void testCtrlStep();
    void testCtrlStepFromZero();

    // Fixed step mode
    void testStepModeDefault();
    void testFixedStepSize_data();
    void testFixedStepSize();
    void testFixedStepCtrl();
    void testFixedStepFromZero();
    void testFixedStepReversibility_data();
    void testFixedStepReversibility();
    void testStepModePreserved();
    void testFixedStepSizeAbs();
    void testFixedStepZeroFallback();
    void testFixedStepRangeClamp();

    // Range
    void testRangeClamp();
    void testRangeStep();

    // Signals
    void testValueChangedSignal();

    // Copy-value format
    void testCopyValueFormat_data();
    void testCopyValueFormat();
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

static QString sci(const QString &mantissa, const QString &exp)
{
    // e.g. sci("1.6", "6") → "1.6 × 10^6"
    //      sci("5",   "(-8)") → "5 × 10^(-8)"
    return mantissa + MUL + exp;
}

// ─── Initial state ────────────────────────────────────────────────────────────

void ScientificSpinBoxTest::testInitialState()
{
    ScientificSpinBox sb;
    QCOMPARE(sb.value(), 0.0);
    QCOMPARE(sb.displayMode(), ScientificSpinBox::DisplayMode::Auto);
    QCOMPARE(sb.displayPrecision(), -1);
    QCOMPARE(sb.suffix(), QString());
    QCOMPARE(sb.lineEdit()->text(), u"0"_s);
    QVERIFY(sb.minimum() < -1e300);
    QVERIFY(sb.maximum() > 1e300);
}

// ─── Auto-mode fixed ──────────────────────────────────────────────────────────

void ScientificSpinBoxTest::testAutoModeFixed_data()
{
    QTest::addColumn<double>("value");
    QTest::addColumn<bool>("hasSuffix");
    QTest::addColumn<QString>("expected");

    auto add = [](const char *tag, double v, bool s, const QString &e) {
        QTest::newRow(tag) << v << s << e;
    };

    add("zero",               0.0,     false, u"0"_s);
    add("zero+suffix",        0.0,     true,  u"0"_s + SUFFIX);
    add("one",                1.0,     false, u"1"_s);
    add("one+suffix",         1.0,     true,  u"1"_s + SUFFIX);
    add("1.5",                1.5,     false, u"1.5"_s);
    add("1.5+suffix",         1.5,     true,  u"1.5"_s + SUFFIX);
    add("1.23",               1.23,    false, u"1.23"_s);
    add("1.23+suffix",        1.23,    true,  u"1.23"_s + SUFFIX);
    add("0.5",                0.5,     false, u"0.5"_s);
    add("0.5+suffix",         0.5,     true,  u"0.5"_s + SUFFIX);
    add("0.00123",            0.00123, false, u"0.00123"_s);
    add("0.00123+suffix",     0.00123, true,  u"0.00123"_s + SUFFIX);
    add("1000",               1000.0,  false, u"1000"_s);
    add("1234.5",             1234.5,  false, u"1234.5"_s);
    add("1234.5+suffix",      1234.5,  true,  u"1234.5"_s + SUFFIX);
    add("999999",             999999.0,false, u"999999"_s);
    add("1e-6",               1e-6,    false, u"0.000001"_s);
    add("1e-6+suffix",        1e-6,    true,  u"0.000001"_s + SUFFIX);
    add("-1.5",              -1.5,     false, u"-1.5"_s);
    add("-1.5+suffix",       -1.5,     true,  u"-1.5"_s + SUFFIX);
}

void ScientificSpinBoxTest::testAutoModeFixed()
{
    QFETCH(double, value);
    QFETCH(bool, hasSuffix);
    QFETCH(QString, expected);

    ScientificSpinBox sb;
    if (hasSuffix) sb.setSuffix(SUFFIX);
    sb.setValue(value);
    QCOMPARE(sb.lineEdit()->text(), expected);
}

// ─── Auto-mode scientific ─────────────────────────────────────────────────────

void ScientificSpinBoxTest::testAutoModeScientific_data()
{
    QTest::addColumn<double>("value");
    QTest::addColumn<bool>("hasSuffix");
    QTest::addColumn<QString>("expected");

    auto add = [](const char *tag, double v, bool s, const QString &e) {
        QTest::newRow(tag) << v << s << e;
    };

    add("1.6e6",          1.6e6,  false, sci(u"1.6"_s,  u"6"_s));
    add("1.6e6+suffix",   1.6e6,  true,  sci(u"1.6"_s,  u"6"_s) + SUFFIX);
    add("2e6",            2e6,    false, sci(u"2"_s,    u"6"_s));
    add("2e6+suffix",     2e6,    true,  sci(u"2"_s,    u"6"_s) + SUFFIX);
    add("1e6",            1e6,    false, sci(u"1"_s,    u"6"_s));
    add("1.23e10",        1.23e10,false, sci(u"1.23"_s, u"10"_s));
    add("5e-8",           5e-8,   false, sci(u"5"_s,    u"(-8)"_s));
    add("5e-8+suffix",    5e-8,   true,  sci(u"5"_s,    u"(-8)"_s) + SUFFIX);
    add("9.9e-7",         9.9e-7, false, sci(u"9.9"_s,  u"(-7)"_s));
    add("-1.5e9",        -1.5e9,  false, sci(u"-1.5"_s, u"9"_s));
    add("-1.5e9+suffix", -1.5e9,  true,  sci(u"-1.5"_s, u"9"_s) + SUFFIX);
}

void ScientificSpinBoxTest::testAutoModeScientific()
{
    QFETCH(double, value);
    QFETCH(bool, hasSuffix);
    QFETCH(QString, expected);

    ScientificSpinBox sb;
    if (hasSuffix) sb.setSuffix(SUFFIX);
    sb.setValue(value);
    QCOMPARE(sb.lineEdit()->text(), expected);
}

// ─── Forced Fixed mode ────────────────────────────────────────────────────────

void ScientificSpinBoxTest::testDisplayModeFixed_data()
{
    QTest::addColumn<double>("value");
    QTest::addColumn<bool>("hasSuffix");
    QTest::addColumn<QString>("expected");

    auto add = [](const char *tag, double v, bool s, const QString &e) {
        QTest::newRow(tag) << v << s << e;
    };

    // Values outside auto-fixed range, forced to fixed
    add("1.6e6",          1.6e6,   false, u"1600000"_s);
    add("1.6e6+suffix",   1.6e6,   true,  u"1600000"_s + SUFFIX);
    add("1.23e-4",        1.23e-4, false, u"0.000123"_s);
    add("1.23e-4+suffix", 1.23e-4, true,  u"0.000123"_s + SUFFIX);
    add("5e-8",           5e-8,    false, u"0.00000005"_s);
    // Values that are already in auto-fixed range stay the same
    add("1.5 in range",   1.5,     false, u"1.5"_s);
    add("1.5+suffix",     1.5,     true,  u"1.5"_s + SUFFIX);
}

void ScientificSpinBoxTest::testDisplayModeFixed()
{
    QFETCH(double, value);
    QFETCH(bool, hasSuffix);
    QFETCH(QString, expected);

    ScientificSpinBox sb;
    sb.setDisplayMode(ScientificSpinBox::DisplayMode::Fixed);
    if (hasSuffix) sb.setSuffix(SUFFIX);
    sb.setValue(value);
    QCOMPARE(sb.lineEdit()->text(), expected);
}

// ─── Forced Scientific mode ───────────────────────────────────────────────────

void ScientificSpinBoxTest::testDisplayModeScientific_data()
{
    QTest::addColumn<double>("value");
    QTest::addColumn<bool>("hasSuffix");
    QTest::addColumn<QString>("expected");

    auto add = [](const char *tag, double v, bool s, const QString &e) {
        QTest::newRow(tag) << v << s << e;
    };

    // Values that are in auto-fixed range, forced to scientific
    add("0.5",          0.5,    false, sci(u"5"_s,      u"(-1)"_s));
    add("0.5+suffix",   0.5,    true,  sci(u"5"_s,      u"(-1)"_s) + SUFFIX);
    add("1.5",          1.5,    false, sci(u"1.5"_s,    u"0"_s));
    add("1.5+suffix",   1.5,    true,  sci(u"1.5"_s,    u"0"_s) + SUFFIX);
    add("1000",         1000.0, false, sci(u"1"_s,      u"3"_s));
    add("1234.5",       1234.5, false, sci(u"1.2345"_s, u"3"_s));
    // Values already in scientific range stay the same
    add("1.6e6",        1.6e6,  false, sci(u"1.6"_s,    u"6"_s));
}

void ScientificSpinBoxTest::testDisplayModeScientific()
{
    QFETCH(double, value);
    QFETCH(bool, hasSuffix);
    QFETCH(QString, expected);

    ScientificSpinBox sb;
    sb.setDisplayMode(ScientificSpinBox::DisplayMode::Scientific);
    if (hasSuffix) sb.setSuffix(SUFFIX);
    sb.setValue(value);
    QCOMPARE(sb.lineEdit()->text(), expected);
}

// ─── Explicit precision ───────────────────────────────────────────────────────

void ScientificSpinBoxTest::testExplicitPrecision_data()
{
    QTest::addColumn<double>("value");
    QTest::addColumn<int>("precision");
    QTest::addColumn<bool>("hasSuffix");
    QTest::addColumn<QString>("expected");

    auto add = [](const char *tag, double v, int p, bool s, const QString &e) {
        QTest::newRow(tag) << v << p << s << e;
    };

    // Fixed range: precision sets decimal places
    add("1.5 p3",          1.5,   3, false, u"1.500"_s);
    add("1.5 p3+suffix",   1.5,   3, true,  u"1.500"_s + SUFFIX);
    add("1.5 p0",          1.5,   0, false, u"2"_s); // rounds
    add("1000 p2",         1000.0,2, false, u"1000.00"_s);

    // Scientific range: precision sets mantissa decimal places
    add("1.6e6 p3",        1.6e6, 3, false, sci(u"1.600"_s, u"6"_s));
    add("1.6e6 p3+suffix", 1.6e6, 3, true,  sci(u"1.600"_s, u"6"_s) + SUFFIX);
    add("1.6e6 p0",        1.6e6, 0, false, sci(u"2"_s,     u"6"_s)); // rounds
    add("5e-8 p2",         5e-8,  2, false, sci(u"5.00"_s,  u"(-8)"_s));
}

void ScientificSpinBoxTest::testExplicitPrecision()
{
    QFETCH(double, value);
    QFETCH(int, precision);
    QFETCH(bool, hasSuffix);
    QFETCH(QString, expected);

    ScientificSpinBox sb;
    sb.setDisplayPrecision(precision);
    if (hasSuffix) sb.setSuffix(SUFFIX);
    sb.setValue(value);
    QCOMPARE(sb.lineEdit()->text(), expected);
}

void ScientificSpinBoxTest::testPrecisionClamp()
{
    ScientificSpinBox sb;
    sb.setDisplayPrecision(-5);
    QCOMPARE(sb.displayPrecision(), -1); // clamped to -1

    sb.setDisplayPrecision(20);
    QCOMPARE(sb.displayPrecision(), 15); // clamped to MAX_PRECISION

    sb.setDisplayPrecision(7);
    QCOMPARE(sb.displayPrecision(), 7);

    sb.setDisplayPrecision(-1);
    QCOMPARE(sb.displayPrecision(), -1); // auto
}

void ScientificSpinBoxTest::testPrecisionAutoResetOnSetValue()
{
    // Programmatic setValue resets precision to auto, re-detecting from the new value
    ScientificSpinBox sb;
    sb.setValue(1.5);
    QCOMPARE(sb.lineEdit()->text(), u"1.5"_s);

    sb.setValue(1.23);
    QCOMPARE(sb.lineEdit()->text(), u"1.23"_s);

    sb.setValue(1.6e6);
    QCOMPARE(sb.lineEdit()->text(), sci(u"1.6"_s, u"6"_s));
}

// ─── Suffix ───────────────────────────────────────────────────────────────────

void ScientificSpinBoxTest::testSuffixDisplay_data()
{
    QTest::addColumn<double>("value");
    QTest::addColumn<QString>("suffix");
    QTest::addColumn<QString>("expected");

    QTest::newRow("MHz") << 1.5  << u" MHz"_s << u"1.5 MHz"_s;
    QTest::newRow("ns")  << 10.0 << u" ns"_s  << u"10 ns"_s;
    QTest::newRow("empty suffix") << 1.5 << QString() << u"1.5"_s;
}

void ScientificSpinBoxTest::testSuffixDisplay()
{
    QFETCH(double, value);
    QFETCH(QString, suffix);
    QFETCH(QString, expected);

    ScientificSpinBox sb;
    sb.setSuffix(suffix);
    sb.setValue(value);
    QCOMPARE(sb.lineEdit()->text(), expected);
}

void ScientificSpinBoxTest::testSuffixValueParsing()
{
    ScientificSpinBox sb;
    sb.setSuffix(SUFFIX);
    sb.setValue(1.5);

    // valueFromText strips the suffix correctly
    QCOMPARE(sb.valueFromText(u"1.5 MHz"_s), 1.5);
    QCOMPARE(sb.valueFromText(u"1.5e6 MHz"_s), 1.5e6);
    QCOMPARE(sb.valueFromText(u"1.5"_s), 1.5); // also works without suffix
    QCOMPARE(sb.valueFromText(u"0 MHz"_s), 0.0);
}

void ScientificSpinBoxTest::testSuffixValidation()
{
    ScientificSpinBox sb;
    sb.setSuffix(SUFFIX);

    int pos = 0;
    QString valid = u"1.5 MHz"_s;
    QVERIFY(sb.validate(valid, pos) != QValidator::Invalid);

    QString invalid = u"abc MHz"_s;
    QVERIFY(sb.validate(invalid, pos) == QValidator::Invalid);
}

// ─── Stepping ─────────────────────────────────────────────────────────────────

void ScientificSpinBoxTest::testStepFromZero()
{
    ScientificSpinBox sb;
    sb.setValue(0.0);
    sb.stepBy(1);
    QCOMPARE(sb.value(), 1.0);

    sb.setValue(0.0);
    sb.stepBy(-1);
    QCOMPARE(sb.value(), -1.0);
}

void ScientificSpinBoxTest::testStepSize_data()
{
    QTest::addColumn<double>("startValue");
    QTest::addColumn<int>("steps");
    QTest::addColumn<double>("expectedValue");

    // Fixed range: step = 10^(-decimal_places)
    QTest::newRow("1.5 up")    << 1.5    << 1  << 1.6;    // prec=1, step=0.1
    QTest::newRow("1.5 down")  << 1.5    << -1 << 1.4;
    QTest::newRow("1.23 up")   << 1.23   << 1  << 1.24;   // prec=2, step=0.01
    QTest::newRow("0.00123 up")<< 0.00123<< 1  << 0.00124;// prec=5, step=1e-5
    QTest::newRow("1000 up")   << 1000.0 << 1  << 1001.0; // prec=0, step=1
    QTest::newRow("two steps") << 1.5    << 2  << 1.7;    // 2 × step=0.1

    // Scientific range: step = 10^(exponent - precision)
    QTest::newRow("1.6e6 up")  << 1.6e6  << 1  << 1.7e6;  // prec=1, exp=6, step=1e5
    QTest::newRow("1.6e6 down")<< 1.6e6  << -1 << 1.5e6;
    QTest::newRow("5e-8 up")   << 5e-8   << 1  << 6e-8;   // prec=0, exp=-8, step=1e-8
}

void ScientificSpinBoxTest::testStepSize()
{
    QFETCH(double, startValue);
    QFETCH(int, steps);
    QFETCH(double, expectedValue);

    ScientificSpinBox sb;
    sb.setValue(startValue);
    sb.stepBy(steps);
    QCOMPARE(sb.value(), expectedValue);
}

void ScientificSpinBoxTest::testStepReversibility_data()
{
    QTest::addColumn<double>("value");
    QTest::newRow("1.5")    << 1.5;
    QTest::newRow("0.00123")<< 0.00123;
    QTest::newRow("1234.5") << 1234.5;
    QTest::newRow("1.6e6")  << 1.6e6;
    QTest::newRow("5e-8")   << 5e-8;
    QTest::newRow("1.23e10")<< 1.23e10;
}

void ScientificSpinBoxTest::testStepReversibility()
{
    QFETCH(double, value);

    ScientificSpinBox sb;
    sb.setValue(value);
    sb.stepBy(1);
    sb.stepBy(-1);
    QCOMPARE(sb.value(), value);
}

void ScientificSpinBoxTest::testCtrlStep()
{
    ScientificSpinBox sb;
    sb.resize(200, 40);
    sb.show();
    sb.setFocus();
    QApplication::processEvents();

    sb.setValue(2.0);
    const QPointF pos(sb.rect().center());

    QWheelEvent up(pos, sb.mapToGlobal(pos), QPoint(), QPoint(0, 120),
                   Qt::NoButton, Qt::ControlModifier, Qt::NoScrollPhase, false);
    QApplication::sendEvent(&sb, &up);
    QCOMPARE(sb.value(), 4.0);

    QWheelEvent down(pos, sb.mapToGlobal(pos), QPoint(), QPoint(0, -120),
                     Qt::NoButton, Qt::ControlModifier, Qt::NoScrollPhase, false);
    QApplication::sendEvent(&sb, &down);
    QCOMPARE(sb.value(), 2.0);

    // Multiple steps: 3 up → ×8
    sb.setValue(1.0);
    QWheelEvent triple(pos, sb.mapToGlobal(pos), QPoint(), QPoint(0, 360),
                       Qt::NoButton, Qt::ControlModifier, Qt::NoScrollPhase, false);
    QApplication::sendEvent(&sb, &triple);
    QCOMPARE(sb.value(), 8.0);
}

void ScientificSpinBoxTest::testCtrlStepFromZero()
{
    ScientificSpinBox sb;
    sb.resize(200, 40);
    sb.show();
    sb.setFocus();
    QApplication::processEvents();

    const QPointF pos(sb.rect().center());

    sb.setValue(0.0);
    QWheelEvent up(pos, sb.mapToGlobal(pos), QPoint(), QPoint(0, 120),
                   Qt::NoButton, Qt::ControlModifier, Qt::NoScrollPhase, false);
    QApplication::sendEvent(&sb, &up);
    QCOMPARE(sb.value(), 1.0);

    sb.setValue(0.0);
    QWheelEvent down(pos, sb.mapToGlobal(pos), QPoint(), QPoint(0, -120),
                     Qt::NoButton, Qt::ControlModifier, Qt::NoScrollPhase, false);
    QApplication::sendEvent(&sb, &down);
    QCOMPARE(sb.value(), -1.0);
}

// ─── Fixed step mode ──────────────────────────────────────────────────────────

void ScientificSpinBoxTest::testStepModeDefault()
{
    ScientificSpinBox sb;
    QCOMPARE(sb.stepMode(), ScientificSpinBox::StepMode::Adaptive);
    QCOMPARE(sb.fixedStepSize(), 0.0);
}

void ScientificSpinBoxTest::testFixedStepSize_data()
{
    QTest::addColumn<double>("startValue");
    QTest::addColumn<double>("stepSize");
    QTest::addColumn<int>("steps");
    QTest::addColumn<double>("expectedValue");

    // Basic fixed-range cases
    QTest::newRow("1.5 + 1*0.5")          << 1.5      << 0.5   << 1  << 2.0;
    QTest::newRow("1.23 + 1*0.01")        << 1.23     << 0.01  << 1  << 1.24;
    QTest::newRow("cross boundary 1e6")   << 999999.0 << 1.0   << 1  << 1e6;
    QTest::newRow("1.6e6 + 1*1e5")        << 1.6e6    << 1e5   << 1  << 1.7e6;
    QTest::newRow("negative steps")       << 2.0      << 0.5   << -1 << 1.5;
    QTest::newRow("negative steps sci")   << 1.7e6    << 1e5   << -2 << 1.5e6;

    // Floating-point precision cases (Type 1)
    QTest::newRow("5e-8 + 1*1e-8")        << 5e-8     << 1e-8  << 1  << 6e-8;
    QTest::newRow("9.9e-7 + 1*1e-9")      << 9.9e-7   << 1e-9  << 1  << 9.91e-7;
    QTest::newRow("1.23e10 + 1*1e9")      << 1.23e10  << 1e9   << 1  << 1.33e10;
}

void ScientificSpinBoxTest::testFixedStepSize()
{
    QFETCH(double, startValue);
    QFETCH(double, stepSize);
    QFETCH(int, steps);
    QFETCH(double, expectedValue);

    ScientificSpinBox sb;
    sb.setFixedStepSize(stepSize);
    sb.setStepMode(ScientificSpinBox::StepMode::Fixed);
    sb.setValue(startValue);
    sb.stepBy(steps);
    QCOMPARE(sb.value(), expectedValue);
}

void ScientificSpinBoxTest::testFixedStepCtrl()
{
    ScientificSpinBox sb;
    sb.resize(200, 40);
    sb.show();
    sb.setFocus();
    QApplication::processEvents();

    sb.setFixedStepSize(0.5);
    sb.setStepMode(ScientificSpinBox::StepMode::Fixed);
    sb.setValue(1.0);

    const QPointF pos(sb.rect().center());

    // Ctrl+wheel up: 1.0 + 10*0.5 = 6.0
    QWheelEvent up(pos, sb.mapToGlobal(pos), QPoint(), QPoint(0, 120),
                   Qt::NoButton, Qt::ControlModifier, Qt::NoScrollPhase, false);
    QApplication::sendEvent(&sb, &up);
    QCOMPARE(sb.value(), 6.0);

    // Ctrl+wheel down: 6.0 - 10*0.5 = 1.0
    QWheelEvent down(pos, sb.mapToGlobal(pos), QPoint(), QPoint(0, -120),
                     Qt::NoButton, Qt::ControlModifier, Qt::NoScrollPhase, false);
    QApplication::sendEvent(&sb, &down);
    QCOMPARE(sb.value(), 1.0);
}

void ScientificSpinBoxTest::testFixedStepFromZero()
{
    ScientificSpinBox sb;
    sb.setFixedStepSize(0.25);
    sb.setStepMode(ScientificSpinBox::StepMode::Fixed);

    sb.setValue(0.0);
    sb.stepBy(1);
    QCOMPARE(sb.value(), 0.25);

    sb.setValue(0.0);
    sb.stepBy(-1);
    QCOMPARE(sb.value(), -0.25);
}

void ScientificSpinBoxTest::testFixedStepReversibility_data()
{
    QTest::addColumn<double>("value");
    QTest::addColumn<double>("stepSize");

    QTest::newRow("1.5, 0.5")         << 1.5      << 0.5;
    QTest::newRow("1.23, 0.01")       << 1.23     << 0.01;
    QTest::newRow("1234.5, 0.1")      << 1234.5   << 0.1;
    QTest::newRow("1.6e6, 1e5")       << 1.6e6    << 1e5;

    // Type-1 precision-problematic pairs
    QTest::newRow("5e-8, 1e-8")       << 5e-8     << 1e-8;
    QTest::newRow("9.9e-7, 1e-9")     << 9.9e-7   << 1e-9;
    QTest::newRow("1.23e10, 1e9")     << 1.23e10  << 1e9;
}

void ScientificSpinBoxTest::testFixedStepReversibility()
{
    QFETCH(double, value);
    QFETCH(double, stepSize);

    ScientificSpinBox sb;
    sb.setFixedStepSize(stepSize);
    sb.setStepMode(ScientificSpinBox::StepMode::Fixed);
    sb.setValue(value);
    sb.stepBy(1);
    sb.stepBy(-1);
    QCOMPARE(sb.value(), value);
}

void ScientificSpinBoxTest::testStepModePreserved()
{
    ScientificSpinBox sb;
    sb.setFixedStepSize(0.25);
    sb.setStepMode(ScientificSpinBox::StepMode::Fixed);
    sb.setStepMode(ScientificSpinBox::StepMode::Adaptive);
    sb.setStepMode(ScientificSpinBox::StepMode::Fixed);

    // fixedStepSize must survive the round-trip
    QCOMPARE(sb.fixedStepSize(), 0.25);

    sb.setValue(0.0);
    sb.stepBy(1);
    QCOMPARE(sb.value(), 0.25);
}

void ScientificSpinBoxTest::testFixedStepSizeAbs()
{
    ScientificSpinBox sb;
    sb.setFixedStepSize(-0.5);

    // Negative input is stored as absolute value
    QCOMPARE(sb.fixedStepSize(), 0.5);

    sb.setStepMode(ScientificSpinBox::StepMode::Fixed);
    sb.setValue(1.0);
    sb.stepBy(1);
    QCOMPARE(sb.value(), 1.5);
}

void ScientificSpinBoxTest::testFixedStepZeroFallback()
{
    // Fixed mode with d_fixedStepSize == 0 falls through to adaptive behavior.
    // From 1.5 (precision=1, adaptive step=0.1), stepBy(1) → 1.6.
    ScientificSpinBox sb;
    sb.setStepMode(ScientificSpinBox::StepMode::Fixed);
    sb.setValue(1.5);
    sb.stepBy(1);
    QCOMPARE(sb.value(), 1.6);
}

void ScientificSpinBoxTest::testFixedStepRangeClamp()
{
    ScientificSpinBox sb;
    sb.setRange(0.0, 10.0);
    sb.setFixedStepSize(3.0);
    sb.setStepMode(ScientificSpinBox::StepMode::Fixed);
    sb.setValue(9.0);
    sb.stepBy(1); // 9 + 3 = 12 → clamped to 10
    QCOMPARE(sb.value(), 10.0);
}

// ─── Range ────────────────────────────────────────────────────────────────────

void ScientificSpinBoxTest::testRangeClamp()
{
    ScientificSpinBox sb;
    sb.setRange(0.0, 10.0);

    sb.setValue(15.0);
    QCOMPARE(sb.value(), 10.0);

    sb.setValue(-5.0);
    QCOMPARE(sb.value(), 0.0);

    sb.setValue(5.0);
    QCOMPARE(sb.value(), 5.0);
}

void ScientificSpinBoxTest::testRangeStep()
{
    ScientificSpinBox sb;
    sb.setRange(0.0, 2.0);
    sb.setValue(1.9); // prec=1, step=0.1

    sb.stepBy(2); // 1.9 + 2*0.1 = 2.1 → clamped to 2.0
    QCOMPARE(sb.value(), 2.0);

    sb.stepBy(-10); // clamped to 0.0
    QCOMPARE(sb.value(), 0.0);
}

// ─── Signals ──────────────────────────────────────────────────────────────────

void ScientificSpinBoxTest::testValueChangedSignal()
{
    ScientificSpinBox sb;
    QSignalSpy spy(&sb, &ScientificSpinBox::valueChanged);

    sb.setValue(1.5);
    QCOMPARE(spy.count(), 1);
    QCOMPARE(spy.last().at(0).toDouble(), 1.5);

    // Same value: no signal
    sb.setValue(1.5);
    QCOMPARE(spy.count(), 1);

    sb.setValue(2.5);
    QCOMPARE(spy.count(), 2);
    QCOMPARE(spy.last().at(0).toDouble(), 2.5);
}

// ─── Copy-value format ────────────────────────────────────────────────────────

void ScientificSpinBoxTest::testCopyValueFormat_data()
{
    QTest::addColumn<double>("value");
    QTest::addColumn<bool>("hasSuffix");

    auto add = [](const char *tag, double v, bool s) {
        QTest::newRow(tag) << v << s;
    };

    add("1.6e6",          1.6e6,       false);
    add("1.6e6+suffix",   1.6e6,       true);
    add("1.23456789",     1.23456789,  false);
    add("1e-10",          1e-10,       false);
    add("0.0",            0.0,         false);
    add("-1234.5",       -1234.5,      false);
    add("-1234.5+suffix",-1234.5,      true);
}

void ScientificSpinBoxTest::testCopyValueFormat()
{
    QFETCH(double, value);
    QFETCH(bool, hasSuffix);

    ScientificSpinBox sb;
    if (hasSuffix) sb.setSuffix(SUFFIX);
    sb.setValue(value);

    // Verify the copy-value text format (as produced by the Copy Value action)
    const QString copyText = QString::number(sb.value(), 'g', 15);

    // The text must be parseable back to the original value
    bool ok = false;
    const double roundTrip = copyText.toDouble(&ok);
    QVERIFY(ok);
    QCOMPARE(roundTrip, value);

    // The text must not contain the suffix
    if (hasSuffix)
        QVERIFY(!copyText.contains(SUFFIX));

    // Verify via valueFromText round-trip (suffix stripped correctly)
    if (hasSuffix)
        QCOMPARE(sb.valueFromText(copyText + SUFFIX), value);
    else
        QCOMPARE(sb.valueFromText(copyText), value);
}

QTEST_MAIN(ScientificSpinBoxTest)
#include "tst_scientificspinboxtest.moc"
