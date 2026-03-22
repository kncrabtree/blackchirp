#include <QtTest>
#include <QCoreApplication>

#include "src/data/experiment/hardwaredatacontainer.h"
#include "src/data/experiment/experiment.h"

using namespace BC::Data;

class ExperimentLoadingTest : public QObject
{
    Q_OBJECT

public:
    ExperimentLoadingTest() = default;
    ~ExperimentLoadingTest() = default;

private slots:
    void initTestCase();

    // HardwareDataContainer::loadFromFile tests
    void loadHardwareOldestFormat();
    void loadHardwareLegacyFormat();
    void loadHardwareNewFormat();
    void loadHardwareNonExistentFile();

    // HardwareDataContainer round-trip
    void hardwareSaveLoadRoundTrip();

    // Full experiment loading - oldest format (exp 200)
    void loadExperiment200_hardware();
    void loadExperiment200_header();

    // Full experiment loading - devel format (exp 2638)
    void loadExperiment2638_hardware();
    void loadExperiment2638_header();

    // Full experiment loading - new format (exp 27)
    void loadExperiment27_hardware();
    void loadExperiment27_header();

private:
    QString testDataDir() const;
};

void ExperimentLoadingTest::initTestCase()
{
    QCoreApplication::setOrganizationDomain("crabtreelab.ucdavis.edu");
    QCoreApplication::setApplicationName("BlackchirpTest");
}

QString ExperimentLoadingTest::testDataDir() const
{
    return QString(TESTDATA_DIR);
}

// ============================================================
// HardwareDataContainer::loadFromFile
// ============================================================

void ExperimentLoadingTest::loadHardwareOldestFormat()
{
    // Exp 200: oldest format — 2-column, no index in keys
    // "AWG;awg70002a", "Clock0;valon5009", "FtmwDigitizer;dsa71604c", etc.
    auto container = HardwareDataContainer::loadFromFile(
        testDataDir() + "/200/hardware.csv");

    QVERIFY(container.hasAnyHardware());

    // Should have 5 entries
    QCOMPARE(container.hardwareMap.size(), 5);

    // "AWG" key (no dot separator) — legacy 2-column parsing extracts type from key
    QVERIFY(container.hardwareMap.contains("AWG"));
    QCOMPARE(container.hardwareMap["AWG"].implementation, QString("awg70002a"));
    QCOMPARE(container.hardwareMap["AWG"].type, HardwareType::AWG);

    // "FtmwDigitizer" — legacy name, should map to FtmwScope type
    QVERIFY(container.hardwareMap.contains("FtmwDigitizer"));
    QCOMPARE(container.hardwareMap["FtmwDigitizer"].implementation, QString("dsa71604c"));
    QCOMPARE(container.hardwareMap["FtmwDigitizer"].type, HardwareType::FtmwScope);

    // "Clock0" — no dot separator, type extraction gets "Clock0"
    // which won't match legacyStringToHardwareType → Unknown
    QVERIFY(container.hardwareMap.contains("Clock0"));
    QCOMPARE(container.hardwareMap["Clock0"].implementation, QString("valon5009"));

    QVERIFY(container.hardwareMap.contains("PulseGenerator"));
    QCOMPARE(container.hardwareMap["PulseGenerator"].implementation, QString("qc9528"));
    QCOMPARE(container.hardwareMap["PulseGenerator"].type, HardwareType::PulseGenerator);
}

void ExperimentLoadingTest::loadHardwareLegacyFormat()
{
    // Exp 2638: devel format — 2-column, index-based keys
    // "AWG.0;awg70002a", "Clock.0;valon5009", etc.
    auto container = HardwareDataContainer::loadFromFile(
        testDataDir() + "/2638/hardware.csv");

    QVERIFY(container.hasAnyHardware());
    QCOMPARE(container.hardwareMap.size(), 5);

    // Index-based keys preserved as-is
    QVERIFY(container.hardwareMap.contains("AWG.0"));
    QCOMPARE(container.hardwareMap["AWG.0"].implementation, QString("awg70002a"));
    QCOMPARE(container.hardwareMap["AWG.0"].type, HardwareType::AWG);

    QVERIFY(container.hardwareMap.contains("FtmwDigitizer.0"));
    QCOMPARE(container.hardwareMap["FtmwDigitizer.0"].implementation, QString("dsa71604c"));
    QCOMPARE(container.hardwareMap["FtmwDigitizer.0"].type, HardwareType::FtmwScope);

    QVERIFY(container.hardwareMap.contains("Clock.0"));
    QCOMPARE(container.hardwareMap["Clock.0"].implementation, QString("valon5009"));
    QCOMPARE(container.hardwareMap["Clock.0"].type, HardwareType::Clock);

    // Two pulse generators
    QVERIFY(container.hardwareMap.contains("PulseGenerator.0"));
    QCOMPARE(container.hardwareMap["PulseGenerator.0"].type, HardwareType::PulseGenerator);
    QVERIFY(container.hardwareMap.contains("PulseGenerator.1"));
    QCOMPARE(container.hardwareMap["PulseGenerator.1"].implementation, QString("dg645"));
}

void ExperimentLoadingTest::loadHardwareNewFormat()
{
    // Exp 27: new format — 3-column, label-based keys
    // "LifLaser.default;VirtualLifLaser;11", etc.
    auto container = HardwareDataContainer::loadFromFile(
        testDataDir() + "/27/hardware.csv");

    QVERIFY(container.hasAnyHardware());
    QCOMPARE(container.hardwareMap.size(), 5);

    QVERIFY(container.hardwareMap.contains("LifLaser.default"));
    QCOMPARE(container.hardwareMap["LifLaser.default"].implementation, QString("VirtualLifLaser"));
    QCOMPARE(container.hardwareMap["LifLaser.default"].type, HardwareType::LifLaser);

    QVERIFY(container.hardwareMap.contains("FtmwScope.default"));
    QCOMPARE(container.hardwareMap["FtmwScope.default"].implementation, QString("VirtualFtmwScope"));
    QCOMPARE(container.hardwareMap["FtmwScope.default"].type, HardwareType::FtmwScope);

    QVERIFY(container.hardwareMap.contains("PulseGenerator.default"));
    QCOMPARE(container.hardwareMap["PulseGenerator.default"].implementation, QString("VirtualPulseGenerator"));
    QCOMPARE(container.hardwareMap["PulseGenerator.default"].type, HardwareType::PulseGenerator);

    QVERIFY(container.hardwareMap.contains("Clock.default"));
    QCOMPARE(container.hardwareMap["Clock.default"].implementation, QString("FixedClock"));
    QCOMPARE(container.hardwareMap["Clock.default"].type, HardwareType::Clock);

    QVERIFY(container.hardwareMap.contains("LifScope.default"));
    QCOMPARE(container.hardwareMap["LifScope.default"].implementation, QString("VirtualLifScope"));
    QCOMPARE(container.hardwareMap["LifScope.default"].type, HardwareType::LifScope);
}

void ExperimentLoadingTest::loadHardwareNonExistentFile()
{
    auto container = HardwareDataContainer::loadFromFile("/nonexistent/path/hardware.csv");
    QVERIFY(!container.hasAnyHardware());
    QCOMPARE(container.hardwareMap.size(), 0);
}

void ExperimentLoadingTest::hardwareSaveLoadRoundTrip()
{
    // Create a container with known data
    HardwareDataContainer original;
    original.hardwareMap["FtmwScope.main"] = HardwareDataContainer::HardwareEntry("VirtualFtmwScope", HardwareType::FtmwScope);
    original.hardwareMap["Clock.reference"] = HardwareDataContainer::HardwareEntry("FixedClock", HardwareType::Clock);
    original.hardwareMap["PulseGenerator.primary"] = HardwareDataContainer::HardwareEntry("VirtualPulseGenerator", HardwareType::PulseGenerator);

    // Save to temp file
    QTemporaryFile tmp;
    QVERIFY(tmp.open());
    QString tmpPath = tmp.fileName();
    tmp.close();

    QVERIFY(original.saveToFile(tmpPath));

    // Load back
    auto loaded = HardwareDataContainer::loadFromFile(tmpPath);
    QCOMPARE(loaded.hardwareMap.size(), original.hardwareMap.size());

    for (auto it = original.hardwareMap.cbegin(); it != original.hardwareMap.cend(); ++it) {
        QVERIFY(loaded.hardwareMap.contains(it.key()));
        QCOMPARE(loaded.hardwareMap[it.key()].implementation, it.value().implementation);
        QCOMPARE(loaded.hardwareMap[it.key()].type, it.value().type);
    }
}

// ============================================================
// Full Experiment Loading
// ============================================================

void ExperimentLoadingTest::loadExperiment200_hardware()
{
    // Load oldest format experiment (header only to avoid FID dependency issues)
    Experiment exp(200, testDataDir() + "/200", true);

    QVERIFY(exp.d_hardwareSuccess);
    QCOMPARE(exp.d_number, 200);
    QVERIFY(exp.d_errorString.isEmpty());
}

void ExperimentLoadingTest::loadExperiment200_header()
{
    Experiment exp(200, testDataDir() + "/200", true);

    // Experiment number loaded from header
    QCOMPARE(exp.d_number, 200);

    // FTMW should be enabled (objectives.csv has FtmwType)
    QVERIFY(exp.ftmwEnabled());
    QVERIFY(!exp.lifEnabled());

    // Version info loaded
    QCOMPARE(exp.d_majorVersion, QString("1"));
    QCOMPARE(exp.d_minorVersion, QString("0"));
    QCOMPARE(exp.d_patchVersion, QString("0"));
}

void ExperimentLoadingTest::loadExperiment2638_hardware()
{
    Experiment exp(2638, testDataDir() + "/2638", true);

    QVERIFY(exp.d_hardwareSuccess);
    QCOMPARE(exp.d_number, 2638);
    QVERIFY(exp.d_errorString.isEmpty());

    // Should have PulseGenerator configs created
    QCOMPARE(exp.d_hardwareData.hardwareMap.size(), 5);
}

void ExperimentLoadingTest::loadExperiment2638_header()
{
    Experiment exp(2638, testDataDir() + "/2638", true);

    QCOMPARE(exp.d_number, 2638);
    QVERIFY(exp.ftmwEnabled());
    QVERIFY(!exp.lifEnabled());

    QCOMPARE(exp.d_majorVersion, QString("1"));
    QCOMPARE(exp.d_minorVersion, QString("0"));
}

void ExperimentLoadingTest::loadExperiment27_hardware()
{
    Experiment exp(27, testDataDir() + "/27", true);

    QVERIFY(exp.d_hardwareSuccess);
    QCOMPARE(exp.d_number, 27);
    QVERIFY(exp.d_errorString.isEmpty());

    // 5 hardware entries in new format
    QCOMPARE(exp.d_hardwareData.hardwareMap.size(), 5);

    // Verify specific hardware types were parsed correctly
    QVERIFY(exp.d_hardwareData.hardwareMap.contains("FtmwScope.default"));
    QVERIFY(exp.d_hardwareData.hardwareMap.contains("LifScope.default"));
}

void ExperimentLoadingTest::loadExperiment27_header()
{
    Experiment exp(27, testDataDir() + "/27", true);

    QCOMPARE(exp.d_number, 27);

    // LIF experiment (objectives.csv has LifType)
    QVERIFY(exp.lifEnabled());

    QCOMPARE(exp.d_majorVersion, QString("1"));
    QCOMPARE(exp.d_minorVersion, QString("1"));
}

QTEST_MAIN(ExperimentLoadingTest)
#include "tst_experimentloading.moc"
