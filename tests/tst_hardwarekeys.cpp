#include <QtTest>
#include <QCoreApplication>

#include "src/data/bcglobals.h"
#include "src/data/experiment/hardwaredatacontainer.h"

using namespace BC::Data;

class HardwareKeysTest : public QObject
{
    Q_OBJECT

public:
    HardwareKeysTest() = default;
    ~HardwareKeysTest() = default;

private slots:
    void initTestCase();

    // BC::Key::hwKey tests
    void hwKeyLabelBased();
    void hwKeyIndexBased();

    // BC::Key::parseKey tests
    void parseKeyValid();
    void parseKeyNoSeparator();
    void parseKeyMultipleSeparators();

    // BC::Key::parseIndexKey tests
    void parseIndexKeyValid();
    void parseIndexKeyNonInteger();
    void parseIndexKeyNoSeparator();

    // BC::Key::isIndexKey tests
    void isIndexKeyTrue();
    void isIndexKeyFalse();

    // Round-trip tests
    void hwKeyParseKeyRoundTrip();
    void hwKeyParseIndexKeyRoundTrip();

    // BC::Key::widgetKey tests
    void widgetKeyBasic();

    // BC::Key::migrateIndexKey tests
    void migrateIndexKeyMatching();
    void migrateIndexKeyNonMatching();

    // BC::Key::generateDefaultLabel tests
    void generateDefaultLabelEmpty();
    void generateDefaultLabelSomeUsed();
    void generateDefaultLabelAllDefaultsUsed();

    // HardwareDataContainer legacy mapping tests
    void legacyStringToHardwareTypeKnown();
    void legacyStringToHardwareTypeAliases();
    void legacyStringToHardwareTypeUnknown();
    void extractHardwareTypeFromKey();
};

void HardwareKeysTest::initTestCase()
{
    QCoreApplication::setOrganizationDomain("crabtreelab.ucdavis.edu");
    QCoreApplication::setApplicationName("BlackchirpTest");
}

// --- hwKey ---

void HardwareKeysTest::hwKeyLabelBased()
{
    QCOMPARE(BC::Key::hwKey("FtmwDigitizer", "main"), QString("FtmwDigitizer.main"));
    QCOMPARE(BC::Key::hwKey("FlowController", "frontPanel"), QString("FlowController.frontPanel"));
    QCOMPARE(BC::Key::hwKey("Clock", "reference"), QString("Clock.reference"));
}

void HardwareKeysTest::hwKeyIndexBased()
{
    QCOMPARE(BC::Key::hwKey("Clock", 0), QString("Clock.0"));
    QCOMPARE(BC::Key::hwKey("FlowController", 3), QString("FlowController.3"));
}

// --- parseKey ---

void HardwareKeysTest::parseKeyValid()
{
    auto [type, label] = BC::Key::parseKey("FtmwDigitizer.main");
    QCOMPARE(type, QString("FtmwDigitizer"));
    QCOMPARE(label, QString("main"));
}

void HardwareKeysTest::parseKeyNoSeparator()
{
    auto [type, label] = BC::Key::parseKey("FtmwDigitizer");
    QCOMPARE(type, QString("FtmwDigitizer"));
    QVERIFY(label.isEmpty());
}

void HardwareKeysTest::parseKeyMultipleSeparators()
{
    // "Widget.FtmwDigitizer.main" — split on first dot only
    auto [type, label] = BC::Key::parseKey("Widget.FtmwDigitizer.main");
    QCOMPARE(type, QString("Widget"));
    // Implementation splits on ".", second element is "FtmwDigitizer"
    QCOMPARE(label, QString("FtmwDigitizer"));
}

// --- parseIndexKey ---

void HardwareKeysTest::parseIndexKeyValid()
{
    auto [type, index] = BC::Key::parseIndexKey("Clock.0");
    QCOMPARE(type, QString("Clock"));
    QCOMPARE(index, 0);

    auto [type2, index2] = BC::Key::parseIndexKey("FlowController.5");
    QCOMPARE(type2, QString("FlowController"));
    QCOMPARE(index2, 5);
}

void HardwareKeysTest::parseIndexKeyNonInteger()
{
    auto [type, index] = BC::Key::parseIndexKey("FtmwDigitizer.main");
    QCOMPARE(type, QString("FtmwDigitizer"));
    QCOMPARE(index, -1);
}

void HardwareKeysTest::parseIndexKeyNoSeparator()
{
    auto [type, index] = BC::Key::parseIndexKey("FtmwDigitizer");
    QCOMPARE(type, QString("FtmwDigitizer"));
    QCOMPARE(index, -1);
}

// --- isIndexKey ---

void HardwareKeysTest::isIndexKeyTrue()
{
    QVERIFY(BC::Key::isIndexKey("Clock.0"));
    QVERIFY(BC::Key::isIndexKey("FlowController.2"));
}

void HardwareKeysTest::isIndexKeyFalse()
{
    QVERIFY(!BC::Key::isIndexKey("FtmwDigitizer.main"));
    QVERIFY(!BC::Key::isIndexKey("FtmwDigitizer"));
    QVERIFY(!BC::Key::isIndexKey("Clock.reference"));
}

// --- round-trips ---

void HardwareKeysTest::hwKeyParseKeyRoundTrip()
{
    QString type = "FlowController";
    QString label = "frontPanel";
    auto key = BC::Key::hwKey(type, label);
    auto [parsedType, parsedLabel] = BC::Key::parseKey(key);
    QCOMPARE(parsedType, type);
    QCOMPARE(parsedLabel, label);
}

void HardwareKeysTest::hwKeyParseIndexKeyRoundTrip()
{
    QString type = "Clock";
    int index = 3;
    auto key = BC::Key::hwKey(type, index);
    auto [parsedType, parsedIndex] = BC::Key::parseIndexKey(key);
    QCOMPARE(parsedType, type);
    QCOMPARE(parsedIndex, index);
}

// --- widgetKey ---

void HardwareKeysTest::widgetKeyBasic()
{
    auto wk = BC::Key::widgetKey("ControlWidget", "FtmwDigitizer.main");
    QCOMPARE(wk, QString("ControlWidget.FtmwDigitizer.main"));
}

// --- migrateIndexKey ---

void HardwareKeysTest::migrateIndexKeyMatching()
{
    auto result = BC::Key::migrateIndexKey("Clock.2", "Clock", 2);
    QCOMPARE(result, QString("Clock.Device2"));
}

void HardwareKeysTest::migrateIndexKeyNonMatching()
{
    // Type mismatch
    auto result = BC::Key::migrateIndexKey("Clock.2", "FlowController", 2);
    QCOMPARE(result, QString("Clock.2"));

    // Index mismatch
    auto result2 = BC::Key::migrateIndexKey("Clock.2", "Clock", 5);
    QCOMPARE(result2, QString("Clock.2"));
}

// --- generateDefaultLabel ---

void HardwareKeysTest::generateDefaultLabelEmpty()
{
    auto label = BC::Key::generateDefaultLabel("Clock", {});
    QCOMPARE(label, QString("Default"));
}

void HardwareKeysTest::generateDefaultLabelSomeUsed()
{
    auto label = BC::Key::generateDefaultLabel("Clock", {"Default"});
    QCOMPARE(label, QString("Main"));

    auto label2 = BC::Key::generateDefaultLabel("Clock", {"Default", "Main"});
    QCOMPARE(label2, QString("Primary"));

    auto label3 = BC::Key::generateDefaultLabel("Clock", {"Default", "Main", "Primary"});
    QCOMPARE(label3, QString("Secondary"));

    auto label4 = BC::Key::generateDefaultLabel("Clock", {"Default", "Main", "Primary", "Secondary"});
    QCOMPARE(label4, QString("Backup"));
}

void HardwareKeysTest::generateDefaultLabelAllDefaultsUsed()
{
    auto label = BC::Key::generateDefaultLabel("Clock",
        {"Default", "Main", "Primary", "Secondary", "Backup"});
    QCOMPARE(label, QString("Device1"));

    auto label2 = BC::Key::generateDefaultLabel("Clock",
        {"Default", "Main", "Primary", "Secondary", "Backup", "Device1"});
    QCOMPARE(label2, QString("Device2"));
}

// --- HardwareDataContainer legacy mappings ---

void HardwareKeysTest::legacyStringToHardwareTypeKnown()
{
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("IOBoard"), HardwareType::IOBoard);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("PulseGenerator"), HardwareType::PulseGenerator);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("FlowController"), HardwareType::FlowController);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("PressureController"), HardwareType::PressureController);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("TemperatureController"), HardwareType::TemperatureController);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("FtmwDigitizer"), HardwareType::FtmwDigitizer);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("Clock"), HardwareType::Clock);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("AWG"), HardwareType::AWG);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("LifDigitizer"), HardwareType::LifDigitizer);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("LifLaser"), HardwareType::LifLaser);
}

void HardwareKeysTest::legacyStringToHardwareTypeAliases()
{
    // "FtmwDigitizer" is a pre-label-era name for FtmwDigitizer
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("FtmwDigitizer"), HardwareType::FtmwDigitizer);
    // "GpibController" maps to GPIBController enum
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("GpibController"), HardwareType::GPIBController);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("GPIBController"), HardwareType::GPIBController);
}

void HardwareKeysTest::legacyStringToHardwareTypeUnknown()
{
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("NonExistentHardware"), HardwareType::Unknown);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType(""), HardwareType::Unknown);
}

void HardwareKeysTest::extractHardwareTypeFromKey()
{
    QCOMPARE(HardwareDataContainer::extractHardwareType("FlowController.frontPanel"), HardwareType::FlowController);
    QCOMPARE(HardwareDataContainer::extractHardwareType("FtmwDigitizer.0"), HardwareType::FtmwDigitizer);
    QCOMPARE(HardwareDataContainer::extractHardwareType("Clock.reference"), HardwareType::Clock);
    QCOMPARE(HardwareDataContainer::extractHardwareType("UnknownType.foo"), HardwareType::Unknown);
}

QTEST_MAIN(HardwareKeysTest)
#include "tst_hardwarekeys.moc"
