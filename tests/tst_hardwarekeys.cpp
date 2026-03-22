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
    void hardwareTypeToLegacyStringKnown();
    void hardwareTypeToLegacyStringUnknown();
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
    QCOMPARE(BC::Key::hwKey("FtmwScope", "main"), QString("FtmwScope.main"));
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
    auto [type, label] = BC::Key::parseKey("FtmwScope.main");
    QCOMPARE(type, QString("FtmwScope"));
    QCOMPARE(label, QString("main"));
}

void HardwareKeysTest::parseKeyNoSeparator()
{
    auto [type, label] = BC::Key::parseKey("FtmwScope");
    QCOMPARE(type, QString("FtmwScope"));
    QVERIFY(label.isEmpty());
}

void HardwareKeysTest::parseKeyMultipleSeparators()
{
    // "Widget.FtmwScope.main" — split on first dot only
    auto [type, label] = BC::Key::parseKey("Widget.FtmwScope.main");
    QCOMPARE(type, QString("Widget"));
    // Implementation splits on ".", second element is "FtmwScope"
    QCOMPARE(label, QString("FtmwScope"));
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
    auto [type, index] = BC::Key::parseIndexKey("FtmwScope.main");
    QCOMPARE(type, QString("FtmwScope"));
    QCOMPARE(index, -1);
}

void HardwareKeysTest::parseIndexKeyNoSeparator()
{
    auto [type, index] = BC::Key::parseIndexKey("FtmwScope");
    QCOMPARE(type, QString("FtmwScope"));
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
    QVERIFY(!BC::Key::isIndexKey("FtmwScope.main"));
    QVERIFY(!BC::Key::isIndexKey("FtmwScope"));
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
    auto wk = BC::Key::widgetKey("ControlWidget", "FtmwScope.main");
    QCOMPARE(wk, QString("ControlWidget.FtmwScope.main"));
}

// --- migrateIndexKey ---

void HardwareKeysTest::migrateIndexKeyMatching()
{
    auto result = BC::Key::migrateIndexKey("Clock.2", "Clock", 2);
    QCOMPARE(result, QString("Clock.device2"));
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
    QCOMPARE(label, QString("default"));
}

void HardwareKeysTest::generateDefaultLabelSomeUsed()
{
    auto label = BC::Key::generateDefaultLabel("Clock", {"default"});
    QCOMPARE(label, QString("main"));

    auto label2 = BC::Key::generateDefaultLabel("Clock", {"default", "main"});
    QCOMPARE(label2, QString("primary"));

    auto label3 = BC::Key::generateDefaultLabel("Clock", {"default", "main", "primary"});
    QCOMPARE(label3, QString("secondary"));

    auto label4 = BC::Key::generateDefaultLabel("Clock", {"default", "main", "primary", "secondary"});
    QCOMPARE(label4, QString("backup"));
}

void HardwareKeysTest::generateDefaultLabelAllDefaultsUsed()
{
    auto label = BC::Key::generateDefaultLabel("Clock",
        {"default", "main", "primary", "secondary", "backup"});
    QCOMPARE(label, QString("device1"));

    auto label2 = BC::Key::generateDefaultLabel("Clock",
        {"default", "main", "primary", "secondary", "backup", "device1"});
    QCOMPARE(label2, QString("device2"));
}

// --- HardwareDataContainer legacy mappings ---

void HardwareKeysTest::legacyStringToHardwareTypeKnown()
{
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("IOBoard"), HardwareType::IOBoard);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("PulseGenerator"), HardwareType::PulseGenerator);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("FlowController"), HardwareType::FlowController);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("PressureController"), HardwareType::PressureController);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("TemperatureController"), HardwareType::TemperatureController);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("FtmwScope"), HardwareType::FtmwScope);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("Clock"), HardwareType::Clock);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("AWG"), HardwareType::AWG);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("LifScope"), HardwareType::LifScope);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("LifLaser"), HardwareType::LifLaser);
}

void HardwareKeysTest::legacyStringToHardwareTypeAliases()
{
    // "FtmwDigitizer" is a pre-label-era name for FtmwScope
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("FtmwDigitizer"), HardwareType::FtmwScope);
    // "GpibController" maps to GPIBController enum
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("GpibController"), HardwareType::GPIBController);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("GPIBController"), HardwareType::GPIBController);
}

void HardwareKeysTest::legacyStringToHardwareTypeUnknown()
{
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType("NonExistentHardware"), HardwareType::Unknown);
    QCOMPARE(HardwareDataContainer::legacyStringToHardwareType(""), HardwareType::Unknown);
}

void HardwareKeysTest::hardwareTypeToLegacyStringKnown()
{
    QCOMPARE(HardwareDataContainer::hardwareTypeToLegacyString(HardwareType::FtmwScope), QString("FtmwScope"));
    QCOMPARE(HardwareDataContainer::hardwareTypeToLegacyString(HardwareType::GPIBController), QString("GPIBController"));
    QCOMPARE(HardwareDataContainer::hardwareTypeToLegacyString(HardwareType::FlowController), QString("FlowController"));
}

void HardwareKeysTest::hardwareTypeToLegacyStringUnknown()
{
    QVERIFY(HardwareDataContainer::hardwareTypeToLegacyString(HardwareType::Unknown).isEmpty());
}

void HardwareKeysTest::extractHardwareTypeFromKey()
{
    QCOMPARE(HardwareDataContainer::extractHardwareType("FlowController.frontPanel"), HardwareType::FlowController);
    QCOMPARE(HardwareDataContainer::extractHardwareType("FtmwDigitizer.0"), HardwareType::FtmwScope);
    QCOMPARE(HardwareDataContainer::extractHardwareType("Clock.reference"), HardwareType::Clock);
    QCOMPARE(HardwareDataContainer::extractHardwareType("UnknownType.foo"), HardwareType::Unknown);
}

QTEST_MAIN(HardwareKeysTest)
#include "tst_hardwarekeys.moc"
