#include <QtTest>
#include <QCoreApplication>
#include <QSettings>

#include <src/data/storage/settingsstorage.h>

class SettingsStorageTest : public QObject, public SettingsStorage
{
    Q_OBJECT

public:
    SettingsStorageTest();
    ~SettingsStorageTest();

    enum TestEnum {
        TestValue1,
        TestValue2,
        TestValue3,
        TestValue4,
        TestValue5
    };
    Q_ENUM(TestEnum)

    int intGetter() const { return d_int; }
    double doubleGetter() const { return d_double; }

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testBaseRead();
    void testGetter();
    void testGetMultiple();
    void testContains();
    void testSet();
    void testDefault();
    void testNestedGroupRead();
    void testHardwareRead();
    void testDestruction();
    
    // Enhanced coverage tests
    void testArrayOperations();
    void testArrayValueSetGet();
    void testEdgeCases();
    void testKeyManagement();
    void testDiscardChanges();
    void testGetterConflicts();
    void testArrayAppend();
    void testClearValue();
    void testConstructorVariants();
    
    // Group functionality tests
    void testGroupOperations();
    void testGroupValueSetGet();
    void testGroupConflicts();
    void testGroupMultipleValues();
    void testGroupKeys();
    void testCrossContamination();


private:
    int d_int = 10;
    double d_double = 10.1;

    void initSettingsFile();

};

class DestructorTest : public SettingsStorage
{
public:
    // Use the explicit-org/app SettingsStorage ctor with a test-only
    // organization name so the test never reads or writes the user's
    // real Blackchirp settings, regardless of platform.
    DestructorTest() : SettingsStorage("CrabtreeLabTest","BlackchirpTest",{},General) {
        registerGetter("destructTest",this,&DestructorTest::desructGetter);
    }
    virtual ~DestructorTest(){};

    int desructGetter() const { return d_test; }

private:
    int d_test = 144;
};

Q_DECLARE_METATYPE(SettingsStorageTest::TestEnum)

// All QSettings instances in the test target the "CrabtreeLabTest"
// organization, so the test never reads or writes the user's real
// Blackchirp settings under "CrabtreeLab". The QCoreApplication
// globals are set to match so any 0-arg QSettings() constructed
// during the test (e.g., inside SettingsStorage's helpers) resolves
// to the same plist on every platform.
//
// organizationDomain is deliberately *not* set: on macOS, Qt's 0-arg
// QSettings ctor prefers organizationDomain over organizationName for
// the plist filename, while the 2-arg QSettings(orgName, appName)
// form always uses the literal orgName argument. Setting both pulls
// 0-arg readOnly instances and 2-arg d_settings into different plist
// files (CrabtreeLabTest.BlackchirpTest.plist vs the reversed-domain
// form), silently splitting reads and writes. Linux ignores the
// domain in both ctor forms, which is why the bug was macOS-only.
SettingsStorageTest::SettingsStorageTest() : SettingsStorage("CrabtreeLabTest","BlackchirpTest",{},General)
{
    QCoreApplication::setApplicationName("BlackchirpTest");
    QCoreApplication::setOrganizationName("CrabtreeLabTest");

}

SettingsStorageTest::~SettingsStorageTest()
{

}

void SettingsStorageTest::initTestCase()
{
    initSettingsFile();
}

void SettingsStorageTest::cleanupTestCase()
{
    // Wipe the test organization's settings so subsequent runs start
    // from a clean state. This stays inside "CrabtreeLabTest" and
    // never touches the user's real Blackchirp config.
    QSettings cleanup("CrabtreeLabTest","BlackchirpTest");
    cleanup.setFallbacksEnabled(false);
    cleanup.clear();
    cleanup.sync();
}

void SettingsStorageTest::testBaseRead()
{
    readAll();
    QCOMPARE(get("testInt").toInt(),42);
    QCOMPARE(get<int>("testInt"),42);
    QCOMPARE(get("testDouble").toDouble(),1.3e-1);
    QCOMPARE(get<double>("testDouble"),1.3e-1);
    QCOMPARE(get("testString").toString(),QString("Hello world!"));
    QCOMPARE(get<QString>("testString"),QString("Hello world!"));
    QCOMPARE(get("testEnum"),QVariant(TestValue3));
    QCOMPARE(get<SettingsStorageTest::TestEnum>("testEnum"),TestValue3);

    QCOMPARE(getArraySize("testArray"),10);
    QCOMPARE(getArraySize("nonExistentKey"),0);

    for(int i=0; i<10; ++i)
    {
        auto m = getArrayMap("testArray",i);
        QCOMPARE(m.at("testArrayInt").toInt(),i);
        QCOMPARE(m.at("testArrayDouble").toDouble(),0.5+i);
        QCOMPARE(m.at("testArrayString").toString(),QString::number(i));
        QCOMPARE(m.at("testArrayEnum"),QVariant(TestValue2));
    }

    //default values
    QCOMPARE(get("nonExistentKey"),QVariant());
    QCOMPARE(get<int>("nonExistentKey"),0);
    QCOMPARE(get<double>("nonExistentKey")+1.0,1.0);
    QCOMPARE(get("nonExistentKey",10),10);
    QCOMPARE(get("nonExistentKey",12.3),12.3);

    //reading from map directly
    QCOMPARE(getArrayValue("testArray",3,"testArrayInt"),QVariant(3));
    QCOMPARE(getArrayValue("testArray",20,"testArrayInt",6),QVariant(6));
    QCOMPARE(getArrayValue("testArray",20,"testArrayInt"),QVariant());
    QCOMPARE(getArrayValue<int>("testArray",3,"testArrayInt"),3);
    QCOMPARE(getArrayValue<int>("testArray",20,"testArrayInt",6),6);
    QCOMPARE(getArrayValue<int>("testArray",20,"testArrayInt"),0);

}

void SettingsStorageTest::testGetter()
{
    readAll();
    QCOMPARE(registerGetter("testInt",this,&SettingsStorageTest::intGetter),true);
    QCOMPARE(registerGetter("testDouble",this,&SettingsStorageTest::doubleGetter),true);
    QCOMPARE(get<int>("testInt"),10);
    QCOMPARE(get<double>("testDouble"),10.1);

    d_int = 20;
    d_double *= 3;

    save();
    SettingsStorage readOnly;

    QCOMPARE(readOnly.get<int>("testInt"),20);
    QCOMPARE(readOnly.get<double>("testDouble"),10.1*3);

    d_int = 30;
    d_double = 5.1;

    QCOMPARE(unRegisterGetter("testInt",true),QVariant(30));
    QCOMPARE(unRegisterGetter("testDouble",false),QVariant(5.1));
    QCOMPARE(unRegisterGetter("nonExistentKey"),QVariant());
    QCOMPARE(unRegisterGetter("testString"),QVariant());

    SettingsStorage readOnly2;

    QCOMPARE(readOnly2.get<int>("testInt"),30);
    QCOMPARE(readOnly2.get<double>("testDouble"),10.1*3);

    QCOMPARE(registerGetter("testInt",this,&SettingsStorageTest::intGetter),true);
    QCOMPARE(registerGetter("testDouble",this,&SettingsStorageTest::doubleGetter),true);

    d_int = 50;
    d_double = 96.1;
    clearGetters(true);
    QCOMPARE(get<int>("testInt"),50);
    QCOMPARE(get<double>("testDouble"),96.1);

    SettingsStorage readOnly3;
    QCOMPARE(readOnly3.get<int>("testInt"),50);
    QCOMPARE(readOnly3.get<double>("testDouble"),96.1);

    int x = 3;
    registerGetter("testLambda",std::function<int()>{[x]() { return x + 1; }});
    QCOMPARE(get<int>("testLambda",0),4);

}

void SettingsStorageTest::testGetMultiple()
{
    initSettingsFile();
    clearGetters(false);
    readAll();

    d_int = 50;
    //getMultiple should skip any nonexistent keys and keys corresponding to arrays
    registerGetter("testInt",this,&SettingsStorageTest::intGetter);
    auto m = getMultiple({"testInt","testDouble","testString","testEnum","testArray","nonExistentKey" });

    d_int = 20;

    QCOMPARE(m.find("testArray"),m.end());
    QCOMPARE(m.find("nonExistentKey"),m.end());
    QCOMPARE(m.find("testInt")->second,QVariant(50)); //m should contain d_int at the time it was called
    QCOMPARE(m.find("testDouble")->second,QVariant(1.3e-1));
    QCOMPARE(m.find("testString")->second,QVariant(QString("Hello world!")));
    QCOMPARE(m.find("testEnum")->second,QVariant(TestValue3));
}

void SettingsStorageTest::testContains()
{
    initSettingsFile();
    readAll();

    QCOMPARE(containsValue("testInt"),true);
    QCOMPARE(containsValue("nonExistentKey"),false);
    QCOMPARE(containsArray("testArray"),true);
    QCOMPARE(containsArray("nonExistentKey"),false);

    registerGetter("testDouble",this,&SettingsStorageTest::doubleGetter);
    QCOMPARE(containsValue("testDouble"),true);
    QCOMPARE(containsArray("testDouble"),false);
    QCOMPARE(containsValue("testArray"),false);

}

void SettingsStorageTest::testSet()
{
    initSettingsFile();
    clearGetters(false);
    readAll();

    //change value of int and double, one using a getter
    registerGetter("testInt",this,&SettingsStorageTest::intGetter);
    d_int = 20;
    set("testDouble",0.5,false);

    //at this point, readOnly should contain original values; nothing has been saved
    SettingsStorage readOnly;
    QCOMPARE(readOnly.get("testInt"),QVariant(42));
    QCOMPARE(readOnly.get("testDouble"),QVariant(1.3e-1));

    //now set multiple values and save
    auto out = setMultiple({ {"testDouble",5.5}, {"testString","Different string"}},true);
    QCOMPARE(out.at("testString"),true);
    QCOMPARE(out.at("testDouble"),true);

    setArray("testArray",{},true);

    setArray("testArray2", { {{"testArray2Key1",true},{"testArray2Key2",false}}},true);

    SettingsStorage readOnly2;
    QCOMPARE(readOnly2.get("testDouble"),QVariant(5.5));
    QCOMPARE(readOnly2.get("testString"),QVariant("Different string"));
    QCOMPARE(readOnly2.containsArray("testArray"),false);
    QCOMPARE(readOnly2.containsArray("testArray2"),true);
    auto l = readOnly2.getArray("testArray2");
    QCOMPARE(l.size(),1);
    QCOMPARE(l.front().at("testArray2Key1"),true);
    QCOMPARE(l.front().at("testArray2Key2"),false);

    //test clearing and appending array maps
    clearValue("testInt");
    clearValue("testDouble");
    appendArrayMap("testArray2",{{"newMapKey",1.0}});
    save();

    SettingsStorage readOnly3;
    QCOMPARE(readOnly3.get<int>("testInt"),0);
    QCOMPARE(readOnly3.get<double>("testDouble",11.4),11.4);
    QCOMPARE(readOnly3.getArrayValue<double>("testArray2",1,"newMapKey",2.0),1.0);
}

void SettingsStorageTest::testDefault()
{
    initSettingsFile();
    clearGetters(false);
    readAll();

    QCOMPARE(getOrSetDefault("testInt",1),42);
    QCOMPARE(getOrSetDefault("newKey",1000),1000);
    QCOMPARE(getOrSetDefault("testArray",2),0);

    SettingsStorage readOnly;
    QCOMPARE(readOnly.containsValue("newKey"),true);
    QCOMPARE(readOnly.get("newKey"),QVariant(1000));
}

void SettingsStorageTest::testNestedGroupRead()
{
    initSettingsFile();
    SettingsStorage readOnly(QStringList{"readOnly","nested"},General);

    QCOMPARE(readOnly.get("testInt").toInt(),420);
    QCOMPARE(readOnly.get<int>("testInt"),420);
    QCOMPARE(readOnly.get("testDouble").toDouble(),1.3e-2);
    QCOMPARE(readOnly.get<double>("testDouble"),1.3e-2);
    QCOMPARE(readOnly.get("testString").toString(),QString("Hello world 2.0!"));
    QCOMPARE(readOnly.get<QString>("testString"),QString("Hello world 2.0!"));
    QCOMPARE(readOnly.get("testEnum"),QVariant(TestValue4));
    QCOMPARE(readOnly.get<SettingsStorageTest::TestEnum>("testEnum"),TestValue4);

    for(int i=0; i<10; ++i)
    {
        auto m = readOnly.getArrayMap("testArray",i);
        QCOMPARE(m.at("testArrayInt").toInt(),i*10);
        QCOMPARE(m.at("testArrayDouble").toDouble(),0.5+i*10);
        QCOMPARE(m.at("testArrayString").toString(),QString::number(i*10));
        QCOMPARE(m.at("testArrayEnum"),QVariant(TestValue5));
    }
}

void SettingsStorageTest::testHardwareRead()
{
    initSettingsFile();
    SettingsStorage readHardware("hardwareKey",Hardware);
    QCOMPARE(readHardware.get<int>("hardwareInt"),10);
    QCOMPARE(readHardware.get<double>("hardwareDouble"),44.4);
    QCOMPARE(readHardware.get<QString>("hardwareName"),QString("My Hardware"));
}

void SettingsStorageTest::testDestruction()
{
    initSettingsFile();
    DestructorTest *d = new DestructorTest;
    delete d;

    SettingsStorage readOnly;
    QCOMPARE(readOnly.get<int>("destructTest"),144);
}

void SettingsStorageTest::initSettingsFile()
{
    // Construct with the same explicit org/app as the test class's own
    // d_settings so both QSettings instances resolve to the same domain
    // on every platform — mixing the 0-arg form here with the 2-arg
    // form inside SettingsStorage produced two different CFPreferences
    // domains on macOS (Qt's 0-arg ctor prefers organizationDomain
    // there, while 2-arg uses the literal arg).
    QSettings s("CrabtreeLabTest","BlackchirpTest");
    s.setFallbacksEnabled(false);
    s.clear();
    s.sync();

    //Write some test settings for read/write
    s.beginGroup("Blackchirp");
    s.setValue("testInt",42);
    s.setValue("testDouble",1.3e-1);
    s.setValue("testString","Hello world!");
    s.setValue("testEnum",TestValue3);
    s.beginWriteArray("testArray");
    for(int i=0; i<10; ++i)
    {
        s.setArrayIndex(i);
        s.setValue("testArrayInt",i);
        s.setValue("testArrayDouble",0.5+i);
        s.setValue("testArrayString",QString::number(i));
        s.setValue("testArrayEnum",TestValue2);
    }
    s.endArray();
    s.endGroup();

    //Write settings for a readonly test of a nested group path
    s.beginGroup("readOnly");
    s.beginGroup("nested");
    s.setValue("testInt",420);
    s.setValue("testDouble",1.3e-2);
    s.setValue("testString","Hello world 2.0!");
    s.setValue("testEnum",TestValue4);
    s.beginWriteArray("testArray");
    for(int i=0; i<10; ++i)
    {
        s.setArrayIndex(i);
        s.setValue("testArrayInt",i*10);
        s.setValue("testArrayDouble",0.5+i*10);
        s.setValue("testArrayString",QString::number(i*10));
        s.setValue("testArrayEnum",TestValue5);
    }
    s.endArray();
    s.endGroup();
    s.endGroup();

    //Write settings for hardware (flat format — no sub-group nesting)
    s.beginGroup("hardwareKey");
    s.setValue("model","hardwareSubKey");
    s.setValue("hardwareInt",10);
    s.setValue("hardwareName","My Hardware");
    s.setValue("hardwareDouble",44.4);
    s.endGroup();

    s.sync();
}

void SettingsStorageTest::testArrayOperations()
{
    initSettingsFile();
    clearGetters(false);
    readAll();
    
    // Test setArray with various scenarios
    std::vector<SettingsStorage::SettingsMap> newArray = {
        {{"key1", 10}, {"key2", "test1"}},
        {{"key1", 20}, {"key2", "test2"}},
        {{"key1", 30}, {"key2", "test3"}}
    };
    
    setArray("newTestArray", newArray, true);
    
    // Verify array was written correctly
    SettingsStorage readOnly;
    QCOMPARE(readOnly.getArraySize("newTestArray"), 3);
    
    for(std::size_t i = 0; i < 3; ++i) {
        auto map = readOnly.getArrayMap("newTestArray", i);
        QCOMPARE(map.at("key1").toInt(), static_cast<int>(10 * (i + 1)));
        QCOMPARE(map.at("key2").toString(), QString("test%1").arg(i + 1));
    }
    
    // Test empty array (should remove from settings)
    setArray("newTestArray", {}, true);
    SettingsStorage readOnly2;
    QCOMPARE(readOnly2.containsArray("newTestArray"), false);
    QCOMPARE(readOnly2.getArraySize("newTestArray"), 0);
}

void SettingsStorageTest::testArrayValueSetGet()
{
    initSettingsFile();
    clearGetters(false);
    readAll();
    
    // Create a test array
    std::vector<SettingsStorage::SettingsMap> testArray = {
        {{"value", 100}, {"name", "first"}},
        {{"value", 200}, {"name", "second"}},
        {{"value", 300}, {"name", "third"}}
    };
    setArray("setGetArray", testArray, false);
    
    // Test setArrayValue
    QCOMPARE(setArrayValue("setGetArray", 1, "value", 250, false), true);
    QCOMPARE(setArrayValue("setGetArray", 1, "newKey", QString("newValue"), false), true);
    QCOMPARE(setArrayValue("setGetArray", 10, "value", 999, false), false); // out of bounds
    QCOMPARE(setArrayValue("nonExistentArray", 0, "key", QString("value"), false), false);
    
    // Verify changes
    QCOMPARE(getArrayValue<int>("setGetArray", 1, "value", 0), 250);
    QCOMPARE(getArrayValue<QString>("setGetArray", 1, "newKey", ""), QString("newValue"));
    QCOMPARE(getArrayValue<QString>("setGetArray", 1, "name", ""), QString("second")); // unchanged
    
    // Test template versions
    QCOMPARE(setArrayValue("setGetArray", 0, "templateTest", 42.5, false), true);
    QCOMPARE(getArrayValue<double>("setGetArray", 0, "templateTest", 0.0), 42.5);
    
    // Test out of bounds and non-existent keys
    QCOMPARE(getArrayValue("setGetArray", 99, "value", QVariant("default")), QVariant("default"));
    QCOMPARE(getArrayValue<int>("setGetArray", 99, "value", 123), 123);
    QCOMPARE(getArrayValue<QString>("setGetArray", 0, "nonExistent", "default"), QString("default"));
}

void SettingsStorageTest::testEdgeCases()
{
    initSettingsFile();
    clearGetters(false);
    readAll();
    
    // Test empty key names
    QCOMPARE(set("", QString("emptyKey"), false), true);
    QCOMPARE(get(""), QVariant("emptyKey"));
    
    // Test very long key names
    QString longKey = QString("a").repeated(1000);
    QCOMPARE(set(longKey, QString("longKeyValue"), false), true);
    QCOMPARE(get(longKey).toString(), QString("longKeyValue"));
    
    // Test special characters in keys
    QString specialKey = "key/with\\special:chars<>|*?\"";
    QCOMPARE(set(specialKey, QString("specialValue"), false), true);
    QCOMPARE(get(specialKey).toString(), QString("specialValue"));
    
    // Test null and empty QVariant values
    QCOMPARE(set("nullTest", QVariant(), false), true);
    QCOMPARE(get("nullTest"), QVariant());
    QCOMPARE(get("nullTest").isNull(), true);
    
    // Test very large numbers
    double largeDouble = 1.23456789e308;
    qint64 largeInt = 9223372036854775807LL;
    QCOMPARE(set("largeDouble", largeDouble, false), true);
    QCOMPARE(set("largeInt", largeInt, false), true);
    QCOMPARE(get<double>("largeDouble"), largeDouble);
    QCOMPARE(get<qint64>("largeInt"), largeInt);
    
    // Test Unicode strings
    QString unicodeString = "Hello 世界 🌍 мир";
    QCOMPARE(set("unicode", unicodeString, false), true);
    QCOMPARE(get<QString>("unicode"), unicodeString);
}

void SettingsStorageTest::testKeyManagement()
{
    initSettingsFile();
    clearGetters(false);
    readAll();
    
    // Test keys() and arrayKeys()
    QStringList currentKeys = keys();
    QStringList currentArrayKeys = arrayKeys();
    
    // Should have test data keys
    QVERIFY(currentKeys.contains("testInt"));
    QVERIFY(currentKeys.contains("testDouble"));
    QVERIFY(currentKeys.contains("testString"));
    QVERIFY(currentArrayKeys.contains("testArray"));
    
    // Add new keys and verify they appear
    set("newKey1", 123, false);
    set("newKey2", QString("test"), false);
    setArray("newArray", {{{QString("arrayKey"), QString("arrayValue")}}}, false);
    
    QStringList updatedKeys = keys();
    QStringList updatedArrayKeys = arrayKeys();
    
    QVERIFY(updatedKeys.contains("newKey1"));
    QVERIFY(updatedKeys.contains("newKey2"));
    QVERIFY(updatedArrayKeys.contains("newArray"));
    
    // Verify keys are sorted/consistent
    QCOMPARE(updatedKeys.count(), currentKeys.count() + 2);
    QCOMPARE(updatedArrayKeys.count(), currentArrayKeys.count() + 1);
}

void SettingsStorageTest::testDiscardChanges()
{
    initSettingsFile();
    clearGetters(false);
    readAll();
    
    // Make changes without writing
    set("discardTest1", QString("value1"), false);
    set("discardTest2", QString("value2"), false);
    registerGetter("discardGetter", this, &SettingsStorageTest::intGetter);
    d_int = 999;
    
    // Enable discard mode
    discardChanges(true);
    
    // Changes should not be saved when object is destroyed
    save(); // Should do nothing due to discard flag
    
    // Verify changes weren't written
    SettingsStorage readOnly;
    QCOMPARE(readOnly.containsValue("discardTest1"), false);
    QCOMPARE(readOnly.containsValue("discardTest2"), false);
    QCOMPARE(readOnly.get<int>("discardGetter", 0), 0); // Should not exist
    
    // Disable discard mode and save
    discardChanges(false);
    save();
    
    // Now changes should be written
    SettingsStorage readOnly2;
    QCOMPARE(readOnly2.get<QString>("discardTest1"), QString("value1"));
    QCOMPARE(readOnly2.get<QString>("discardTest2"), QString("value2"));
    QCOMPARE(readOnly2.get<int>("discardGetter"), 999);
}

void SettingsStorageTest::testGetterConflicts()
{
    initSettingsFile();
    clearGetters(false);
    readAll();
    
    // Test getter registration conflicts
    QCOMPARE(registerGetter("testInt", this, &SettingsStorageTest::intGetter), true);
    
    // Try to register getter for array key (should fail)
    QCOMPARE(registerGetter("testArray", this, &SettingsStorageTest::intGetter), false);
    
    // Try to set value for getter key (should fail)
    QCOMPARE(set("testInt", 999, false), false);
    
    // Try to set array for getter key - arrays and getters can coexist
    setArray("testInt", {{{QString("key"), QString("value")}}}, false);
    QCOMPARE(containsValue("testInt"), true); // Should still be a getter
    QCOMPARE(containsArray("testInt"), true); // Array should also exist
    
    // Unregister getter and verify we can then set values
    d_int = 777;
    unRegisterGetter("testInt", false);
    // After unregistering, clear all traces of the key so we can set a new value
    clearValue("testInt"); // This should clear both the stored value from unregistering and the array
    QCOMPARE(containsValue("testInt"), false); // Should be completely clear now
    QCOMPARE(containsArray("testInt"), false); // Array should also be cleared
    QCOMPARE(set("testInt", 888, false), true);
    QCOMPARE(get<int>("testInt"), 888); // Should be stored value, not getter
    
    // Test lambda getter conflicts
    int lambdaValue = 42;
    auto lambdaGetter = [&lambdaValue]() { return lambdaValue; };
    QCOMPARE(registerGetter("lambdaTest", std::function<int()>(lambdaGetter)), true);
    
    lambdaValue = 84;
    QCOMPARE(get<int>("lambdaTest"), 84);
    
    // Verify we can't override with regular set
    QCOMPARE(set("lambdaTest", 999, false), false);
    QCOMPARE(get<int>("lambdaTest"), 84); // Should still use lambda
}

void SettingsStorageTest::testArrayAppend()
{
    initSettingsFile();
    clearGetters(false);
    readAll();
    
    // Test appending to new array
    appendArrayMap("appendTest", {{"first", 1}, {"name", "one"}}, false);
    appendArrayMap("appendTest", {{"first", 2}, {"name", "two"}}, false);
    appendArrayMap("appendTest", {{"first", 3}, {"name", "three"}}, false);
    
    QCOMPARE(getArraySize("appendTest"), 3);
    QCOMPARE(getArrayValue<int>("appendTest", 0, "first", 0), 1);
    QCOMPARE(getArrayValue<QString>("appendTest", 1, "name", ""), QString("two"));
    QCOMPARE(getArrayValue<int>("appendTest", 2, "first", 0), 3);
    
    // Test appending to existing array
    std::size_t originalSize = getArraySize("testArray");
    appendArrayMap("testArray", {{"testArrayInt", 999}, {"testArrayString", "appended"}}, false);
    
    QCOMPARE(getArraySize("testArray"), originalSize + 1);
    QCOMPARE(getArrayValue<int>("testArray", originalSize, "testArrayInt", 0), 999);
    QCOMPARE(getArrayValue<QString>("testArray", originalSize, "testArrayString", ""), QString("appended"));
    
    // Test immediate write option
    appendArrayMap("immediateArray", {{"immediate", true}}, true);
    
    SettingsStorage readOnly;
    QCOMPARE(readOnly.getArraySize("immediateArray"), 1);
    QCOMPARE(readOnly.getArrayValue<bool>("immediateArray", 0, "immediate", false), true);
}

void SettingsStorageTest::testClearValue()
{
    initSettingsFile();
    clearGetters(false);
    readAll();
    
    // Verify test value exists
    QVERIFY(containsValue("testInt"));
    
    // Clear regular value
    clearValue("testInt");
    QCOMPARE(containsValue("testInt"), false);
    
    // Verify it's removed from QSettings
    SettingsStorage readOnly;
    QCOMPARE(readOnly.containsValue("testInt"), false);
    
    // Test clearing getter
    registerGetter("clearGetter", this, &SettingsStorageTest::intGetter);
    QVERIFY(containsValue("clearGetter"));
    
    clearValue("clearGetter");
    QCOMPARE(containsValue("clearGetter"), false);
    
    // Test clearing non-existent key (should not crash)
    clearValue("nonExistentKey");
    
    // Test clearing array key
    QVERIFY(containsArray("testArray"));
    clearValue("testArray");
    QCOMPARE(containsArray("testArray"), false);
    
    SettingsStorage readOnly2;
    QCOMPARE(readOnly2.containsArray("testArray"), false);
}

void SettingsStorageTest::testConstructorVariants()
{
    initSettingsFile();
    
    // Test empty key constructor (uses default org/app name settings)
    SettingsStorage emptyKey("");
    // This will be in a different settings location than our test data, so should be empty
    QCOMPARE(emptyKey.keys().size(), 0);
    
    // Test Hardware type with single key
    SettingsStorage hardwareType("hardwareKey", SettingsStorage::Hardware);
    QCOMPARE(hardwareType.get<QString>("hardwareName"), QString("My Hardware"));
    
    // Test single key constructor - should have empty keys initially
    SettingsStorage singleKey("testGroup");
    QCOMPARE(singleKey.keys().size(), 0); // New group should be empty
    
    // Test QStringList constructor
    SettingsStorage stringList(QStringList{"group1", "group2"});
    QCOMPARE(stringList.keys().size(), 0); // New nested group should be empty
    
    // Verify the hardware constructor found the correct group
    QCOMPARE(hardwareType.get<int>("hardwareInt"), 10);
    QCOMPARE(hardwareType.get<double>("hardwareDouble"), 44.4);
    
    // Test that we can write to different constructor types (using friend access)
    singleKey.set("testKey1", QString("value1"), true);
    stringList.set("testKey2", 42, true);
    
    // Verify settings were written to correct locations
    QSettings s;
    s.beginGroup("testGroup");
    QCOMPARE(s.value("testKey1").toString(), QString("value1"));
    s.endGroup();
    
    s.beginGroup("group1");
    s.beginGroup("group2");
    QCOMPARE(s.value("testKey2").toInt(), 42);
    s.endGroup();
    s.endGroup();
}

void SettingsStorageTest::testGroupOperations()
{
    // Test basic group operations
    QString groupKey = "testGroup";
    QString key1 = "key1";
    QString key2 = "key2";
    
    // Set values in a group
    QVERIFY(setGroupValue(groupKey, key1, QString("value1")));
    QVERIFY(setGroupValue(groupKey, key2, 42));
    
    // Get values from group
    QCOMPARE(getGroupValue(groupKey, key1, QString("default")), QString("value1"));
    QCOMPARE(getGroupValue<int>(groupKey, key2, 0), 42);
    
    // Test default values for non-existent keys
    QCOMPARE(getGroupValue(groupKey, "nonExistent", QString("default")), QString("default"));
    QCOMPARE(getGroupValue<int>("nonExistentGroup", key1, 99), 99);
    
    // Get entire group
    auto group = getGroup(groupKey);
    QCOMPARE(group.size(), 2);
    QVERIFY(group.find(key1) != group.end());
    QVERIFY(group.find(key2) != group.end());
    QCOMPARE(group.at(key1).toString(), QString("value1"));
    QCOMPARE(group.at(key2).toInt(), 42);
    
    // Test empty group
    auto emptyGroup = getGroup("nonExistentGroup");
    QVERIFY(emptyGroup.empty());
}

void SettingsStorageTest::testGroupValueSetGet()
{
    QString groupKey = "typeTestGroup";
    
    // Test different types
    QVERIFY(setGroupValue(groupKey, "string", QString("testString")));
    QVERIFY(setGroupValue(groupKey, "int", 123));
    QVERIFY(setGroupValue(groupKey, "double", 45.67));
    QVERIFY(setGroupValue(groupKey, "bool", true));
    QVERIFY(setGroupValue(groupKey, "enum", TestValue3));
    
    // Verify retrieval with correct types
    QCOMPARE(getGroupValue<QString>(groupKey, "string"), QString("testString"));
    QCOMPARE(getGroupValue<int>(groupKey, "int"), 123);
    QCOMPARE(getGroupValue<double>(groupKey, "double"), 45.67);
    QCOMPARE(getGroupValue<bool>(groupKey, "bool"), true);
    QCOMPARE(getGroupValue<TestEnum>(groupKey, "enum"), TestValue3);
    
    // Test with default values
    QCOMPARE(getGroupValue<QString>(groupKey, "missing", QString("default")), QString("default"));
    QCOMPARE(getGroupValue<int>(groupKey, "missing", 999), 999);
}

void SettingsStorageTest::testGroupConflicts()
{
    QString conflictKey = "conflictKey";
    
    // Set a regular value
    QVERIFY(set(conflictKey, QString("regularValue")));
    
    // Try to create a group with the same key - should fail
    QVERIFY(!setGroupValue(conflictKey, "subkey", QString("groupValue")));
    
    // Verify regular value is still there
    QCOMPARE(get<QString>(conflictKey), QString("regularValue"));
    
    // Test conflict with getter
    registerGetter("getterKey", this, &SettingsStorageTest::intGetter);
    QVERIFY(!setGroupValue("getterKey", "subkey", QString("value")));
    
    // Test conflict with array
    SettingsMap arrayMap;
    arrayMap["arraySubkey"] = QString("arrayValue");
    setArray("arrayKey", {arrayMap});
    QVERIFY(!setGroupValue("arrayKey", "subkey", QString("value")));
    
    // Test reverse conflicts - regular value shouldn't work if group exists
    QVERIFY(setGroupValue("groupFirst", "subkey", QString("value")));
    QVERIFY(!set("groupFirst", QString("regularValue")));
}

void SettingsStorageTest::testGroupMultipleValues()
{
    QString groupKey = "multiGroup";
    
    // Create a map of values to set
    SettingsMap values;
    values["key1"] = QString("value1");
    values["key2"] = 42;
    values["key3"] = 3.14;
    values["key4"] = true;
    
    // Set multiple values at once
    auto results = setGroupValues(groupKey, values);
    
    // All should succeed
    QCOMPARE(results.size(), 4);
    for(const auto& result : results) {
        QVERIFY(result.second);
    }
    
    // Verify all values were set
    QCOMPARE(getGroupValue<QString>(groupKey, "key1"), QString("value1"));
    QCOMPARE(getGroupValue<int>(groupKey, "key2"), 42);
    QCOMPARE(getGroupValue<double>(groupKey, "key3"), 3.14);
    QCOMPARE(getGroupValue<bool>(groupKey, "key4"), true);
    
    // Get the entire group and verify
    auto group = getGroup(groupKey);
    QCOMPARE(group.size(), 4);
    
    // Test clearValue removes groups
    clearValue(groupKey);
    auto clearedGroup = getGroup(groupKey);
    QVERIFY(clearedGroup.empty());
    QCOMPARE(getGroupValue<QString>(groupKey, "key1", QString("default")), QString("default"));
}

void SettingsStorageTest::testGroupKeys()
{
    initSettingsFile();
    clearGetters(false);
    readAll();

    // Initially no groups
    QVERIFY(groupKeys().isEmpty());

    // Add some groups
    QVERIFY(setGroupValue("groupA", "k1", 1));
    QVERIFY(setGroupValue("groupB", "k1", 2));
    QVERIFY(setGroupValue("groupC", "k1", 3));

    auto gk = groupKeys();
    QCOMPARE(gk.size(), 3);
    QVERIFY(gk.contains("groupA"));
    QVERIFY(gk.contains("groupB"));
    QVERIFY(gk.contains("groupC"));

    // Adding more values to an existing group should not add a new group key
    QVERIFY(setGroupValue("groupA", "k2", 10));
    QCOMPARE(groupKeys().size(), 3);

    // Clearing a group should remove it from groupKeys
    clearValue("groupB");
    auto gk2 = groupKeys();
    QCOMPARE(gk2.size(), 2);
    QVERIFY(!gk2.contains("groupB"));
    QVERIFY(gk2.contains("groupA"));
    QVERIFY(gk2.contains("groupC"));
}

void SettingsStorageTest::testCrossContamination()
{
    initSettingsFile();
    clearGetters(false);
    readAll();

    // Set up one of each type
    set("regularKey", 42, false);
    setArray("arrayKey", {{{"ak", 1}}}, false);
    QVERIFY(setGroupValue("groupKey", "gk", 99));

    // Verify each type appears only in its own key list
    auto k = keys();
    auto ak = arrayKeys();
    auto gk = groupKeys();

    // Regular keys should not contain array or group keys
    QVERIFY(k.contains("regularKey"));
    QVERIFY(!k.contains("arrayKey"));
    QVERIFY(!k.contains("groupKey"));

    // Array keys should not contain regular or group keys
    QVERIFY(ak.contains("arrayKey"));
    QVERIFY(!ak.contains("regularKey"));
    QVERIFY(!ak.contains("groupKey"));

    // Group keys should not contain regular or array keys
    QVERIFY(gk.contains("groupKey"));
    QVERIFY(!gk.contains("regularKey"));
    QVERIFY(!gk.contains("arrayKey"));

    // containsValue should only be true for regular values and getters
    QVERIFY(containsValue("regularKey"));
    QVERIFY(!containsValue("groupKey"));

    // containsArray should only be true for arrays
    QVERIFY(containsArray("arrayKey"));
    QVERIFY(!containsArray("regularKey"));
    QVERIFY(!containsArray("groupKey"));

    // Clearing one type should not affect the others
    clearValue("regularKey");
    QVERIFY(!containsValue("regularKey"));
    QVERIFY(containsArray("arrayKey"));
    QCOMPARE(getGroupValue<int>("groupKey", "gk", 0), 99);

    clearValue("arrayKey");
    QVERIFY(!containsArray("arrayKey"));
    QCOMPARE(getGroupValue<int>("groupKey", "gk", 0), 99);

    clearValue("groupKey");
    QVERIFY(groupKeys().isEmpty());
}

QTEST_MAIN(SettingsStorageTest)

#include "tst_settingsstoragetest.moc"
