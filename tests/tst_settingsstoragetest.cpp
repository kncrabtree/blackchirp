#include <QtTest>
#include <QCoreApplication>

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
    void testSubkeyRead();
    void testHardwareRead();
    void testDestruction();


private:
    int d_int = 10;
    double d_double = 10.1;

    void initSettingsFile();

};

class DestructorTest : public SettingsStorage
{
public:
    DestructorTest() : SettingsStorage("CrabtreeLab","BlackchirpTest",{},General) {
        registerGetter("destructTest",this,&DestructorTest::desructGetter);
    }
    virtual ~DestructorTest(){};

    int desructGetter() const { return d_test; }

private:
    int d_test = 144;
};

Q_DECLARE_METATYPE(SettingsStorageTest::TestEnum)

SettingsStorageTest::SettingsStorageTest() : SettingsStorage("CrabtreeLab","BlackchirpTest",{},General)
{
    QCoreApplication::setApplicationName("BlackchirpTest");
    QCoreApplication::setOrganizationName("CrabtreeLab");
    QCoreApplication::setOrganizationDomain("crabtreelab.ucdavis.edu");

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

void SettingsStorageTest::testSubkeyRead()
{
    initSettingsFile();
    SettingsStorage readOnly({"readOnly","subKey"},General);

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
    //clear out any existing settings
    QSettings s;
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

    //Write settings for a readonly test with subkeys
    s.beginGroup("readOnly");
    s.beginGroup("subKey");
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

    //Write settings for hardware
    s.beginGroup("hardwareKey");
    s.setValue("subKey","hardwareSubKey");
    s.beginGroup("hardwareSubKey");
    s.setValue("hardwareInt",10);
    s.setValue("hardwareName","My Hardware");
    s.setValue("hardwareDouble",44.4);
    s.endGroup();
    s.endGroup();

    s.sync();
}

QTEST_MAIN(SettingsStorageTest)

#include "tst_settingsstoragetest.moc"
