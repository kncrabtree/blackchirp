#include <QtTest>

#include <src/data/storage/headerstorage.h>
#include <src/data/storage/blackchirpcsv.h>

#include <iostream>

class HST : public HeaderStorage
{
public:
    HST(const QString name) : HeaderStorage(name) {}

    // HeaderStorage interface
protected:
    void prepareToSave() override {}
    void loadComplete() override {}
};

class Child : public HeaderStorage
{
    Q_GADGET
public:
    Child(const QString name) : HeaderStorage(name) {}

    enum ChildEnum {
        Child1,
        Child2,
        Child3,
        Child4
    };
    Q_ENUM(ChildEnum)

    // HeaderStorage interface
protected:
    void prepareToSave() override;
    void loadComplete() override;

private:
    friend class HeaderStorageTest;

    int d_myInt{20};
    QString d_myString{"Hello!"};
    ChildEnum d_myEnum{Child2};
    QStringList d_myStringList{"Str0", "Str1", "Str2", "Str3"};
};

void Child::prepareToSave()
{
    store("childInt",d_myInt);
    store("childString",d_myString);
    store("childEnum",d_myEnum);

    for(int i = 0; i <d_myStringList.size(); ++i)
        storeArrayValue("childArray",i,"childArrayString",d_myStringList.at(i));

}

void Child::loadComplete()
{
    d_myInt = retrieve("childInt",0);
    d_myString = retrieve("childString",QString(""));
    d_myEnum = retrieve("childEnum",Child1);

    d_myStringList.clear();
    auto n = arrayStoreSize("childArray");
    d_myStringList.reserve(n);
    for(std::size_t i=0; i<n; ++i)
        d_myStringList.append(retrieveArrayValue("childArray",i,"childArrayString",QString("")));
}

class HeaderStorageTest : public QObject, public HST
{
    Q_OBJECT
public:
    HeaderStorageTest() : HST("Test"){};
    ~HeaderStorageTest() {};

    enum Test {
        Test1,
        Test2,
        Test3
    };
    Q_ENUM(Test)

    Child c{"ChildTest"};

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testStoreRetrieve();
    void testChild();
    void testStrings();
    void testCSV();
};

void HeaderStorageTest::initTestCase()
{

}

void HeaderStorageTest::cleanupTestCase()
{

}

void HeaderStorageTest::testStoreRetrieve()
{

    store("testInt",10);
    store("testDouble",42.1,"V");
    store("testEnum",Test3);

    storeArrayValue("testArray",0,"testArrayInt",100);
    storeArrayValue("testArray",0,"testArrayDouble",1e-2,"V");

    QCOMPARE(retrieve<int>("testInt"),10);
    QCOMPARE(retrieve<int>("testInt"),0);
    QCOMPARE(retrieve("testInt",30),30);
    QCOMPARE(retrieve("testEnum",Test1),Test3);
    QCOMPARE(retrieve("testDouble",90.0),42.1);

    QCOMPARE(arrayStoreSize("testArray"),1);
    QCOMPARE(retrieveArrayValue("testArray",0,"testArrayInt",10),100);
    QCOMPARE(retrieveArrayValue("testArray",0,"testArrayInt",10),10);
    QCOMPARE(retrieveArrayValue("testArray",0,"testArrayDouble",2e-5),1e-2);
    QCOMPARE(retrieveArrayValue("testArray",0,"testArrayDouble",2e-5),2e-5);

    QVariant v{QString("Test3")};
    QCOMPARE(v.value<Test>(),Test3);
}

void HeaderStorageTest::testChild()
{
    addChild(&c);

    store("testInt",10);
    store("testDouble",42.1,"V");
    store("testEnum",Test3);

    storeArrayValue("testArray",0,"testArrayInt",100);
    storeArrayValue("testArray",0,"testArrayDouble",1e-2,"V");

    int cn = c.d_myStringList.size();
    auto sl = c.d_myStringList;

    auto m = getStrings();
    QList<QStringList> l;
    for(auto it = m.cbegin(); it != m.cend(); it++)
    {
        auto k = it->first;
        auto [s1,s2,s3,s4,s5] = it->second;
        QStringList l2{k,s1,s2,s3,s4,s5};
        l.append(l2);
    }

    for(auto line : l)
        storeLine(line);

    c.d_myInt = 0;
    c.d_myString = "";
    c.d_myEnum = Child::Child1;
    c.d_myStringList.clear();

    readComplete();

    QCOMPARE(c.d_myInt,20);
    QCOMPARE(c.d_myString,QString("Hello!"));
    QCOMPARE(c.d_myEnum,Child::Child2);
    QCOMPARE(c.d_myStringList.size(),cn);
    for(int i = 0; i<cn; ++i)
        QCOMPARE(c.d_myStringList.at(i),sl.at(i));
}

void HeaderStorageTest::testStrings()
{
    store("testInt",10);
    store("testDouble",42.1,"V");
    store("testEnum",Test3);

    storeArrayValue("testArray",0,"testArrayInt",100);
    storeArrayValue("testArray",0,"testArrayDouble",1e-2,"V");

    auto m = getStrings();
    QList<QStringList> l;
    for(auto it = m.cbegin(); it != m.cend(); it++)
    {
        auto k = it->first;
        auto [s1,s2,s3,s4,s5] = it->second;
        QStringList l2{k,s1,s2,s3,s4,s5};
        l.append(l2);
    }

    for(auto line : l)
        storeLine(line);

    QCOMPARE(retrieve<int>("testInt"),10);
    QCOMPARE(retrieve<int>("testInt"),0);
    QCOMPARE(retrieve("testInt",30),30);
    QCOMPARE(retrieve("testEnum",Test1),Test3);
    QCOMPARE(retrieve("testDouble",90.0),42.1);

    QCOMPARE(arrayStoreSize("testArray"),1);
    QCOMPARE(retrieveArrayValue("testArray",0,"testArrayInt",10),100);
    QCOMPARE(retrieveArrayValue("testArray",0,"testArrayInt",10),10);
    QCOMPARE(retrieveArrayValue("testArray",0,"testArrayDouble",2e-5),1e-2);
    QCOMPARE(retrieveArrayValue("testArray",0,"testArrayDouble",2e-5),2e-5);

}

void HeaderStorageTest::testCSV()
{
    store("testInt",10);
    store("testDouble",42.1,"V");
    store("testEnum",Test3);

    storeArrayValue("testArray",0,"testArrayInt",100);
    storeArrayValue("testArray",0,"testArrayDouble",1e-2,"V");

    int cn = c.d_myStringList.size();
    auto sl = c.d_myStringList;

    BlackchirpCSV csv;
    QByteArray b;
    QBuffer f(&b);
    QCOMPARE(csv.writeHeader(f,getStrings()),true);

    f.open(QIODevice::ReadOnly);
    while(!f.atEnd())
    {
        auto l = QString(f.readLine().trimmed()).split(',');
        if(l.size() == 6)
            storeLine(l);
    }

    readComplete();

    QCOMPARE(retrieve<int>("testInt"),10);
    QCOMPARE(retrieve<int>("testInt"),0);
    QCOMPARE(retrieve("testInt",30),30);
    QCOMPARE(retrieve("testEnum",Test1),Test3);
    QCOMPARE(retrieve("testDouble",90.0),42.1);

    QCOMPARE(c.d_myInt,20);
    QCOMPARE(c.d_myString,QString("Hello!"));
    QCOMPARE(c.d_myEnum,Child::Child2);
    QCOMPARE(c.d_myStringList.size(),cn);
    for(int i = 0; i<cn; ++i)
        QCOMPARE(c.d_myStringList.at(i),sl.at(i));

    QTextStream t(stdout);
    t << "\n\n" << b << "\n\n";

    t << staticMetaObject.className() << "\n\n";
}

QTEST_MAIN(HeaderStorageTest)

#include "tst_headerstoragetest.moc"
