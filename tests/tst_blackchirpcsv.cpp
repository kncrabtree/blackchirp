#include <QtTest>

#include <src/data/storage/blackchirpcsv.h>

class BlackchirpCSVTest : public QObject
{
    Q_OBJECT
public:
    BlackchirpCSVTest() {};
    ~BlackchirpCSVTest() {};


private slots:
    void initTestCase();
    void cleanupTestCase();
    void testExportXY();
    void testExportMultiple();
    void testExportMultipleDiffLength();
    void testExportY();
    void testExportYMulitple();
    void testFidConversion();
};

void BlackchirpCSVTest::initTestCase()
{

}

void BlackchirpCSVTest::cleanupTestCase()
{

}

void BlackchirpCSVTest::testExportXY()
{
    QVector<QPointF> d;
    d.reserve(10);
    for(int i=0; i<10; ++i)
    {
        double x = i;
        d.append( {x,sin(x)} );
    }

    BlackchirpCSV csv;
    QByteArray b;
    QBuffer f(&b);
    f.open(QIODevice::WriteOnly|QIODevice::Text);
    QCOMPARE(csv.writeXY(f,d),true);
    f.close();

    QCOMPARE(b.startsWith("x;y"),true);

    QTextStream t(stdout);
    t << "\n\n" << b << "\n\n";

    b.clear();

    f.open(QIODevice::WriteOnly|QIODevice::Text);
    QCOMPARE(csv.writeXY(f,d,"sin"),true);
    f.close();
    QCOMPARE(b.startsWith("sin_x;sin_y"),true);
    t << "\n\n" << b << "\n\n";
}

void BlackchirpCSVTest::testExportMultiple()
{
    QVector<QPointF> dsin, dcos;
    dsin.reserve(10), dcos.reserve(10);

    for(int i=0; i<10; ++i)
    {
        double x = i;
        dsin.append( {x,sin(x)} );
        dcos.append( {x,cos(x)} );
    }

    BlackchirpCSV csv;
    QByteArray b;
    QBuffer f(&b);
    f.open(QIODevice::WriteOnly|QIODevice::Text);
    QCOMPARE(csv.writeMultiple(f,{dsin,dcos},{}),true);
    f.close();
    QCOMPARE(b.startsWith("x0;y0;x1;y1"),true);

    QTextStream t(stdout);
    t << "\n\n" << b << "\n\n";

    b.clear();

    f.open(QIODevice::WriteOnly|QIODevice::Text);
    QCOMPARE(csv.writeMultiple(f,{dsin,dcos},{"sin","cos"}),true);
    f.close();
    QCOMPARE(b.startsWith("sin_x;sin_y;cos_x;cos_y"),true);

    t << "\n\n" << b << "\n\n";
}

void BlackchirpCSVTest::testExportMultipleDiffLength()
{
    QVector<QPointF> dsin, dcos;
    dsin.reserve(20), dcos.reserve(10);

    for(int i=0; i<10; ++i)
    {
        double x = i;
        dsin.append( {x,sin(x)} );
        dcos.append( {x,cos(x)} );
    }
    for(int i=10; i<20; ++i)
    {
        double x = i;
        dsin.append( {x,sin(x)} );
    }

    BlackchirpCSV csv;
    QByteArray b;
    QBuffer f(&b);
    f.open(QIODevice::WriteOnly|QIODevice::Text);
    QCOMPARE(csv.writeMultiple(f,{dsin,dcos},{}),true);
    f.close();
    QCOMPARE(b.startsWith("x0;y0;x1;y1"),true);

    QTextStream t(stdout);
    t << "\n\n" << b << "\n\n";
}

void BlackchirpCSVTest::testExportY()
{
    QVector<double> y;
    y.reserve(10);

    for(double i=0; i<10; i+=1.0)
        y.append(sin(i));

    BlackchirpCSV csv;
    QByteArray b;
    QBuffer f(&b);
    f.open(QIODevice::WriteOnly|QIODevice::Text);
    QCOMPARE(csv.writeY(f,y,"sin"),true);
    f.close();
    QCOMPARE(b.startsWith("sin"),true);

    QTextStream t(stdout);
    t << "\n\n" << b << "\n\n";

    QVector<QString> y2;
    y2 << "H" << "e" << "l" << "l" << "o" << "!";

    b.clear();
    f.open(QIODevice::WriteOnly|QIODevice::Text);
    QCOMPARE(csv.writeY(f,y2),true);
    f.close();
    QCOMPARE(b.startsWith("y"),true);

    t << "\n\n" << b << "\n\n";

    QVector<int> y3;
    y3 << 12 << 1 << 0xff;

    b.clear();
    f.open(QIODevice::WriteOnly|QIODevice::Text);
    QCOMPARE(csv.writeY(f,y3,"int"),true);
    f.close();
    QCOMPARE(b.startsWith("int"),true);

    t << "\n\n" << b << "\n\n";

}

void BlackchirpCSVTest::testExportYMulitple()
{
    QVector<QString> y1,y2,y3;
    for(int i=0;i<30;i++)
    {
        y3 << QVariant(exp(0.1*(double)i)).toString();
        if(i<10)
            y1 << QVariant(i).toString(); ;
        if(i<20)
            y2 << QVariant("Hello!").toString();
    }

    BlackchirpCSV csv;
    QByteArray b;
    QBuffer f(&b);
    f.open(QIODevice::WriteOnly|QIODevice::Text);
    QCOMPARE(csv.writeYMultiple(f,{"double","int","string"},{y1,y2,y3}),true);
    f.close();

    QTextStream t(stdout);
    t << "\n\n" << b << "\n\n";
}

void BlackchirpCSVTest::testFidConversion()
{
    QTextStream t(stdout);

    qint64 n1,n2,n3;
    n1 = 20;
    n2 = -20;
    n3 = -100;



    t << n1 << ";" << n2 << ";" << n3 << "\n\n";

    t << BlackchirpCSV::formatInt64(n1) << ";" << BlackchirpCSV::formatInt64(n2) << ";" << BlackchirpCSV::formatInt64(n3) << "\n\n";

    t << BlackchirpCSV::formatInt64(n2).toLongLong(nullptr,36) << "\n\n";


}

QTEST_MAIN(BlackchirpCSVTest)

#include "tst_blackchirpcsv.moc"
