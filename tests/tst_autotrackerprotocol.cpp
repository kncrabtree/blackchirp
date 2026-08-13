#include <QtTest>

#include <hardware/optional/laserfreqconversion/autotrackerprotocol.h>

using namespace Qt::Literals::StringLiterals;
using namespace BC::Autotracker;

class AutotrackerProtocolTest : public QObject
{
    Q_OBJECT

private slots:
    // buildCommand()
    void testBuildCommandGetPosition();
    void testBuildCommandGotoPosition();
    void testBuildCommandNoData();
    void testBuildCommandTruncatesLongData();

    // parseResponse()
    void testParseResponseGetPositionAccept();
    void testParseResponseGotoAckAccept();
    void testParseResponseRejectsBadStartByte();
    void testParseResponseRejectsWrongLength();
    void testParseResponseRejectsBadChecksum();
    void testParseResponseLeavesOutUnmodifiedOnFailure();

    // pack/unpack 24-bit position
    void testPackPos24RoundTrip();
    void testUnpackPos24BenchValue();
    void testUnpackPos24ShortBufferReturnsZero();
    void testPackPos24DiscardsHighBits();

    // errorString()
    void testErrorStringKnownCodes();
    void testErrorStringUnknownCode();
};

void AutotrackerProtocolTest::testBuildCommandGetPosition()
{
    // Bench-captured Get Position command (§4.6): motor 1, checksum 0x56.
    QByteArray dat(1, static_cast<char>(0x01));
    auto cmd = buildCommand(0x17, dat);

    QByteArray expected = QByteArray::fromHex("3E0017010000000000000056");
    QCOMPARE(cmd, expected);
}

void AutotrackerProtocolTest::testBuildCommandGotoPosition()
{
    // Bench-captured Goto Position command (§4.6): motor 1, Wait=0, Rel=0,
    // target 0xFAE9BD, checksum 0x01.
    QByteArray dat;
    dat.append(static_cast<char>(0x01)); // motor
    dat.append(static_cast<char>(0x00)); // Wait
    dat.append(static_cast<char>(0x00)); // Rel
    dat.append(packPos24(0xFAE9BD));

    auto cmd = buildCommand(0x22, dat);

    QByteArray expected = QByteArray::fromHex("3E0022010000FAE9BD000001");
    QCOMPARE(cmd, expected);
}

void AutotrackerProtocolTest::testBuildCommandNoData()
{
    // Identify: 0x3E, 0x00, 0x02, all-zero payload, checksum = 0x3E+0x02 = 0x40.
    auto cmd = buildCommand(0x02);
    QCOMPARE(cmd.size(), 12);
    QCOMPARE(static_cast<quint8>(cmd.at(0)), quint8(0x3E));
    QCOMPARE(static_cast<quint8>(cmd.at(1)), quint8(0x00));
    QCOMPARE(static_cast<quint8>(cmd.at(2)), quint8(0x02));
    for(int i=3; i<11; i++)
        QCOMPARE(static_cast<quint8>(cmd.at(i)), quint8(0x00));
    QCOMPARE(static_cast<quint8>(cmd.at(11)), quint8(0x40));
}

void AutotrackerProtocolTest::testBuildCommandTruncatesLongData()
{
    // Only the first 8 bytes of data are transmitted; the rest is dropped
    // rather than overflowing the 12-byte frame.
    QByteArray dat(10, static_cast<char>(0x11));
    auto cmd = buildCommand(0x01, dat);

    QCOMPARE(cmd.size(), 12);
    for(int i=3; i<11; i++)
        QCOMPARE(static_cast<quint8>(cmd.at(i)), quint8(0x11));
}

void AutotrackerProtocolTest::testParseResponseGetPositionAccept()
{
    // Bench-captured Get Position response (§4.6).
    auto resp = QByteArray::fromHex("3C810B01FAE9BD0000000069");

    Response out;
    QVERIFY(parseResponse(resp, out));
    QCOMPARE(out.adrStatus, quint8(0x81));
    QCOMPARE(out.id, quint8(0x0B));
    QCOMPARE(out.payload.size(), 8);
    QCOMPARE(static_cast<quint8>(out.payload.at(0)), quint8(0x01));
    QCOMPARE(unpackPos24(out.payload, 1), quint32(0xFAE9BD));
}

void AutotrackerProtocolTest::testParseResponseGotoAckAccept()
{
    // Bench-captured Goto Position ack (§4.6): a standard ID=0x00 ack whose
    // payload is not a position field.
    auto resp = QByteArray::fromHex("3C810001FB150900000000D7");

    Response out;
    QVERIFY(parseResponse(resp, out));
    QCOMPARE(out.adrStatus, quint8(0x81));
    QCOMPARE(out.id, quint8(0x00));
}

void AutotrackerProtocolTest::testParseResponseRejectsBadStartByte()
{
    auto resp = QByteArray::fromHex("3D810B01FAE9BD0000000069"); // 0x3D instead of 0x3C
    Response out;
    QVERIFY(!parseResponse(resp, out));
}

void AutotrackerProtocolTest::testParseResponseRejectsWrongLength()
{
    auto resp = QByteArray::fromHex("3C810B01FAE9BD00000000"); // 11 bytes
    Response out;
    QVERIFY(!parseResponse(resp, out));
}

void AutotrackerProtocolTest::testParseResponseRejectsBadChecksum()
{
    auto resp = QByteArray::fromHex("3C810B01FAE9BD0000000000"); // wrong checksum (00 instead of 69)
    Response out;
    QVERIFY(!parseResponse(resp, out));
}

void AutotrackerProtocolTest::testParseResponseLeavesOutUnmodifiedOnFailure()
{
    Response out;
    out.adrStatus = 0x42;
    out.id = 0x42;
    out.payload = QByteArray::fromHex("DEADBEEF");

    auto bad = QByteArray::fromHex("00");
    QVERIFY(!parseResponse(bad, out));

    QCOMPARE(out.adrStatus, quint8(0x42));
    QCOMPARE(out.id, quint8(0x42));
    QCOMPARE(out.payload, QByteArray::fromHex("DEADBEEF"));
}

void AutotrackerProtocolTest::testPackPos24RoundTrip()
{
    const quint32 values[] = {0, 1, 0x7FFFFF, 0x800000, 0xFFFFFF, 16443837};
    for(auto v : values)
    {
        auto packed = packPos24(v);
        QCOMPARE(packed.size(), 3);
        QCOMPARE(unpackPos24(packed), v);
    }
}

void AutotrackerProtocolTest::testUnpackPos24BenchValue()
{
    // §4.6: FA E9 BD decodes big-endian to 16,443,837.
    auto bytes = QByteArray::fromHex("FAE9BD");
    QCOMPARE(unpackPos24(bytes), quint32(16443837));
}

void AutotrackerProtocolTest::testUnpackPos24ShortBufferReturnsZero()
{
    QCOMPARE(unpackPos24(QByteArray::fromHex("FAE9")), quint32(0));
    QCOMPARE(unpackPos24(QByteArray()), quint32(0));
    QCOMPARE(unpackPos24(QByteArray::fromHex("00FAE9BD"), 2), quint32(0)); // only 2 bytes remain at offset 2
}

void AutotrackerProtocolTest::testPackPos24DiscardsHighBits()
{
    // Only the low 24 bits are transmitted.
    auto packed = packPos24(0xFFFAE9BD);
    QCOMPARE(unpackPos24(packed), quint32(0xFAE9BD));
}

void AutotrackerProtocolTest::testErrorStringKnownCodes()
{
    QVERIFY(errorString(0).contains(u"No Error"_s));
    QVERIFY(errorString(2).contains(u"Checksum"_s));
    QVERIFY(errorString(7).contains(u"Stack Underflow"_s));
    QVERIFY(errorString(14).contains(u"Motor Number"_s));
    QVERIFY(errorString(26).contains(u"Timeout Occurred"_s));
    QVERIFY(errorString(32).contains(u"Serial Buffer Overflow"_s));
}

void AutotrackerProtocolTest::testErrorStringUnknownCode()
{
    auto msg = errorString(200);
    QVERIFY(msg.contains(u"200"_s));
}

QTEST_APPLESS_MAIN(AutotrackerProtocolTest)

#include "tst_autotrackerprotocol.moc"
