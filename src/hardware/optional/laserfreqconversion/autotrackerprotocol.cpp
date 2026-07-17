#include <hardware/optional/laserfreqconversion/autotrackerprotocol.h>

namespace BC::Autotracker {

QByteArray buildCommand(quint8 cmd, const QByteArray &data)
{
    QByteArray out(12, '\0');
    out[0] = static_cast<char>(0x3E);
    out[1] = static_cast<char>(0x00);
    out[2] = static_cast<char>(cmd);

    for(int i=0; i<data.size() && i<8; i++)
        out[3+i] = data.at(i);

    quint8 checksum = 0;
    for(int i=0; i<11; i++)
        checksum = static_cast<quint8>(checksum + static_cast<quint8>(out.at(i)));
    out[11] = static_cast<char>(checksum);

    return out;
}

bool parseResponse(const QByteArray &resp, Response &out)
{
    if(resp.size() != 12 || static_cast<quint8>(resp.at(0)) != 0x3C)
        return false;

    quint8 checksum = 0;
    for(int i=0; i<11; i++)
        checksum = static_cast<quint8>(checksum + static_cast<quint8>(resp.at(i)));
    if(checksum != static_cast<quint8>(resp.at(11)))
        return false;

    out.adrStatus = static_cast<quint8>(resp.at(1));
    out.id = static_cast<quint8>(resp.at(2));
    out.payload = resp.mid(3,8);

    return true;
}

QByteArray packPos24(quint32 pos)
{
    QByteArray out(3, '\0');
    out[0] = static_cast<char>((pos >> 16) & 0xFF);
    out[1] = static_cast<char>((pos >> 8) & 0xFF);
    out[2] = static_cast<char>(pos & 0xFF);
    return out;
}

quint32 unpackPos24(const QByteArray &b, int offset)
{
    if(offset < 0 || b.size() < offset+3)
        return 0;

    quint32 pos = static_cast<quint32>(static_cast<quint8>(b.at(offset))) << 16;
    pos |= static_cast<quint32>(static_cast<quint8>(b.at(offset+1))) << 8;
    pos |= static_cast<quint32>(static_cast<quint8>(b.at(offset+2)));
    return pos;
}

QString errorString(quint8 code)
{
    switch(code)
    {
    case 0:  return QString("No Error");
    case 1:  return QString("Start Character error");
    case 2:  return QString("Checksum error");
    case 3:  return QString("Watchdog error");
    case 4:  return QString("Command Code error");
    case 5:  return QString("Command Break error");
    case 6:  return QString("Stack Overflow error");
    case 7:  return QString("Stack Underflow error");
    case 8:  return QString("AS Device error (bad device number)");
    case 9:  return QString("AS Inactive error (device inactive)");
    case 10: return QString("AS Register error (bad register)");
    case 11: return QString("AS LAM Source error");
    case 12: return QString("AS LAM Code error");
    case 13: return QString("AS Client error (device error)");
    case 14: return QString("Motor Number error");
    case 15: return QString("Motor Inactive error");
    case 16: return QString("Motor Parameter error");
    case 17: return QString("LED Parameter error");
    case 18: return QString("Trace Parameter error");
    case 19: return QString("Slope Parameter error");
    case 20: return QString("Pre-Trigger Parameter error");
    case 21: return QString("Pre-Trigger required error");
    case 22: return QString("Trigger Level required error");
    case 23: return QString("EXIO Parameter error");
    case 24: return QString("Invert Parameter error");
    case 25: return QString("Average Range error");
    case 26: return QString("Timeout Occurred");
    case 27: return QString("Timeout Parameter error");
    case 28: return QString("Gain Parameter error");
    case 29: return QString("Channel Parameter error");
    case 30: return QString("Auto Timeout Parameter error");
    case 31: return QString("External Trigger Parameter error");
    case 32: return QString("Serial Buffer Overflow");
    default: return QString("Unknown Autotracker error code %1").arg(code);
    }
}

}
