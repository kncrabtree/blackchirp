#include <hardware/core/liflaser/sirahprotocol.h>

namespace BC::Sirah {

QByteArray buildCommand(char cmd, const QByteArray &args)
{
    QByteArray out;
    out.fill(0x00,13);
    out[0] = 0x3c;
    out[1] = cmd;
    out[11] = out.at(11) + out.at(0);
    out[11] = out.at(11) + out.at(1);
    for(int i=0; i<args.size() && i<9; i++)
    {
        out[i+2] = args.at(i);
        out[11] = out.at(11) + args.at(i);
    }
    out[12] = 0x3e;

    return out;
}

bool parseStatus(const QByteArray &resp, Status &out)
{
    if(resp.size() != 14 || !resp.startsWith(0x5b) || !resp.endsWith(0x5d))
        return false;

    out.err = static_cast<quint8>(resp.at(1));
    out.cStatus = static_cast<quint8>(resp.at(2));
    out.m1Status = static_cast<quint8>(resp.at(3));
    qint32 pos = 0;
    pos |= static_cast<quint8>(resp.at(4));
    pos |= (static_cast<quint8>(resp.at(5)) << 8);
    pos |= (static_cast<quint8>(resp.at(6)) << 16);
    pos |= (static_cast<quint8>(resp.at(7)) << 24);
    out.m1Pos = pos;
    out.m2Status = static_cast<quint8>(resp.at(8));
    pos = 0;
    pos |= static_cast<quint8>(resp.at(9));
    pos |= (static_cast<quint8>(resp.at(10)) << 8);
    pos |= (static_cast<quint8>(resp.at(11)) << 16);
    pos |= (static_cast<quint8>(resp.at(12)) << 24);
    out.m2Pos = pos;

    return true;
}

}
