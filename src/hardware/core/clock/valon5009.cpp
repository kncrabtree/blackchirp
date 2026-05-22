#include "valon5009.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(Valon5009, "Valon Technology 5009 Dual Channel Synthesizer (500-6000 MHz)")
REGISTER_HARDWARE_PROTOCOLS(Valon5009, CommunicationProtocol::Rs232)
REGISTER_COMM_DEFAULTS(Valon5009, CommunicationProtocol::Rs232,
    {BC::Key::Comm::timeout, 500},
    {BC::Key::Comm::termChar, QString("\n\r")})
REGISTER_HARDWARE_SETTINGS(Valon5009,
    {BC::Key::Clock::minFreq, "Min Frequency (MHz)", "Minimum output frequency in MHz", 500.0, 0.0, QVariant{}, HwSettingPriority::Important},
    {BC::Key::Clock::maxFreq, "Max Frequency (MHz)", "Maximum output frequency in MHz", 6000.0, 0.0, QVariant{}, HwSettingPriority::Important},
    {BC::Key::Clock::lock, "Requires External Lock", "Clock references an external 10 MHz lock signal.", true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

Valon5009::Valon5009(const QString& label, QObject *parent) :
    Clock(2, true, QString(Valon5009::staticMetaObject.className()), label, parent)
{
    save();
}


bool Valon5009::testClockConnection()
{
    QByteArray resp = valonQueryCmd(QString("ID\r"));

    if(resp.isEmpty())
    {
        d_errorString = QString("No response to ID query.");
        return false;
    }

    if(!resp.startsWith("Valon Technology, 5009"))
    {
        d_errorString = QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp.trimmed())).arg(QString(resp.toHex()));
        return false;
    }

    hwDebug(u"ID response: %1"_s.arg(QString(resp)));

    return true;
}

void Valon5009::initializeClock()
{
}


bool Valon5009::valonWriteCmd(const QString &cmd)
{
    if(!p_comm->writeCmd(cmd))
        return false;

    QByteArray resp = p_comm->queryCmd(QString(""));
    if(resp.isEmpty())
        return false;
    if(!resp.contains(cmd.toLatin1()))
    {
        emit hardwareFailure();
        hwWarn("Did not receive command echo."_L1);
        hwDebug(u"Did not receive command echo. Command = %1, Echo = %2"_s.arg(cmd, QString(resp)));
        return false;
    }

    return true;

}

QByteArray Valon5009::valonQueryCmd(const QString &cmd)
{
    QByteArray resp = p_comm->queryCmd(cmd);
    resp = resp.trimmed();
    while(true)
    {
        if(resp.startsWith("-") || resp.startsWith(">") || resp.startsWith("1") || resp.startsWith("2"))
            resp = resp.mid(1);
        else
            break;
    }
    if(resp.startsWith(cmd.toLatin1()))
        resp.replace(cmd.toLatin1(),QByteArray());
    return resp.trimmed();
}

bool Valon5009::setHwFrequency(double freqMHz, int outputIndex)
{
    auto source = QString("S%1").arg(outputIndex+1);
    return valonWriteCmd(source+QString("; Frequency %2M\r").arg(freqMHz,0,'f',6));
}

double Valon5009::readHwFrequency(int outputIndex)
{
    auto source = QString("S%1").arg(outputIndex+1);
    auto cmd = source + QString("; Frequency?\r");

    QByteArray resp = valonQueryCmd(cmd);
    if(resp.isEmpty())
    {
        emit hardwareFailure();
        hwError(u"Could not read %1 frequency."_s.arg(channelNames().at(outputIndex)));
        hwDebug(u"Could not read %1 frequency. No response received."_s.arg(channelNames().at(outputIndex)));
        return -1.0;
    }

    if(!resp.startsWith("F"))
    {
        hwError(u"Could not read %1 frequency."_s.arg(channelNames().at(outputIndex)));
        hwDebug(u"Could not read %1 frequency. Response = %2 (Hex: %3)"_s
                .arg(channelNames().at(outputIndex), QString(resp), QString(resp.toHex())));
        return -1.0;
    }
    QByteArrayList l = resp.split(' ');
    if(l.size() < 2)
    {
        hwError(u"Could not parse %1 frequency response."_s.arg(channelNames().at(outputIndex)));
        hwDebug(u"Could not parse %1 frequency response. Response = %2 (Hex: %3)"_s
                .arg(channelNames().at(outputIndex), QString(resp), QString(resp.toHex())));
        return -1.0;
    }

    bool ok = false;
    double f = l.at(1).trimmed().toDouble(&ok);
    if(!ok)
    {
        hwError(u"Could not convert %1 frequency to number."_s.arg(channelNames().at(outputIndex)));
        hwDebug(u"Could not convert %1 frequency to number. Response = %2 (Hex: %3)"_s
                .arg(channelNames().at(outputIndex), QString(l.at(1).trimmed()), QString(l.at(1).trimmed().toHex())));
        return -1.0;
    }

    return f;
}


bool Valon5009::prepareClock(Experiment &exp)
{
    if(get<bool>(BC::Key::Clock::lock))
    {
        valonWriteCmd(QString("REFS 1\r"));
        valonWriteCmd(QString("REF 10 MHz\r"));
        auto resp = valonQueryCmd(QString("LOCK?\r"));
        if(resp.contains("not locked"))
        {
            exp.d_errorString = QString("Could not lock to external reference.");
            return false;
        }
    }
    else
    {
        valonWriteCmd(QString("REFS 0\r"));
        valonWriteCmd(QString("REF 20 MHz\r"));
        auto resp = valonQueryCmd(QString("LOCK?\r"));
        if(resp.contains("not locked"))
        {
            exp.d_errorString = QString("Could not lock to internal reference.");
            return false;
        }
    }

    return true;
}
