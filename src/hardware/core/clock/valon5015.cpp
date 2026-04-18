#include "valon5015.h"
#include <hardware/core/hardwareregistration.h>

// Register hardware implementation
REGISTER_HARDWARE_META(Valon5015, "Valon Technology 5015 Single Channel Synthesizer (500-15000 MHz)")
REGISTER_HARDWARE_PROTOCOLS(Valon5015, CommunicationProtocol::Rs232)
REGISTER_HARDWARE_SETTINGS(Valon5015,
    {BC::Key::Clock::minFreq, "Min Frequency (MHz)", "Minimum output frequency in MHz", 500.0, 0.0, QVariant{}, HwSettingPriority::Important},
    {BC::Key::Clock::maxFreq, "Max Frequency (MHz)", "Maximum output frequency in MHz", 15000.0, 0.0, QVariant{}, HwSettingPriority::Important},
    {BC::Key::Clock::lock, "Requires External Lock", "Clock references an external 10 MHz lock signal.", false, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

Valon5015::Valon5015(const QString& label, QObject *parent) :
    Clock(1, true, QString(Valon5015::staticMetaObject.className()), label, parent)
{
    // Communication defaults
    setDefault(BC::Key::Comm::timeout, 500);
    setDefault(BC::Key::Comm::termChar, QString("\n\r"));

    save();
}

bool Valon5015::testClockConnection()
{
    QByteArray resp = valonQueryCmd(QString("ID\r"));

    if(resp.isEmpty())
    {
        d_errorString = QString("No response to ID query.");
        return false;
    }

    if(!resp.startsWith("Valon Technology, 5015"))
    {
        d_errorString = QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp.trimmed())).arg(QString(resp.toHex()));
        return false;
    }

    hwDebug(u"ID response: %1"_s.arg(QString(resp)));

    return true;
}

void Valon5015::initializeClock()
{
}

bool Valon5015::prepareClock(Experiment &exp)
{
    valonWriteCmd(QString("PWR 13\r"));
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
        valonWriteCmd(QString("REF 10 MHz\r"));
        auto resp = valonQueryCmd(QString("LOCK?\r"));
        if(resp.contains("not locked"))
        {
            exp.d_errorString = QString("Could not lock to internal reference.");
            return false;
        }
    }

    return true;
}

bool Valon5015::setHwFrequency(double freqMHz, int outputIndex)
{
    Q_UNUSED(outputIndex)

    return valonWriteCmd(QString("F %1M\r").arg(freqMHz,0,'f',6));
}

double Valon5015::readHwFrequency(int outputIndex)
{
    auto cmd = QString("Frequency?\r");

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

bool Valon5015::valonWriteCmd(const QString &cmd)
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

QByteArray Valon5015::valonQueryCmd(const QString &cmd)
{

    QByteArray resp = p_comm->queryCmd(cmd);
    resp = resp.trimmed();
    while(true)
    {
        if(resp.startsWith("-") || resp.startsWith(">"))
            resp = resp.mid(1);
        else
            break;
    }
    if(resp.startsWith(cmd.toLatin1()))
        resp.replace(cmd.toLatin1(),QByteArray());
    return resp.trimmed();
}
