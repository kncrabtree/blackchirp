#include "valon5009.h"

Valon5009::Valon5009(int clockNum, QObject *parent) :
    Clock(clockNum,2,true,BC::Key::valon5009,BC::Key::valon5009Name,CommunicationProtocol::Rs232,parent)
{
    setDefault(BC::Key::Clock::minFreq,500.0);
    setDefault(BC::Key::Clock::maxFreq,6000.0);
    setDefault(BC::Key::Clock::lock,true);
}


bool Valon5009::testConnection()
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

    emit logMessage(QString("ID response: %1").arg(QString(resp)));

    return true;
}

void Valon5009::initializeClock()
{
    p_comm->setReadOptions(500,true,QByteArray("\n\r"));
}


bool Valon5009::valonWriteCmd(QString cmd)
{
    if(!p_comm->writeCmd(cmd))
        return false;

    QByteArray resp = p_comm->queryCmd(QString(""));
    if(resp.isEmpty())
        return false;
    if(!resp.contains(cmd.toLatin1()))
    {
        emit hardwareFailure();
        emit logMessage(QString("Did not receive command echo. Command = %1, Echo = %2").arg(cmd).arg(QString(resp)));
        return false;
    }

    return true;

}

QByteArray Valon5009::valonQueryCmd(QString cmd)
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
        emit logMessage(QString("Could not read %1 frequency. No response received.").arg(channelNames().at(outputIndex)),BlackChirp::LogError);
        return -1.0;
    }

    if(!resp.startsWith("F"))
    {
        emit logMessage(QString("Could not read %1 frequency. Response: %2 (Hex: %3)")
                            .arg(channelNames().at(outputIndex)).arg(QString(resp)).arg(QString(resp.toHex())));
        return -1.0;
    }
    QByteArrayList l = resp.split(' ');
    if(l.size() < 2)
    {
        emit logMessage(QString("Could not parse %1 frequency response. Response: %2 (Hex: %3)")
                        .arg(channelNames().at(outputIndex)).arg(QString(resp)).arg(QString(resp.toHex())));
        return -1.0;
    }

    bool ok = false;
    double f = l.at(1).trimmed().toDouble(&ok);
    if(!ok)
    {
        emit logMessage(QString("Could not convert %1 frequency to number. Response: %2 (Hex: %3)")
                        .arg(channelNames().at(outputIndex)).arg(QString(l.at(1).trimmed())).arg(QString(l.at(1).trimmed().toHex())));
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
            exp.setErrorString(QString("Could not lock %1 to external reference.").arg(d_name));
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
            exp.setErrorString(QString("Could not lock %1 to internal reference.").arg(d_name));
            return false;
        }
    }

    return true;
}
