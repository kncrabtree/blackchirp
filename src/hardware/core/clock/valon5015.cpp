#include "valon5015.h"

Valon5015::Valon5015(int clockNum, QObject *parent) :
    Clock(clockNum,1,true,BC::Key::valon5015,BC::Key::valon5015Name,CommunicationProtocol::Rs232,parent)
{
    setDefault(BC::Key::Clock::minFreq,500.0);
    setDefault(BC::Key::Clock::maxFreq,15000.0);
    setDefault(BC::Key::Clock::lock,false);
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

    emit logMessage(QString("ID response: %1").arg(QString(resp)));

    return true;
}

void Valon5015::initializeClock()
{
    p_comm->setReadOptions(500,true,QByteArray("\n\r"));
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
            exp.setErrorString(QString("Could not lock %1 to external reference.").arg(d_name));
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
            exp.setErrorString(QString("Could not lock %1 to internal reference.").arg(d_name));
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
        emit logMessage(QString("Could not read %1 frequency. No response received.").arg(channelNames().at(outputIndex)),LogHandler::Error);
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

bool Valon5015::valonWriteCmd(QString cmd)
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

QByteArray Valon5015::valonQueryCmd(QString cmd)
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
