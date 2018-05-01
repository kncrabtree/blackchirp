#include "valon5009.h"

Valon5009::Valon5009(QObject *parent) :
    Synthesizer(parent)
{
    d_subKey = QString("valon5009");
    d_prettyName = QString("Valon Synthesizer 5009");
    d_commType = CommunicationProtocol::Rs232;
    d_threaded = false;


    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    //allow hardware limits to be made in settings
    d_minFreq = s.value(QString("minFreq"),500.0).toDouble();
    d_maxFreq = s.value(QString("maxFreq"),6000.0).toDouble();
    //write the settings if they're not there
    s.setValue(QString("minFreq"),d_minFreq);
    s.setValue(QString("maxFreq"),d_maxFreq);
    s.endGroup();
    s.endGroup();

}



bool Valon5009::testConnection()
{
    if(!p_comm->testConnection())
    {
        emit connected(false,QString("RS232 error."));
        return false;
    }

    QByteArray resp = valonQueryCmd(QString("ID\r"));

    if(resp.isEmpty())
    {
        emit connected(false,QString("No response to ID query."));
        return false;
    }

    if(!resp.startsWith("Valon Technology, 5009"))
    {
        emit connected(false,QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp.trimmed())).arg(QString(resp.toHex())));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp)));

    if(readTxFreq() < 0.0)
    {
        emit connected(false,QString("Error reading TX frequency"));
        return false;
    }
    if(readRxFreq() < 0.0)
    {
        emit connected(false,QString("Error reading RCVR frequency"));
        return false;
    }

    emit connected();
    return true;
}

void Valon5009::initialize()
{
    p_comm->initialize();
    p_comm->setReadOptions(500,true,QByteArray("\n\r"));
    testConnection();
}

Experiment Valon5009::prepareForExperiment(Experiment exp)
{
    //TODO: verify that 10 MHz reference is locked
    //Ensure that TX and RX frequencies match FT and chirp configs, if applicable
    return exp;
}

void Valon5009::beginAcquisition()
{
}

void Valon5009::endAcquisition()
{
}

void Valon5009::readTimeData()
{
}

double Valon5009::readSynthTxFreq()
{
    if(readSynth(1))
        return d_txFreq;
    else
        return -1.0;
}

double Valon5009::readSynthRxFreq()
{
    if(readSynth(2))
        return d_rxFreq;
    else
        return -1.0;
}

double Valon5009::setSynthTxFreq(const double f)
{
    if(!setSynth(1,f))
        return -1.0;

    return readTxFreq();
}

double Valon5009::setSynthRxFreq(const double f)
{
    if(!setSynth(2,f))
        return -1.0;

    return readRxFreq();
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

bool Valon5009::setSynth(int channel, double f)
{
    if(channel != 1)
        channel = 2;
    if(!valonWriteCmd(QString("S%1; Frequency %2M\r").arg(channel).arg(f,0,'f',0)))
        return false;

    return true;
}

bool Valon5009::readSynth(int channel)
{
    QString cmd = QString("S1; Frequency?\r");
    QString type = QString("TX");
    double *freq = &d_txFreq;
    if(channel == 2)
    {
        cmd = QString("S2; Frequency?\r");
        type = QString("RX");
        freq = &d_rxFreq;
    }

    QByteArray resp = valonQueryCmd(cmd);
    if(resp.isEmpty())
        return false;

    if(!resp.startsWith("F"))
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read %1 frequency. Response: %2 (Hex: %3)")
                            .arg(type).arg(QString(resp)).arg(QString(resp.toHex())));
        return false;
    }
    QByteArrayList l = resp.split(' ');
    if(l.size() < 2)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not parse %1 frequency response. Response: %2 (Hex: %3)")
                        .arg(type).arg(QString(resp)).arg(QString(resp.toHex())));
        return false;
    }
    bool ok = false;
    double f = l.at(1).trimmed().toDouble(&ok);
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not convert %1 frequency to number. Response: %2 (Hex: %3)")
                        .arg(type).arg(QString(l.at(1).trimmed())).arg(QString(l.at(1).trimmed().toHex())));
        return false;
    }

    *freq = f;
    return true;

}
