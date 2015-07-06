#include "valon5009.h"

#include "rs232instrument.h"

Valon5009::Valon5009(QObject *parent) :
    Synthesizer(parent), d_minFreq(500.0), d_maxFreq(6000.0)
{
    d_subKey = QString("valon5009");
    d_prettyName = QString("Valon Synthesizer 5009");

    p_comm = new Rs232Instrument(d_key,d_subKey,this);
    connect(p_comm,&CommunicationProtocol::logMessage,this,&HardwareObject::logMessage);
    connect(p_comm,&CommunicationProtocol::hardwareFailure,this,&HardwareObject::hardwareFailure);

    p_comm->setReadOptions(500,true,QByteArray("\n\r"));

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    //allow hardware limits to be made in settings
    d_minFreq = s.value(QString("%1/%2/minFreq").arg(d_key).arg(d_subKey),500.0).toDouble();
    d_maxFreq = s.value(QString("%1/%2/maxFreq").arg(d_key).arg(d_subKey),6000.0).toDouble();

    //these settings are used elsewhere in the program
    s.setValue(QString("%1/minFreq"),d_minFreq);
    s.setValue(QString("%1/maxFreq"),d_maxFreq);
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

    readTxFreq();
    readRxFreq();

    emit connected();
    return true;
}

void Valon5009::initialize()
{
    p_comm->initialize();
    testConnection();
}

Experiment Valon5009::prepareForExperiment(Experiment exp)
{
    //TODO: verify that 10 MHz reference is locked
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

double Valon5009::readTxFreq()
{
    QByteArray resp = valonQueryCmd(QString("S1; Frequency?\r"));
    if(!resp.startsWith("F"))
        return -1.0;
    QByteArrayList l = resp.split(' ');
    if(l.size() < 2)
        return -1.0;
    bool ok = false;
    double f = l.at(1).trimmed().toDouble(&ok);
    if(!ok)
        return -1.0;

    d_txFreq = f;
    emit txFreqRead(d_txFreq);
    return d_txFreq;
}

double Valon5009::readRxFreq()
{
    QByteArray resp = valonQueryCmd(QString("S2; Frequency?\r"));
    if(!resp.startsWith("F"))
        return -1.0;
    QByteArrayList l = resp.split(' ');
    if(l.size() < 2)
        return -1.0;
    bool ok = false;
    double f = l.at(1).trimmed().toDouble(&ok);
    if(!ok)
        return -1.0;

    d_rxFreq = f;
    emit rxFreqRead(d_rxFreq);
    return d_rxFreq;
}

double Valon5009::setTxFreq(const double f)
{
    if(f < d_minFreq || f > d_maxFreq)
        return -1.0;

    if(!valonWriteCmd(QString("S1; Frequency %1M\r").arg(f,0,'f',0)))
        return -1.0;

    return readTxFreq();
}

double Valon5009::setRxFreq(const double f)
{
    if(f < d_minFreq || f > d_maxFreq)
        return -1.0;

    if(!valonWriteCmd(QString("S2; Frequency %1M\r").arg(f,0,'f',0)))
        return -1.0;

    return readRxFreq();
}

bool Valon5009::valonWriteCmd(QString cmd)
{
    if(!p_comm->writeCmd(cmd))
        return false;

    QByteArray resp = p_comm->queryCmd(QString(""));
    if(!resp.contains(cmd.toLatin1()))
        return false;

    return true;

}

QByteArray Valon5009::valonQueryCmd(QString cmd)
{

    QByteArray resp = p_comm->queryCmd(cmd);
    resp = resp.trimmed();
    if(resp.startsWith("-1->") || resp.startsWith("-2->"))
        resp = resp.mid(4).trimmed();
    if(resp.startsWith(cmd.toLatin1()))
        resp.replace(cmd.toLatin1(),QByteArray());
    return resp.trimmed();
}
