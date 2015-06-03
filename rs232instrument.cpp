#include "rs232instrument.h"

Rs232Instrument::Rs232Instrument(QString key, QString subKey, QObject *parent) :
    CommunicationProtocol(CommunicationProtocol::Rs232,key,subKey,parent)
{

}

Rs232Instrument::~Rs232Instrument()
{
    if(p_sp->isOpen())
        p_sp->close();

    delete p_sp;
}

void Rs232Instrument::initialize()
{
    p_sp = new QSerialPort(this);
}

bool Rs232Instrument::testConnection()
{
    if(p_sp->isOpen())
        p_sp->close();

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    int baudRate = s.value(QString("%1/baudrate").arg(key()),57600).toInt();
    QSerialPort::DataBits db = (QSerialPort::DataBits)s.value(QString("%1/databits").arg(key()),QSerialPort::Data8).toInt();
    QSerialPort::Parity parity = (QSerialPort::Parity)s.value(QString("%1/parity").arg(key()),QSerialPort::NoParity).toInt();
    QSerialPort::StopBits stop = (QSerialPort::StopBits)s.value(QString("%1/stopbits").arg(key()),QSerialPort::OneStop).toInt();
    QSerialPort::FlowControl fc = (QSerialPort::FlowControl)s.value(QString("%1/flowcontrol").arg(key()),QSerialPort::NoFlowControl).toInt();
    QString id = s.value(QString("%1/id").arg(key()),QString("")).toString();

    p_sp->setPortName(id);

    if(p_sp->open(QIODevice::ReadWrite))
    {
        p_sp->setBaudRate((qint32)baudRate);
        p_sp->setParity(parity);
        p_sp->setStopBits(stop);
        p_sp->setDataBits(db);
        p_sp->setFlowControl(fc);
        return true;
    }
    else
        return false;
}

bool Rs232Instrument::writeCmd(QString cmd)
{
    if(!p_sp->isOpen())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write command. Serial port is not open. (Command = %1)").arg(cmd),BlackChirp::LogError);
        return false;
    }

    qint64 ret = p_sp->write(cmd.toLatin1());

    if(ret == -1)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write command. (Command = %1)").arg(cmd),BlackChirp::LogError);
        return false;
    }
    return true;
}

QByteArray Rs232Instrument::queryCmd(QString cmd)
{
    if(!p_sp->isOpen())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write query. Serial port is not open. (Query = %1)").arg(cmd),BlackChirp::LogError);
        return QByteArray();
    }

    if(p_sp->bytesAvailable())
        p_sp->readAll();

    qint64 ret = p_sp->write(cmd.toLatin1());

    if(ret == -1)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write query. (query = %1)").arg(cmd),BlackChirp::LogError);
        return QByteArray();
    }

    //write to serial port here, return response
    if(!d_useTermChar || d_readTerminator.isEmpty())
    {
        if(!p_sp->waitForReadyRead(d_timeOut))
        {
            emit hardwareFailure();
            emit logMessage(QString("Did not respond to query. (query = %1)").arg(cmd),BlackChirp::LogError);
            return QByteArray();
        }

        return p_sp->readAll();
    }
    else
    {
        QByteArray out;
        bool done = false;
        while(!done)
        {
            if(!p_sp->waitForReadyRead(d_timeOut))
                break;

            out.append(p_sp->readAll());
            if(out.endsWith(d_readTerminator))
                return out;
        }

        emit hardwareFailure();
        emit logMessage(QString("Timed out while waiting for termination character. (query = %1, partial response = %2)").arg(cmd).arg(QString(out)),BlackChirp::LogError);
        emit logMessage(QString("Hex response: %1").arg(QString(out.toHex())),BlackChirp::LogError);
        return out;
    }
    return QByteArray();
}
