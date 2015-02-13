#include "rs232instrument.h"

Rs232Instrument::Rs232Instrument(QString key, QString name, QObject *parent) :
	HardwareObject(key,name,parent)
{
}

Rs232Instrument::~Rs232Instrument()
{
    if(d_sp->isOpen())
        d_sp->close();

    delete d_sp;
}

void Rs232Instrument::initialize()
{
	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	s.setValue(key().append(QString("/prettyName")),name());
	s.sync();

    d_sp = new QSerialPort(this);
}

bool Rs232Instrument::testConnection()
{
    if(d_sp->isOpen())
        d_sp->close();

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    int baudRate = s.value(QString("%1/baudrate").arg(key()),57600).toInt();
    QSerialPort::DataBits db = (QSerialPort::DataBits)s.value(QString("%1/databits").arg(key()),QSerialPort::Data8).toInt();
    QSerialPort::Parity parity = (QSerialPort::Parity)s.value(QString("%1/parity").arg(key()),QSerialPort::NoParity).toInt();
    QSerialPort::StopBits stop = (QSerialPort::StopBits)s.value(QString("%1/stopbits").arg(key()),QSerialPort::OneStop).toInt();
    QSerialPort::FlowControl fc = (QSerialPort::FlowControl)s.value(QString("%1/flowcontrol").arg(key()),QSerialPort::NoFlowControl).toInt();
    QString id = s.value(QString("%1/id").arg(key()),QString("")).toString();

    d_sp->setPortName(id);

//    QList<QSerialPortInfo> l = QSerialPortInfo::availablePorts();
//    for(int i=0; i<l.size(); i++)
//    {
//        if(l.at(i).portName() == id)
//            d_sp->setPort(l.at(i));
//    }

    if(d_sp->open(QIODevice::ReadWrite))
    {
        d_sp->setBaudRate((qint32)baudRate);
        d_sp->setParity(parity);
        d_sp->setStopBits(stop);
        d_sp->setDataBits(db);
        d_sp->setFlowControl(fc);
        return true;
    }
    else
        return false;
}

bool Rs232Instrument::writeCmd(QString cmd)
{
    if(!d_sp->isOpen())
    {
        emit hardwareFailure(this);
        emit logMessage(QString("Could not write command to %1. Serial port is not open. (Command = %2)").arg(d_prettyName).arg(cmd),LogHandler::Error);
        return false;
    }

    d_sp->write(cmd.toLatin1());

    if(!d_sp->flush())
    {
        emit hardwareFailure(this);
        emit logMessage(QString("Could not write command to %1. (Command = %2)").arg(d_prettyName).arg(cmd),LogHandler::Error);
        return false;
    }
    return true;
}

QByteArray Rs232Instrument::queryCmd(QString cmd)
{
    if(!d_sp->isOpen())
    {
        emit hardwareFailure(this);
        emit logMessage(QString("Could not write query to %1. Serial port is not open. (Query = %2)").arg(d_prettyName).arg(cmd),LogHandler::Error);
        return QByteArray();
    }

    if(d_sp->bytesAvailable())
        d_sp->readAll();

    d_sp->write(cmd.toLatin1());

    if(!d_sp->flush())
    {
        emit hardwareFailure(this);
        emit logMessage(QString("Could not write query to %1. (query = %2)").arg(d_prettyName).arg(cmd),LogHandler::Error);
        return QByteArray();
    }

    //write to serial port here, return response
    if(!d_useTermChar || d_readTerminator.isEmpty())
    {
        if(!d_sp->waitForReadyRead(d_timeOut))
        {
            emit hardwareFailure(this);
            emit logMessage(QString("%1 did not respond to query. (query = %2)").arg(d_prettyName).arg(cmd),LogHandler::Error);
            return QByteArray();
        }

        return d_sp->readAll();
    }
    else
    {
        QByteArray out;
        bool done = false;
        while(!done)
        {
            if(!d_sp->waitForReadyRead(d_timeOut))
                break;

            out.append(d_sp->readAll());
            if(out.endsWith(d_readTerminator))
                return out;
        }

        emit hardwareFailure(this);
        emit logMessage(QString("%1 timed out while waiting for termination character. (query = %2, partial response = %3)").arg(d_prettyName).arg(cmd).arg(QString(out)),LogHandler::Error);
        emit logMessage(QString("Hex response: %1").arg(QString(out.toHex())),LogHandler::Error);
        return out;
    }
    return QByteArray();
}
