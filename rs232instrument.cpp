#include "rs232instrument.h"

Rs232Instrument::Rs232Instrument(QString key, QString subKey, QObject *parent) :
    CommunicationProtocol(CommunicationProtocol::Rs232,key,subKey,parent)
{

}

Rs232Instrument::~Rs232Instrument()
{
    if(p_device->isOpen())
        p_device->close();

    delete p_device;
}

void Rs232Instrument::initialize()
{
    p_device = new QSerialPort(this);
}

bool Rs232Instrument::testConnection()
{
    if(p_device->isOpen())
        p_device->close();

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    int baudRate = s.value(QString("%1/baudrate").arg(key()),57600).toInt();
    QSerialPort::DataBits db = (QSerialPort::DataBits)s.value(QString("%1/databits").arg(key()),QSerialPort::Data8).toInt();
    QSerialPort::Parity parity = (QSerialPort::Parity)s.value(QString("%1/parity").arg(key()),QSerialPort::NoParity).toInt();
    QSerialPort::StopBits stop = (QSerialPort::StopBits)s.value(QString("%1/stopbits").arg(key()),QSerialPort::OneStop).toInt();
    QSerialPort::FlowControl fc = (QSerialPort::FlowControl)s.value(QString("%1/flowcontrol").arg(key()),QSerialPort::NoFlowControl).toInt();
    QString id = s.value(QString("%1/id").arg(key()),QString("")).toString();

    auto p_sp = dynamic_cast<QSerialPort*>(p_device);
    p_sp->setPortName(id);

    if(p_device->open(QIODevice::ReadWrite))
    {
        p_sp->setBaudRate((qint32)baudRate);
        p_sp->setParity(parity);
        p_sp->setStopBits(stop);
        p_sp->setDataBits(db);
        p_sp->setFlowControl(fc);
        return true;
    }
    else
    {
        emit logMessage(p_device->errorString(),BlackChirp::LogError);
        return false;
    }
}

