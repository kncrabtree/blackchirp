#include "rs232instrument.h"

Rs232Instrument::Rs232Instrument(QString key, QObject *parent) :
    CommunicationProtocol(key,parent)
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

    SettingsStorage s(d_key,SettingsStorage::Hardware);

    auto baudRate = s.get<qint32>("baudrate",
                                  57600);
    auto db = s.get<QSerialPort::DataBits>("databits",
                                           QSerialPort::Data8);
    auto parity = s.get<QSerialPort::Parity>("parity",
                                             QSerialPort::NoParity);
    auto stop = s.get<QSerialPort::StopBits>("stopbits",
                                             QSerialPort::OneStop);
    auto fc = s.get<QSerialPort::FlowControl>("flowcontrol",
                                              QSerialPort::NoFlowControl);
    auto id = s.get<QString>("id","");

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
        return false;
}

