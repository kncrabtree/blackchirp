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
    using namespace BC::Key::RS232;
    if(p_device->isOpen())
        p_device->close();

    SettingsStorage s(d_key,SettingsStorage::Hardware);

    auto baudRate = s.get<qint32>(baud,
                                  57600);
    auto db = s.get<QSerialPort::DataBits>(dataBits,
                                           QSerialPort::Data8);
    auto p = s.get<QSerialPort::Parity>(parity,
                                             QSerialPort::NoParity);
    auto stop = s.get<QSerialPort::StopBits>(stopBits,
                                             QSerialPort::OneStop);
    auto fc = s.get<QSerialPort::FlowControl>(flowControl,
                                              QSerialPort::NoFlowControl);
    auto name = s.get<QString>(id,"");

    auto p_sp = dynamic_cast<QSerialPort*>(p_device);
    p_sp->setPortName(name);

    if(p_device->open(QIODevice::ReadWrite))
    {
        p_sp->setBaudRate((qint32)baudRate);
        p_sp->setParity(p);
        p_sp->setStopBits(stop);
        p_sp->setDataBits(db);
        p_sp->setFlowControl(fc);
        return true;
    }
    else
        return false;
}

