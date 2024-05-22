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
    auto db = s.get<DataBits>(dataBits,Data8);
    auto p = s.get<Parity>(parity,NoParity);
    auto stop = s.get<StopBits>(stopBits,OneStop);
    auto fc = s.get<FlowControl>(flowControl,NoFlowControl);
    auto name = s.get<QString>(id,"");

    auto p_sp = dynamic_cast<QSerialPort*>(p_device);
    p_sp->setPortName(name);

    if(p_device->open(QIODevice::ReadWrite))
    {
        p_sp->setBaudRate((qint32)baudRate);
        p_sp->setParity(static_cast<QSerialPort::Parity>(p));
        p_sp->setStopBits(static_cast<QSerialPort::StopBits>(stop));
        p_sp->setDataBits(static_cast<QSerialPort::DataBits>(db));
        p_sp->setFlowControl(static_cast<QSerialPort::FlowControl>(fc));
        return true;
    }
    else
        return false;
}



QIODevice *Rs232Instrument::_device()
{
    return p_device;
}
