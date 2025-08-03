#include "rs232instrument.h"
#include <data/settings/hardwarekeys.h>

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

    // Load RS232 settings from group with backward compatibility fallback
    auto baudRate = s.getGroupValue<qint32>(BC::Key::Comm::rs232, baud, s.get<qint32>(baud, 57600));
    auto db = static_cast<DataBits>(s.getGroupValue<int>(BC::Key::Comm::rs232, dataBits, 
                                                        s.get<int>(dataBits, static_cast<int>(Data8))));
    auto p = static_cast<Parity>(s.getGroupValue<int>(BC::Key::Comm::rs232, parity, 
                                                      s.get<int>(parity, static_cast<int>(NoParity))));
    auto stop = static_cast<StopBits>(s.getGroupValue<int>(BC::Key::Comm::rs232, stopBits, 
                                                           s.get<int>(stopBits, static_cast<int>(OneStop))));
    auto fc = static_cast<FlowControl>(s.getGroupValue<int>(BC::Key::Comm::rs232, flowControl, 
                                                            s.get<int>(flowControl, static_cast<int>(NoFlowControl))));
    auto name = s.getGroupValue<QString>(BC::Key::Comm::rs232, id, s.get<QString>(id, ""));

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

bool Rs232Instrument::testManual(QString name, qint32 br)
{
    using namespace BC::Key::RS232;
    if(p_device->isOpen())
        p_device->close();

    auto db = Data8;
    auto p = NoParity;
    auto stop = OneStop;
    auto fc = NoFlowControl;

    auto p_sp = dynamic_cast<QSerialPort*>(p_device);
    p_sp->setPortName(name);

    if(p_device->open(QIODevice::ReadWrite))
    {
        p_sp->setBaudRate(br);
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
