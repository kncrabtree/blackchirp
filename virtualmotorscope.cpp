#include "virtualmotorscope.h"

VirtualMotorScope::VirtualMotorScope(QObject *parent) : MotorOscilloscope(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Motor Oscilloscope");
    d_threaded = true;
    d_commType = CommunicationProtocol::Virtual;

    //establish settings parameters (min/max sample rate, vertical scale, etc)
}



bool VirtualMotorScope::testConnection()
{
    emit connected();
    return true;
}

void VirtualMotorScope::initialize()
{
    testConnection();
}

bool VirtualMotorScope::configure(const BlackChirp::MotorScopeConfig &sc)
{
    BlackChirp::MotorScopeConfig out = sc;
    out.byteOrder = QDataStream::BigEndian;
    out.bytesPerPoint = 2;

    d_currentConfig = out;
    emit configChanged(out);
    return true;
}

MotorScan VirtualMotorScope::prepareForMotorScan(MotorScan s)
{
    BlackChirp::MotorScopeConfig sc = s.scopeConfig();
    bool success = configure(sc);
    s.setScopeConfig(sc);
    if(!success)
        s.setHardwareError();
    return s;


}