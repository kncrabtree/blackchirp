#include "virtualmotorscope.h"

VirtualMotorScope::VirtualMotorScope(QObject *parent) : MotorOscilloscope(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Motor Oscilloscope");
    d_threaded = true;
    d_commType = CommunicationProtocol::Virtual;

    //establish settings parameters (min/max sample rate, vertical scale, etc)
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    s.setValue(QString("minDataChannel"),1);
    s.setValue(QString("maxDataChannel"),2);
    s.setValue(QString("minTriggerChannel"),1);
    s.setValue(QString("maxTriggerChannel"),2);
    s.setValue(QString("minVerticalScale"),0.02);
    s.setValue(QString("maxVerticalScale"),20);
    s.setValue(QString("minRecordLength"),1);
    s.setValue(QString("maxRecordLength"),32e6); // ?
    s.setValue(QString("minSampleRate"),16);
    s.setValue(QString("maxSampleRate"),69e9);

    s.endGroup();
    s.endGroup();
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
