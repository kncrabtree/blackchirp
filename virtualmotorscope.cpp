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
    int minDataChannel = s.value(QString("minDataChannel"),1).toInt();
    int maxDataChannel = s.value(QString("maxDataChannel"),2).toInt();
    int minTriggerChannel = s.value(QString("minTriggerChannel"),1).toInt();
    int maxTriggerChannel = s.value(QString("maxTriggerChannel"),2).toInt();
    double minVerticalScale = s.value(QString("minVerticalScale"),0.02).toDouble();
    double maxVerticalScale = s.value(QString("maxVerticalScale"),20).toDouble();
    int minRecordLength = s.value(QString("minRecordLength"),1).toInt();
    int maxRecordLength = s.value(QString("maxRecordLength"),32e6).toInt();
    double minSampleRate = s.value(QString("minSampleRate"),16).toDouble();
    double maxSampleRate = s.value(QString("maxSampleRate"),69e9).toDouble();
    s.setValue(QString("minDataChannel"),minDataChannel);
    s.setValue(QString("maxDataChannel"),maxDataChannel);
    s.setValue(QString("minTriggerChannel"),minTriggerChannel);
    s.setValue(QString("maxTriggerChannel"),maxTriggerChannel);
    s.setValue(QString("minVerticalScale"),minVerticalScale);
    s.setValue(QString("maxVerticalScale"),maxVerticalScale);
    s.setValue(QString("minRecordLength"),minRecordLength);
    s.setValue(QString("maxRecordLength"),maxRecordLength);
    s.setValue(QString("minSampleRate"),minSampleRate);
    s.setValue(QString("maxSampleRate"),maxSampleRate);
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
