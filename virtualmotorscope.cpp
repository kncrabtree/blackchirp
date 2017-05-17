#include "virtualmotorscope.h"

#include <QTimer>

VirtualMotorScope::VirtualMotorScope(QObject *parent) : MotorOscilloscope(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Motor Oscilloscope");
    d_threaded = false;
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

    d_config.dataChannel = s.value(QString("dataChannel"),1).toInt();
    d_config.triggerChannel = s.value(QString("triggerChannel"),2).toInt();
    d_config.verticalScale = s.value(QString("verticalScale"),5.0).toDouble();
    d_config.recordLength = s.value(QString("recordLength"),100).toInt();
    d_config.sampleRate = s.value(QString("sampleRate"),500.0).toDouble();
    d_config.slope = static_cast<BlackChirp::ScopeTriggerSlope>(s.value(QString("slope"),BlackChirp::ScopeTriggerSlope::RisingEdge).toUInt());

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
    p_testTimer = new QTimer(this);
    p_testTimer->setInterval(200);
    connect(p_testTimer,&QTimer::timeout,this,&VirtualMotorScope::queryScope);

}

bool VirtualMotorScope::configure(const BlackChirp::MotorScopeConfig &sc)
{
    BlackChirp::MotorScopeConfig out = sc;
    out.byteOrder = QDataStream::BigEndian;
    out.bytesPerPoint = 2;

    d_config = out;
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

void VirtualMotorScope::queryScope()
{
    QVector<double> out;
    out.resize(d_config.recordLength);

    for(int i=0; i<d_config.recordLength; i++)
    {
        double dat = static_cast<double>((qrand() % 65536 - 32768)) / 65536.0 * d_config.verticalScale;
        out[i] = dat;
    }

    emit traceAcquired(out);
}


void VirtualMotorScope::beginAcquisition()
{
    p_testTimer->start();
}

void VirtualMotorScope::endAcquisition()
{
    p_testTimer->stop();
}
