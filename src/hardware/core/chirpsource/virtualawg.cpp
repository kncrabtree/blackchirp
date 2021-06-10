#include "virtualawg.h"

VirtualAwg::VirtualAwg(QObject *parent) : AWG(BC::Key::hwVirtual,parent)
{
    d_prettyName = QString("Virtual Arbitrary Waveform Generator");
    d_commType = CommunicationProtocol::Virtual;
    d_threaded = false;
}

VirtualAwg::~VirtualAwg()
{

}

void VirtualAwg::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    double awgRate = s.value(QString("sampleRate"),16e9).toDouble();
    double awgMaxSamples = s.value(QString("maxSamples"),2e9).toDouble();
    double awgMinFreq = s.value(QString("minFreq"),100.0).toDouble();
    double awgMaxFreq = s.value(QString("maxFreq"),6250.0).toDouble();
    bool pp = s.value(QString("hasProtectionPulse"),true).toBool();
    bool ep = s.value(QString("hasAmpEnablePulse"),true).toBool();
    bool ro = s.value(QString("rampOnly"),false).toBool();
    s.setValue(QString("sampleRate"),awgRate);
    s.setValue(QString("maxSmaples"),awgMaxSamples);
    s.setValue(QString("minFreq"),awgMinFreq);
    s.setValue(QString("maxFreq"),awgMaxFreq);
    s.setValue(QString("hasProtectionPulse"),pp);
    s.setValue(QString("hasAmpEnablePulse"),ep);
    s.setValue(QString("rampOnly"),ro);
    s.endGroup();
    s.endGroup();
    s.sync();
}



bool VirtualAwg::testConnection()
{
    return true;
}

void VirtualAwg::initialize()
{
}
