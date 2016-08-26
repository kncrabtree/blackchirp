#include "virtualsynth.h"

VirtualSynth::VirtualSynth(QObject *parent) : Synthesizer(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Synthesizer");
    d_commType = CommunicationProtocol::Virtual;
    d_threaded = false;

    d_txFreq = 5760.0;
    d_rxFreq = 5120.0;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    //allow hardware limits to be made in settings
    d_minFreq = s.value(QString("minFreq"),500.0).toDouble();
    d_maxFreq = s.value(QString("maxFreq"),6000.0).toDouble();
    //write the settings if they're not there
    s.setValue(QString("minFreq"),d_minFreq);
    s.setValue(QString("maxFreq"),d_maxFreq);
    s.endGroup();
    s.endGroup();
}

VirtualSynth::~VirtualSynth()
{

}



bool VirtualSynth::testConnection()
{
    readTxFreq();
    readRxFreq();

    emit connected();
    return true;
}

void VirtualSynth::initialize()
{
    testConnection();
}

Experiment VirtualSynth::prepareForExperiment(Experiment exp)
{
    return exp;
}

void VirtualSynth::beginAcquisition()
{
}

void VirtualSynth::endAcquisition()
{
}

void VirtualSynth::readTimeData()
{
}

double VirtualSynth::readSynthTxFreq()
{
    return d_txFreq;
}

double VirtualSynth::readSynthRxFreq()
{
    return d_rxFreq;
}

double VirtualSynth::setSynthTxFreq(const double f)
{
    d_txFreq = f;
    return readTxFreq();
}

double VirtualSynth::setSynthRxFreq(const double f)
{
    d_rxFreq = f;
    return readRxFreq();
}
