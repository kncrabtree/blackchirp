#include "virtualvalonsynth.h"

#include "virtualinstrument.h"

VirtualValonSynth::VirtualValonSynth(QObject *parent) : Synthesizer(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Synthesizer");

    d_comm = new VirtualInstrument(d_key,this);
    connect(d_comm,&CommunicationProtocol::logMessage,this,&VirtualValonSynth::logMessage);
    connect(d_comm,&CommunicationProtocol::hardwareFailure,[=](){ emit hardwareFailure(); });
}

VirtualValonSynth::~VirtualValonSynth()
{

}



bool VirtualValonSynth::testConnection()
{
    readTxFreq();
    readRxFreq();

    emit connected();
    return true;
}

void VirtualValonSynth::initialize()
{
    testConnection();
}

Experiment VirtualValonSynth::prepareForExperiment(Experiment exp)
{
    return exp;
}

void VirtualValonSynth::beginAcquisition()
{
}

void VirtualValonSynth::endAcquisition()
{
}

void VirtualValonSynth::readTimeData()
{
}

double VirtualValonSynth::readTxFreq()
{
    emit txFreqRead(d_txFreq);
    return d_txFreq;
}

double VirtualValonSynth::readRxFreq()
{
    emit txFreqRead(d_rxFreq);
    return d_rxFreq;
}

double VirtualValonSynth::setTxFreq(const double f)
{
    d_txFreq = f;
    return readTxFreq();
}

double VirtualValonSynth::setRxFreq(const double f)
{
    d_rxFreq = f;
    return readRxFreq();
}
