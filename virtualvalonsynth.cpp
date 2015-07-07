#include "virtualvalonsynth.h"

#include "virtualinstrument.h"

VirtualValonSynth::VirtualValonSynth(QObject *parent) : Synthesizer(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Synthesizer");

    p_comm = new VirtualInstrument(d_key,this);
    d_txFreq = 5760.0;
    d_rxFreq = 5120.0;
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
    Synthesizer::initialize();
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
    emit rxFreqRead(d_rxFreq);
    return d_rxFreq;
}

double VirtualValonSynth::setSynthTxFreq(const double f)
{
    d_txFreq = f;
    return readTxFreq();
}

double VirtualValonSynth::setSynthRxFreq(const double f)
{
    d_rxFreq = f;
    return readRxFreq();
}
