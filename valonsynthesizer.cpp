#include "valonsynthesizer.h"

ValonSynthesizer::ValonSynthesizer()
 : Rs232Instrument(QString("valonSynth"),QString("Valon Synthesizer"))
{
#ifdef BC_NOVALONSYNTH
    d_virtual = true;
#endif
}

ValonSynthesizer::~ValonSynthesizer()
{

}



Experiment ValonSynthesizer::prepareForExperiment(Experiment exp)
{
    return exp;
}

void ValonSynthesizer::beginAcquisition()
{
}

void ValonSynthesizer::endAcquisition()
{
}

void ValonSynthesizer::readTimeData()
{
}

double ValonSynthesizer::readTxFreq()
{
    if(d_virtual)
    {
        emit txFreqRead(d_txFreq);
        return d_txFreq;
    }

    //implement communication
    emit txFreqRead(d_txFreq);
    return d_txFreq;
}

double ValonSynthesizer::readRxFreq()
{
    if(d_virtual)
    {
        emit rxFreqRead(d_rxFreq);
        return d_rxFreq;
    }

    //implement communication
    emit rxFreqRead(d_rxFreq);
    return d_rxFreq;
}

double ValonSynthesizer::setTxFreq(const double f)
{
    if(d_virtual)
        d_txFreq = f;
    else
    {

    }

    return readTxFreq();
}

double ValonSynthesizer::setRxFreq(const double f)
{
    if(d_virtual)
        d_rxFreq = f;
    else
    {

    }

    return readRxFreq();
}

bool ValonSynthesizer::testConnection()
{
    if(!Rs232Instrument::testConnection())
    {
        emit connectionResult(this,false);
        return false;
    }

    //device-specific connection test
    if(!d_virtual)
    {
        //IMPLEMENT
    }

    readTxFreq();
    readRxFreq();

    emit connectionResult(this,true);
    return true;
}

void ValonSynthesizer::initialize()
{
     Rs232Instrument::initialize();

    if(d_virtual)
    {
        QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
        d_txFreq = s.value(QString("%1/txFreq").arg(key()),5760.0).toDouble();
        d_rxFreq = s.value(QString("%1/rxFreq").arg(key()),5120.0).toDouble();
    }

    testConnection();
}
