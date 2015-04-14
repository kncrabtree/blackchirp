#include "awg.h"

AWG::AWG() : TcpInstrument(QString("awg"),QString("Arbitrary Waveform Generator"))
{
#ifdef BC_NOAWG
    d_virtual = true;
#endif
}

AWG::~AWG()
{

}



bool AWG::testConnection()
{
    if(!TcpInstrument::testConnection())
    {
        emit connectionResult(this,false);
        return false;
    }

    if(!d_virtual)
    {
        //implement
    }

    emit connectionResult(this,true);
    return true;
}

void AWG::initialize()
{
    TcpInstrument::initialize();
    testConnection();
}

Experiment AWG::prepareForExperiment(Experiment exp)
{
    if(d_virtual)
        return exp;

    //write chirp waveform; verify settings
}

void AWG::beginAcquisition()
{
    //enable AWG output
}

void AWG::endAcquisition()
{
    //disable awg output
}

void AWG::readTimeData()
{
    //no data to read
}
