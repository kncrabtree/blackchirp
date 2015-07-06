#include "virtualawg.h"

#include "virtualinstrument.h"

VirtualAwg::VirtualAwg(QObject *parent) : AWG(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Arbitrary Waveform Generator");

    p_comm = new VirtualInstrument(d_key,this);
}

VirtualAwg::~VirtualAwg()
{

}



bool VirtualAwg::testConnection()
{
    emit connected();
    return true;
}

void VirtualAwg::initialize()
{
    testConnection();
}

Experiment VirtualAwg::prepareForExperiment(Experiment exp)
{
    return exp;
}

void VirtualAwg::beginAcquisition()
{
}

void VirtualAwg::endAcquisition()
{
}

void VirtualAwg::readTimeData()
{
}
