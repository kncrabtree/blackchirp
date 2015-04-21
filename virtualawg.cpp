#include "virtualawg.h"

#include "virtualinstrument.h"

VirtualAwg::VirtualAwg(QObject *parent) : AWG(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Arbitrary Waveform Generator");

    d_comm = new VirtualInstrument(d_key,this);
    connect(d_comm,&CommunicationProtocol::logMessage,this,&VirtualAwg::logMessage);
    connect(d_comm,&CommunicationProtocol::hardwareFailure,[=](){ emit hardwareFailure(); });
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
