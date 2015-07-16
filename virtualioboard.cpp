#include "virtualioboard.h"

#include "virtualinstrument.h"

VirtualIOBoard::VirtualIOBoard(QObject *parent) :
    IOBoard(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual IO Board");

    p_comm = new VirtualInstrument(d_key,this);

    //See labjacku3.cpp for an explanation of these parameters
    d_numAnalog = 4;
    d_numDigital = 16-d_numAnalog;
    d_reservedAnalog = 0;
    d_reservedDigital = 0;
}



bool VirtualIOBoard::testConnection()
{
    readSettings();
    emit connected();
    return true;
}

void VirtualIOBoard::initialize()
{
    testConnection();
}

Experiment VirtualIOBoard::prepareForExperiment(Experiment exp)
{
    return exp;
}

void VirtualIOBoard::beginAcquisition()
{
}

void VirtualIOBoard::endAcquisition()
{
}

void VirtualIOBoard::readTimeData()
{
}
