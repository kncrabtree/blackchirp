#include "virtualioboard.h"

VirtualIOBoard::VirtualIOBoard(QObject *parent) :
    IOBoard(BC::Key::hwVirtual,BC::Key::IOB::viobName,CommunicationProtocol::Virtual,parent)
{
    //See labjacku3.cpp for an explanation of these parameters
    d_numAnalog = 4;
    d_numDigital = 16-d_numAnalog;
    d_reservedAnalog = 0;
    d_reservedDigital = 0;
}



bool VirtualIOBoard::testConnection()
{
    return true;
}

void VirtualIOBoard::initialize()
{
}

bool VirtualIOBoard::prepareForExperiment(Experiment &exp)
{
    d_config = exp.iobConfig();

    for(auto it = d_config.analogList().constBegin();it!=d_config.analogList().constEnd();it++)
    {
        if(it.value().enabled)
            exp.auxData()->registerKey(d_key,d_subKey,QString("ain.%1").arg(it.key()));
    }
    for(auto it = d_config.digitalList().constBegin();it!=d_config.digitalList().constEnd();it++)
    {
        if(it.value().enabled)
            exp.auxData()->registerKey(d_key,d_subKey,QString("din.%1").arg(it.key()));
    }

    return true;
}

AuxDataStorage::AuxDataMap VirtualIOBoard::readAuxData()
{
    return auxData(true);
}

AuxDataStorage::AuxDataMap VirtualIOBoard::readValidationData()
{
    return auxData(false);
}

AuxDataStorage::AuxDataMap VirtualIOBoard::auxData(bool plot)
{
    AuxDataStorage::AuxDataMap out;

    for(auto it = d_config.analogList().constBegin();it!=d_config.analogList().constEnd();it++)
    {
        auto ch = it.value();
        if(ch.enabled)
        {
            double val = static_cast<double>((qrand() % 20000) - 10000)/1000.0;
            if(ch.plot == plot)
                out.insert({QString("ain.%1").arg(it.key()),val});
        }
    }

    for(auto it = d_config.digitalList().constBegin();it != d_config.digitalList().constEnd(); it++)
    {
        auto ch = it.value();
        if(ch.enabled)
        {
            int val = qrand() % 2;
            if(ch.plot == plot)
                out.insert({QString("din.%1").arg(it.key()),val});
        }
    }

    return out;
}


void VirtualIOBoard::readIOBSettings()
{
}
