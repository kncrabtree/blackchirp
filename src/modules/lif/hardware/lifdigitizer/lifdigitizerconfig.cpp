#include "lifdigitizerconfig.h"

LifDigitizerConfig::LifDigitizerConfig() : DigitizerConfig(BC::Store::Digi::lifKey)
{

}


void LifDigitizerConfig::storeValues()
{
    using namespace BC::Store::Digi;
    store(lifChannel,d_lifChannel);
    store(lifRefChannel,d_refChannel);
    store(lifRefEnabled,d_refEnabled);
    store(lifChannelOrder,d_channelOrder);

    DigitizerConfig::storeValues();
}

void LifDigitizerConfig::retrieveValues()
{
    using namespace BC::Store::Digi;
    d_lifChannel = retrieve(lifChannel,0);
    d_refChannel = retrieve(lifRefChannel,1);
    d_refEnabled = retrieve(lifRefEnabled,false);
    d_channelOrder = retrieve(lifChannelOrder,Interleaved);

    DigitizerConfig::retrieveValues();
}
