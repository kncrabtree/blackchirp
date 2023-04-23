#include "ftmwdigitizerconfig.h"

FtmwDigitizerConfig::FtmwDigitizerConfig() : DigitizerConfig(BC::Store::Digi::ftmwKey)
{

}


void FtmwDigitizerConfig::storeValues()
{
    using namespace BC::Store::Digi;
    store(fidCh,d_fidChannel);

    DigitizerConfig::storeValues();
}

void FtmwDigitizerConfig::retrieveValues()
{
    using namespace BC::Store::Digi;
    d_fidChannel = retrieve(fidCh,0);

    DigitizerConfig::retrieveValues();
}
