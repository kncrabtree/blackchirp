#include "ftmwdigitizerconfig.h"

FtmwDigitizerConfig::FtmwDigitizerConfig() : DigitizerConfig(BC::Store::Digi::ftmwKey)
{

}


void FtmwDigitizerConfig::prepareToSave()
{
    using namespace BC::Store::Digi;
    store(fidCh,d_fidChannel);

    DigitizerConfig::prepareToSave();
}

void FtmwDigitizerConfig::loadComplete()
{
    using namespace BC::Store::Digi;
    d_fidChannel = retrieve(fidCh,0);

    DigitizerConfig::loadComplete();
}
