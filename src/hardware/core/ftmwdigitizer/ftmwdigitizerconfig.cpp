#include "ftmwdigitizerconfig.h"

#include <hardware/core/ftmwdigitizer/ftmwscope.h>

FtmwDigitizerConfig::FtmwDigitizerConfig(const QString subKey) : DigitizerConfig(BC::Key::hwKey(BC::Key::FtmwScope::ftmwScope,0),subKey)
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
