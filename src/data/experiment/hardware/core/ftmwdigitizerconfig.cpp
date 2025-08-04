#include "ftmwdigitizerconfig.h"

#include <hardware/core/ftmwdigitizer/ftmwscope.h>

FtmwDigitizerConfig::FtmwDigitizerConfig(const QString& hwType, const QString& impl, const QString& label) : DigitizerConfig(BC::Key::hwKey(hwType, label), impl)
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
