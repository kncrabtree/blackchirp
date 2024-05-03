#include "lifdigitizerconfig.h"

#include <modules/lif/hardware/lifdigitizer/lifscope.h>

LifDigitizerConfig::LifDigitizerConfig(const QString subKey) : DigitizerConfig(BC::Key::hwKey(BC::Key::LifDigi::lifScope,0),subKey)
{

}


void LifDigitizerConfig::storeValues()
{
    using namespace BC::Store::Digi;
    store(lifChannel,d_lifChannel);
    store(lifRefChannel,d_refChannel);
    store(lifRefEnabled,d_refEnabled);

    DigitizerConfig::storeValues();
}

void LifDigitizerConfig::retrieveValues()
{
    using namespace BC::Store::Digi;
    d_lifChannel = retrieve(lifChannel,0);
    d_refChannel = retrieve(lifRefChannel,1);
    d_refEnabled = retrieve(lifRefEnabled,false);

    DigitizerConfig::retrieveValues();
}
