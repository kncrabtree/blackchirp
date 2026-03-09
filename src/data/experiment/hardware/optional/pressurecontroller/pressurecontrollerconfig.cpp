#include "pressurecontrollerconfig.h"

#include <data/storage/settingsstorage.h>
#include <data/settings/hardwarekeys.h>

PressureControllerConfig::PressureControllerConfig(const QString& hwKey) : HeaderStorage(hwKey)
{
}


void PressureControllerConfig::storeValues()
{
    SettingsStorage s(headerKey(),SettingsStorage::Hardware);
    using namespace BC::Store::PressureController;
    store(pcEnabled,d_pressureControlMode);
    store(pSetPoint,d_setPoint,s.get(BC::Key::PController::units,QString("")));
}

void PressureControllerConfig::retrieveValues()
{
    using namespace BC::Store::PressureController;
    d_pressureControlMode = retrieve(pcEnabled,false);
    d_setPoint = retrieve(pSetPoint,0.0);
}
