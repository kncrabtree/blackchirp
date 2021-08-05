#include "pressurecontrollerconfig.h"

#include <data/storage/settingsstorage.h>
#include <hardware/optional/pressurecontroller/pressurecontroller.h>

PressureControllerConfig::PressureControllerConfig() : HeaderStorage(BC::Store::PressureController::key)
{

}


void PressureControllerConfig::storeValues()
{
    SettingsStorage s(BC::Key::PController::key,SettingsStorage::Hardware);
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
