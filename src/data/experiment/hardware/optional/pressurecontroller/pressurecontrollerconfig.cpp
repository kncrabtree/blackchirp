#include "pressurecontrollerconfig.h"

#include <data/storage/settingsstorage.h>
#include <data/settings/hardwarekeys.h>
#include <hardware/optional/pressurecontroller/pressurecontroller.h>

PressureControllerConfig::PressureControllerConfig(const QString& hwType, const QString& impl, const QString& label) : HeaderStorage(BC::Key::hwKey(hwType, label))
{
    Q_UNUSED(impl)
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
