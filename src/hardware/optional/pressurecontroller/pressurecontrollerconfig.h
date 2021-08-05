#ifndef PRESSURECONTROLLERCONFIG_H
#define PRESSURECONTROLLERCONFIG_H

#include <data/storage/headerstorage.h>

namespace BC::Store::PressureController {
static const QString key("PressureController");
static const QString pSetPoint("SetPoint");
static const QString pcEnabled("ControlEnabled");
}

class PressureControllerConfig : public HeaderStorage
{
public:
    PressureControllerConfig();

    double d_pressure;
    double d_setPoint;
    bool d_pressureControlMode;


    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

#endif // PRESSURECONTROLLERCONFIG_H
