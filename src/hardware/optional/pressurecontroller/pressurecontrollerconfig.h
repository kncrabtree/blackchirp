#ifndef PRESSURECONTROLLERCONFIG_H
#define PRESSURECONTROLLERCONFIG_H

#include <data/storage/headerstorage.h>

namespace BC::Store::PressureController {
static const QString pSetPoint{"SetPoint"};
static const QString pcEnabled{"ControlEnabled"};
}

class PressureControllerConfig : public HeaderStorage
{
public:
    PressureControllerConfig(QString subKey = QString(""), int index=-1);

    double d_pressure{0.0};
    double d_setPoint{0.0};
    bool d_pressureControlMode{false};


    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

#endif // PRESSURECONTROLLERCONFIG_H
