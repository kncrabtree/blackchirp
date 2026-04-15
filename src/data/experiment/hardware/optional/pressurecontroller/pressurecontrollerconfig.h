#ifndef PRESSURECONTROLLERCONFIG_H
#define PRESSURECONTROLLERCONFIG_H

#include <data/storage/headerstorage.h>

namespace BC::Store::PressureController {
inline constexpr QLatin1StringView pSetPoint{"SetPoint"};
inline constexpr QLatin1StringView pcEnabled{"ControlEnabled"};
}

class PressureControllerConfig : public HeaderStorage
{
public:
    PressureControllerConfig(const QString& hwKey);

    double d_pressure{0.0};
    double d_setPoint{0.0};
    bool d_pressureControlMode{false};


    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

#endif // PRESSURECONTROLLERCONFIG_H
