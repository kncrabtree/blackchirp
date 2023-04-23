#ifndef VIRTUALTEMPCONTROLLER_H
#define VIRTUALTEMPCONTROLLER_H
#include <hardware/optional/tempcontroller/temperaturecontroller.h>

namespace BC::Key {
static const QString vtcName("Virtual Temperature Controller");
}

class VirtualTemperatureController : public TemperatureController
{
   Q_OBJECT
public:
    VirtualTemperatureController(QObject *parent =nullptr);
    ~VirtualTemperatureController();


    // TemperatureController interface
public slots:

protected:
    bool tcTestConnection() override;
    void tcInitialize() override;

    // TemperatureController interface
protected:
    double readHwTemperature(const int ch) override;
};

#endif // VIRTUALTEMPCONTROLLER_H
