#ifndef VIRTUALTEMPCONTROLLER_H
#define VIRTUALTEMPCONTROLLER_H
#include <hardware/optional/tempcontroller/temperaturecontroller.h>

class VirtualTemperatureController : public TemperatureController
{
   Q_OBJECT
public:
    VirtualTemperatureController(const QString& label, QObject *parent = nullptr);
    ~VirtualTemperatureController();


    // TemperatureController interface
public slots:

protected:
    bool tcTestConnection() override;
    void tcInitialize() override;

    // TemperatureController interface
protected:
    double readHwTemperature(const uint ch) override;
};

#endif // VIRTUALTEMPCONTROLLER_H
