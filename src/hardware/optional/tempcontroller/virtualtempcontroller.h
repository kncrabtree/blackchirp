#ifndef VIRTUALTEMPCONTROLLER_H
#define VIRTUALTEMPCONTROLLER_H
#include <src/hardware/optional/tempcontroller/temperaturecontroller.h>

namespace BC::Key {
static const QString vtcName("Virtual Temperature Controller");
}

class VirtualTemperatureController : public TemperatureController
{
   Q_OBJECT
public:
    VirtualTemperatureController(QObject *parent =nullptr);
    ~VirtualTemperatureController();

    // HardwareObject interface
public slots:
    void readSettings() override;

    // TemperatureController interface
public slots:

protected:
    bool testConnection() override;
    void tcInitialize() override;


private:
    QTimer *p_readTimer;
    double randTemperature;

    // TemperatureController interface
protected:
    QList<double> readHWTemperature() override;
};

#endif // VIRTUALTEMPCONTROLLER_H
