#ifndef VIRTUALTEMPCONTROLLER_H
#define VIRTUALTEMPCONTROLLER_H
#include "temperaturecontroller.h"

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
    double readHWTemperature() override;
};

#endif // VIRTUALTEMPCONTROLLER_H
