#ifndef VIRTUALPRESSURECONTROLLER_H
#define VIRTUALPRESSURECONTROLLER_H

#include <src/hardware/optional/pressurecontroller/pressurecontroller.h>

class VirtualPressureController : public PressureController
{
   Q_OBJECT
public:
    VirtualPressureController(QObject *parent =nullptr);
    ~VirtualPressureController();

    // HardwareObject interface
public slots:
    void readSettings() override;

    // PressureController interface
public slots:
    double readPressure() override;
    double setPressureSetpoint(const double val) override;
    double readPressureSetpoint() override;
    void setPressureControlMode(bool enabled) override;
    bool readPressureControlMode() override;
    void openGateValve() override;
    void closeGateValve() override;

protected:
    bool testConnection() override;
    void pcInitialize() override;


private:
    QTimer *p_readTimer;
    double randPressure;
};

#endif // VIRTUALPRESSURECONTROLLER_H
