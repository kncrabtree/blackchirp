#ifndef INTELLISYSIQPLUS_H
#define INTELLISYSIQPLUS_H

#include <QTimer>

#include "pressurecontroller.h"

class IntellisysIQPlus : public PressureController
{
    Q_OBJECT
public:
    IntellisysIQPlus(QObject *parent =nullptr);

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
    double d_fullScale;
    QTimer *p_readTimer;

};

#endif // INTELLISYSIQPLUS_H
