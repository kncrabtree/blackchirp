#ifndef VIRTUALMOTORCONTROLLER_H
#define VIRTUALMOTORCONTROLLER_H

#include "motorcontroller.h"

class VirtualMotorController : public MotorController
{
    Q_OBJECT
public:
    VirtualMotorController(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    // MotorController interface
public slots:
    void moveToPosition(double x, double y, double z);
    bool prepareForMotorScan(const MotorScan ms);
    void moveToRestingPos();
    void checkLimit();
};

#endif // VIRTUALMOTORCONTROLLER_H
