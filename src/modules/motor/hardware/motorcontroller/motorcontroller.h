#ifndef MOTORCONTROLLER_H
#define MOTORCONTROLLER_H

#include <src/hardware/core/hardwareobject.h>

#include <src/modules/motor/data/motorscan.h>

class QTimer;

class MotorController : public HardwareObject
{
    Q_OBJECT
public:
    MotorController(QObject *parent = nullptr);

signals:
    void motionComplete(bool success = true);
    //void limitStatus(bool nx, bool px, bool ny, bool py, bool nz, bool pz);
    void limitStatus(BlackChirp::MotorAxis axis, bool negLimit, bool posLimit);

    void posUpdate(BlackChirp::MotorAxis axis, double pos);

public slots:
    virtual bool moveToPosition(double x, double y, double z) =0;
    bool prepareForExperiment(Experiment &exp) override final;
    virtual void moveToRestingPos() =0;
    virtual void checkLimit() =0;

protected:
    virtual bool prepareForMotorScan(Experiment &exp) =0;
    double d_xPos, d_yPos, d_zPos;
    QPair<double,double> d_xRange, d_yRange, d_zRange;
    double d_xRestingPos, d_yRestingPos, d_zRestingPos;
    QTimer *p_limitTimer;
};

#if BC_MOTORCONTROLLER==1
#include "scx11.h"
class Scx11;
typedef Scx11 MotorControllerHardware;
#else
#include "virtualmotorcontroller.h"
class VirtualMotorController;
typedef VirtualMotorController MotorControllerHardware;
#endif

#endif // MOTORCONTROLLER_H
