#ifndef MOTORCONTROLLER_H
#define MOTORCONTROLLER_H

#include "hardwareobject.h"

#include "motorscan.h"

class QTimer;

class MotorController : public HardwareObject
{
    Q_OBJECT
public:
    MotorController(QObject *parent = nullptr);

signals:
    void motionComplete();
    void limitStatus(bool nx, bool px, bool ny, bool py, bool nz, bool pz);
    void posUpdate(BlackChirp::MotorAxis axis, double pos);

public slots:
    virtual void moveToPosition(double x, double y, double z) =0;
    virtual Experiment prepareForExperiment(Experiment exp);
    virtual bool prepareForMotorScan(const MotorScan ms) =0;
    virtual void moveToRestingPos() =0;
    virtual void checkLimit() =0;

protected:
    double d_xPos, d_yPos, d_zPos;
    QPair<double,double> d_xRange, d_yRange, d_zRange;
    double d_xRestingPos, d_yRestingPos, d_zRestingPos;
    QTimer *p_limitTimer;
};

#endif // MOTORCONTROLLER_H
