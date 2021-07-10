#ifndef MOTORCONTROLLER_H
#define MOTORCONTROLLER_H

#include <hardware/core/hardwareobject.h>

#include <modules/motor/data/motorscan.h>

class QTimer;

namespace BC::Key::MC {
static const QString key("MotorController");
static const QString mInterval("motionIntervalMs");
static const QString lInterval("limitIntervalMs");
static const QString channels("channels");
static const QString type("type");
static const QString min("min");
static const QString max("max");
static const QString id("id");
static const QString offset("offset");
static const QString rest("restingPos");
static const QString units("units");
static const QString axName("name");
static const QString decimal("decimals");
static const QString xIndex("xIndex");
static const QString yIndex("yIndex");
static const QString zIndex("zIndex");
}

using Limits = QPair<bool,bool>;
using AxisInfo = QPair<int,QString>;

class MotorController : public HardwareObject
{
    Q_OBJECT
public:
    MotorController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent = nullptr, bool threaded=false,bool critical=true);

signals:
    void motionComplete(bool success, QPrivateSignal);
    //void limitStatus(bool nx, bool px, bool ny, bool py, bool nz, bool pz);
    void limitStatus(MotorScan::MotorAxis axis, bool negLimit, bool posLimit, QPrivateSignal);
    void posUpdate(MotorScan::MotorAxis axis, double pos, QPrivateSignal);

public slots:
    bool prepareForExperiment(Experiment &exp) override final;
    bool moveToPosition(double x, double y, double z);
    void moveToRestingPos();
    void readCurrentPosition();
    void checkMotion();
    void checkLimits();

protected:   
    AxisInfo getAxisInfo(MotorScan::MotorAxis a) const;
    virtual void mcInitialize() =0;
    virtual bool mcTestConnection() =0;
    virtual bool prepareForMotorScan(Experiment &exp) =0;
    virtual bool hwMoveToPosition(double x, double y, double z) = 0;
    virtual Limits hwCheckLimits(MotorScan::MotorAxis axis) =0;
    virtual double hwReadPosition(MotorScan::MotorAxis axis) =0;
    virtual bool hwCheckAxisMotion(MotorScan::MotorAxis axis) =0;
    virtual bool hwStopMotion(MotorScan::MotorAxis axis) =0;

private:
    void initialize() override final;
    bool testConnection() override final;
    QMap<MotorScan::MotorAxis,bool> d_moving;
    bool d_idle;
    QTimer *p_limitTimer, *p_motionTimer;
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
