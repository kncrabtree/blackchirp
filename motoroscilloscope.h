#ifndef MOTOROSCILLOSCOPE_H
#define MOTOROSCILLOSCOPE_H

#include "hardwareobject.h"

#include "motorscan.h"

class MotorOscilloscope : public HardwareObject
{
    Q_OBJECT
public:
    MotorOscilloscope(QObject *parent = nullptr);

public slots:
    virtual bool configure(const BlackChirp::MotorScopeConfig &sc) =0;
    virtual MotorScan prepareForMotorScan(MotorScan s) =0;

    // HardwareObject interface
public slots:
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

signals:
    void traceAcquired(QVector<double> d);

protected:
    BlackChirp::MotorScopeConfig d_config;

};

#if BC_MOTORSCOPE==1
#include "pico2206b.h"
class Pico2206B;
typedef Pico2206B MotorScopeHardware;
#else
#include "virtualmotorcontroller.h"
class VirtualMotorScope;
typedef VirtualMotorScope MotorScopeHardware;
#endif

#endif // MOTOROSCILLOSCOPE_H
