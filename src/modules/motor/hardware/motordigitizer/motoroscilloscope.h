#ifndef MOTOROSCILLOSCOPE_H
#define MOTOROSCILLOSCOPE_H

#include <src/hardware/core/hardwareobject.h>

#include <src/modules/motor/data/motorscan.h>

namespace BC::Key {
static const QString motorScope("motorScope");
}

class MotorOscilloscope : public HardwareObject
{
    Q_OBJECT
public:
    MotorOscilloscope(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent = nullptr, bool threaded=true,bool critical=true);

public slots:
    virtual bool configure(const BlackChirp::MotorScopeConfig &sc) =0;

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override final;

signals:
    void traceAcquired(QVector<double> d);

protected:
    virtual MotorScan prepareForMotorScan(MotorScan s) =0;
    BlackChirp::MotorScopeConfig d_config;

};

#if BC_MOTORSCOPE==1
#include "pico2206b.h"
class Pico2206B;
typedef Pico2206B MotorScopeHardware;
#else
#include "virtualmotorscope.h"
class VirtualMotorScope;
typedef VirtualMotorScope MotorScopeHardware;
#endif

#endif // MOTOROSCILLOSCOPE_H
