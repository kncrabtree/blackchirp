#ifndef LIFSCOPE_H
#define LIFSCOPE_H

#include "hardwareobject.h"
#include "lifconfig.h"

class LifScope : public HardwareObject
{
    Q_OBJECT
public:
    LifScope(QObject *parent = nullptr);
    virtual ~LifScope();

signals:
    void waveformRead(const LifTrace);
    void configUpdated(const BlackChirp::LifScopeConfig);

public slots:
    void setAll(const BlackChirp::LifScopeConfig c);
    virtual void setLifVScale(double scale) =0;
    virtual void setRefVScale(double scale) =0;
    virtual void setHorizontalConfig(double sampleRate, int recLen) =0;
    virtual void setRefEnabled(bool en) =0;
    virtual void queryScope() =0;

protected:
    BlackChirp::LifScopeConfig d_config;
};

#ifdef BC_LIFSCOPE
#if BC_LIFSCOPE == 1
#include "m4i2211x8.h"
class M4i2211x8;
using LifScopeHardware = M4i2211x8;
#else
#include "virtuallifscope.h"
class VirtualLifScope;
using LifScopeHardware = VirtualLifScope;
#endif
#endif

#endif // LIFSCOPE_H
