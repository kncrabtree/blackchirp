#ifndef LIFSCOPE_H
#define LIFSCOPE_H

#include "hardwareobject.h"
#include "lifconfig.h"

class LifScope : public HardwareObject
{
    Q_OBJECT
public:
    LifScope(QObject *parent = nullptr);
    ~LifScope();

signals:
    void waveformRead(LifConfig::LifScopeConfig, QByteArray);

public slots:
    virtual void setLifVScale(double scale) =0;
    virtual void setRefVScale(double scale) =0;
    virtual void setHorizontalConfig(double sampleRate, int recLen) =0;
    virtual void setRefEnabled(bool en) =0;
    virtual void queryScope() =0;

protected:
    LifConfig::LifScopeConfig d_config;
    bool d_refEnabled;
};

#if BC_LIFSCOPE == 1
#include "dpo3012.h"
class Dpo3012;
typedef Dpo3012 LifScopeHardware;
#else
#include "virtuallifscope.h"
class VirtualLifScope;
typedef VirtualLifScope LifScopeHardware;
#endif

#endif // LIFSCOPE_H
