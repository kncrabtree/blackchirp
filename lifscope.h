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
    void waveformRead(QByteArray);

public slots:
    virtual void setLifVScale(double scale) =0;
    virtual void setRefVScale(double scale) =0;
    virtual void setHorizontalConfig(double sampleRate, int recLen) =0;
    virtual void queryScope() =0;

protected:
    LifConfig::LifScopeConfig d_config;
};

#endif // LIFSCOPE_H
