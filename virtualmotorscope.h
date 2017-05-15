#ifndef VIRTUALMOTORSCOPE_H
#define VIRTUALMOTORSCOPE_H

#include "motoroscilloscope.h"

class VirtualMotorScope : public MotorOscilloscope
{
    Q_OBJECT
public:
    VirtualMotorScope(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();

    // MotorOscilloscope interface
public slots:
    bool configure(const BlackChirp::MotorScopeConfig &sc);
    MotorScan prepareForMotorScan(MotorScan s);
    void queryScope();

signals:
    void configChanged(BlackChirp::MotorScopeConfig sc);

};

#endif // VIRTUALMOTORSCOPE_H
