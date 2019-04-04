#ifndef VIRTUALMOTORSCOPE_H
#define VIRTUALMOTORSCOPE_H

#include "motoroscilloscope.h"

class QTimer;

class VirtualMotorScope : public MotorOscilloscope
{
    Q_OBJECT
public:
    VirtualMotorScope(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    void readSettings();
    bool testConnection();
    void initialize();

    // MotorOscilloscope interface
public slots:
    bool configure(const BlackChirp::MotorScopeConfig &sc);
    MotorScan prepareForMotorScan(MotorScan s);
    void queryScope();

private:
    QTimer *p_testTimer;


    // HardwareObject interface
public slots:
    virtual void beginAcquisition();
    virtual void endAcquisition();
};

#endif // VIRTUALMOTORSCOPE_H
