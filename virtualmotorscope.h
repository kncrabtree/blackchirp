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

    // MotorOscilloscope interface
    bool configure(const BlackChirp::MotorScopeConfig &sc);
    MotorScan prepareForMotorScan(MotorScan s);
    void queryScope();

protected:
    bool testConnection();
    void initialize();


private:
    QTimer *p_testTimer;


    // HardwareObject interface
public slots:
    virtual void beginAcquisition();
    virtual void endAcquisition();
};

#endif // VIRTUALMOTORSCOPE_H
