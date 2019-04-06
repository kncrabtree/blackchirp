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
    void readSettings() override;

    // MotorOscilloscope interface
    bool configure(const BlackChirp::MotorScopeConfig &sc) override;
    MotorScan prepareForMotorScan(MotorScan s) override;
    void queryScope();

protected:
    bool testConnection() override;
    void initialize() override;


private:
    QTimer *p_testTimer;


    // HardwareObject interface
public slots:
    virtual void beginAcquisition() override;
    virtual void endAcquisition() override;
};

#endif // VIRTUALMOTORSCOPE_H
