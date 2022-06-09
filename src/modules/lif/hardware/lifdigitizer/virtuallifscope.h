#pragma once
#ifndef VIRTUALLIFSCOPE_H
#define VIRTUALLIFSCOPE_H

#include <modules/lif/hardware/lifdigitizer/lifscope.h>

class QTimer;

namespace BC::Key {
static const QString vLifScopeName("Virtual LIF Oscilloscope");
}

class VirtualLifScope : public LifScope
{
    Q_OBJECT
public:
    VirtualLifScope(QObject *parent = nullptr);
    ~VirtualLifScope();


public slots:
    // LifScope interface
    void readWaveform() override;

protected:
    bool testConnection() override;
    void initialize() override;

    QTimer *p_timer;


    // HardwareObject interface
public slots:
    void sleep(bool b) override;

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;
};

#endif // VIRTUALLIFSCOPE_H
