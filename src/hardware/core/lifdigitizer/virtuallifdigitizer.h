#pragma once
#ifndef VIRTUALLIFDIGITIZER_H
#define VIRTUALLIFDIGITIZER_H

#include <hardware/core/lifdigitizer/lifdigitizer.h>

class QTimer;

class VirtualLifDigitizer : public LifDigitizer
{
    Q_OBJECT
public:
    VirtualLifDigitizer(const QString& label, QObject *parent = nullptr);
    ~VirtualLifDigitizer();


public slots:
    // LifDigitizer interface
    void readWaveform() override;
    virtual bool configure(const LifDigitizerConfig &c) override;

protected:
    bool testConnection() override;
    void initialize() override;

    QTimer *p_timer;

    // HardwareObject interface
public slots:
    void beginAcquisition() override;
    void endAcquisition() override;
};

#endif // VIRTUALLIFDIGITIZER_H
