#pragma once
#ifndef VIRTUALLIFSCOPE_H
#define VIRTUALLIFSCOPE_H

#include <hardware/core/lifdigitizer/lifscope.h>

class QTimer;

class VirtualLifScope : public LifScope
{
    Q_OBJECT
public:
    VirtualLifScope(const QString& label, QObject *parent = nullptr);
    ~VirtualLifScope();


public slots:
    // LifScope interface
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

#endif // VIRTUALLIFSCOPE_H
