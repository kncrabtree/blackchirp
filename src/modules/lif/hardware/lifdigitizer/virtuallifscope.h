#pragma once
#ifndef VIRTUALLIFSCOPE_H
#define VIRTUALLIFSCOPE_H

#include <src/modules/lif/hardware/lifdigitizer/lifscope.h>

class QTimer;

class VirtualLifScope : public LifScope
{
    Q_OBJECT
public:
    VirtualLifScope(QObject *parent = nullptr);
    ~VirtualLifScope();


public slots:
    // HardwareObject interface
    void readSettings() override;

    // LifScope interface
    void setLifVScale(double scale) override;
    void setRefVScale(double scale) override;
    void setHorizontalConfig(double sampleRate, int recLen) override;
    void queryScope() override;
    void setRefEnabled(bool en) override;

protected:
    bool testConnection() override;
    void initialize() override;

    QTimer *p_timer;


    // HardwareObject interface
public slots:
    void sleep(bool b) override;
};

#endif // VIRTUALLIFSCOPE_H
