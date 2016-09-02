#ifndef PICO2206B_H
#define PICO2206B_H

#include "motoroscilloscope.h"

class Pico2206B : public MotorOscilloscope
{
    Q_OBJECT
public:
    Pico2206B(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();

    // MotorOscilloscope interface
public slots:
    bool configure(const BlackChirp::MotorScopeConfig &sc);
    MotorScan prepareForMotorScan(MotorScan s);
};

#endif // PICO2206B_H
