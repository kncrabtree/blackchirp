#ifndef PICO2206B_H
#define PICO2206B_H

#include "motoroscilloscope.h"
#include <PicoStatus.h>
#include <ps2000aApi.h>

class Pico2206B : public MotorOscilloscope
{
    Q_OBJECT
public:
    Pico2206B(QObject *parent = nullptr);
    ~Pico2206B();

    // HardwareObject interface
public slots:
    void readSettings() override;
    void beginAcquisition() override;
    void endAcquisition() override;

    // MotorOscilloscope interface
public slots:
    bool configure(const BlackChirp::MotorScopeConfig &sc) override;
    MotorScan prepareForMotorScan(MotorScan s) override;

protected:
    bool testConnection() override;
    void initialize() override;


private:
    int16_t d_handle;
    PICO_STATUS status;

    uint32_t timebase;
    int32_t noSamples;
    QTimer *p_acquisitionTimer;
    int16_t isReady;
    bool d_acquiring;
    QVector<qint16> d_buffer;

    void beginScopeAcquisition();
    void endScopeAcquisition();
    void closeConnection();

};



#endif // PICO2206B_H
