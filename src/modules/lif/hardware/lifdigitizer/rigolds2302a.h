#ifndef RIGOLDS2302A_H
#define RIGOLDS2302A_H

#include "lifscope.h"

namespace BC::Key::LifDigi {
static const QString rds2302a{"ds2302a"};
static const QString rds2302aName{"Rigol DS2302A Oscilloscope"};
static const QString queryIntervalMs{"queryInterval_ms"};
}

class RigolDS2302A : public LifScope
{
    Q_OBJECT
public:
    explicit RigolDS2302A(QObject *parent = nullptr);

    // HardwareObject interface
protected:
    void initialize() override;
    bool testConnection() override;
    void timerEvent(QTimerEvent *event) override;

    // LifScope interface
public slots:
    void readWaveform() override;
    bool configure(const LifDigitizerConfig &c) override;
    // HardwareObject interface
    void beginAcquisition() override;
    void endAcquisition() override;

private:
    void waitForOpc();
    bool d_acquiring{false};
    int d_timerId{-1};

};

#endif // RIGOLDS2302A_H
