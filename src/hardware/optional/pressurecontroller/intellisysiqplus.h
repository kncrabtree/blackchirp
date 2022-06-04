#ifndef INTELLISYSIQPLUS_H
#define INTELLISYSIQPLUS_H

#include <QTimer>

#include <hardware/optional/pressurecontroller/pressurecontroller.h>

namespace BC::Key::PController {
static const QString iqplus{"IntellisysIQPlus"};
static const QString iqplusName("Intellisys IQ Plus Pressure Controller");
}

class IntellisysIQPlus : public PressureController
{
    Q_OBJECT
public:
    IntellisysIQPlus(QObject *parent =nullptr);

    // PressureController interface
public slots:
    double hwReadPressure() override;
    double hwSetPressureSetpoint(const double val) override;
    double hwReadPressureSetpoint() override;
    void hwSetPressureControlMode(bool enabled) override;
    int hwReadPressureControlMode() override;
    void hwOpenGateValve() override;
    void hwCloseGateValve() override;

protected:
    bool pcTestConnection() override;
    void pcInitialize() override;


private:
    double d_fullScale;
    bool d_pcOn;

};

#endif // INTELLISYSIQPLUS_H
