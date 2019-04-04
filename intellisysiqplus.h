#ifndef INTELLISYSIQPLUS_H
#define INTELLISYSIQPLUS_H

#include <QTimer>

#include "pressurecontroller.h"

class IntellisysIQPlus : public PressureController
{
    Q_OBJECT
public:
    IntellisysIQPlus(QObject *parent =nullptr);

    // HardwareObject interface
public slots:
    void readSettings();
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();

    // PressureController interface
public slots:
    double readPressure();
    double setPressureSetpoint(const double val);
    double readPressureSetpoint();
    void setPressureControlMode(bool enabled);
    bool readPressureControlMode();
    void openGateValve();
    void closeGateValve();

private:
    double fullScale;
    QTimer *p_readTimer;

};

#endif // INTELLISYSIQPLUS_H
