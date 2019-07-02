#ifndef MKS947_H
#define MKS947_H

#include "flowcontroller.h"

class Mks946 : public FlowController
{
    Q_OBJECT
public:
    Mks946(QObject *parent = nullptr);

    // HardwareObject interface
protected:
    bool testConnection() override;

    // FlowController interface
public slots:
    double setFlowSetpoint(const int ch, const double val) override;
    double setPressureSetpoint(const double val) override;
    double readFlowSetpoint(const int ch) override;
    double readPressureSetpoint() override;
    double readFlow(const int ch) override;
    double readPressure() override;
    void setPressureControlMode(bool enabled) override;
    bool readPressureControlMode() override;

protected:
    void fcInitialize() override;
    bool mksWrite(QString cmd);
    QByteArray mksQuery(QString cmd);

    int d_address;
    int d_pressureChannel;

    // HardwareObject interface
public slots:
    void readSettings() override;
    void sleep(bool b) override;
};

#endif // MKS947_H
