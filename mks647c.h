#ifndef MKS647C_H
#define MKS647C_H

#include "flowcontroller.h"

class Mks647c : public FlowController
{
public:
    explicit Mks647c(QObject *parent = nullptr);

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

    // HardwareObject interface
    void sleep(bool b) override;

protected:
    bool testConnection() override;

    // FlowController interface
    virtual void fcInitialize() override;



private:
    QList<double> d_gasRangeList;
    QList<double> d_pressureRangeList;
    int d_pressureRangeIndex;
    QList<int> d_rangeIndexList;
    QList<double> d_gcfList;

    QByteArray mksQueryCmd(QString cmd, int respLength);
    int d_maxTries;
};

#endif // MKS647C_H
