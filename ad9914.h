#ifndef AD9914_H
#define AD9914_H

#include "awg.h"


class AD9914 : public AWG
{
    Q_OBJECT
public:
    explicit AD9914(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool testConnection();
    void readSettings();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();

private:
    QByteArray d_settingsHex;
    double d_clockFreqHz;
};

#endif // AD9914_H
