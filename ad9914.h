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
    void readSettings();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();

private:
    QByteArray d_settingsHex;
    double d_clockFreqHz;

protected:
    void initialize();
    bool testConnection();
};

#endif // AD9914_H
