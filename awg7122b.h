#ifndef AWG7122B_H
#define AWG7122B_H

#include "awg.h"

#include "chirpconfig.h"

class AWG7122B : public AWG
{
    Q_OBJECT
public:
    explicit AWG7122B(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

private:
    QString getWaveformKey(const ChirpConfig cc);
    QString writeWaveform(const ChirpConfig cc);
};

#endif // AWG7122B_H
