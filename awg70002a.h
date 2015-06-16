#ifndef AWG70002A_H
#define AWG70002A_H

#include "awg.h"

class AWG70002a : public AWG
{
    Q_OBJECT
public:
    explicit AWG70002a(QObject *parent = nullptr);

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

#endif // AWG70002A_H
