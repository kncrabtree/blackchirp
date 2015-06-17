#ifndef QC9528_H
#define QC9528_H

#include "pulsegenerator.h"


class Qc9528 : public PulseGenerator
{
public:
    explicit Qc9528(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    // PulseGenerator interface
public slots:
    QVariant read(const int index, const BlackChirp::PulseSetting s);
    double readRepRate();

    bool set(const int index, const BlackChirp::PulseSetting s, const QVariant val);
    bool setRepRate(double d);

private:
    bool pGenWriteCmd(QString cmd);
};

#endif // QC9528_H
