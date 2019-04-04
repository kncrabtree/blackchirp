#ifndef QC9528_H
#define QC9528_H

#include "pulsegenerator.h"


class Qc9528 : public PulseGenerator
{
public:
    explicit Qc9528(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    void readSettings();
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();

    // PulseGenerator interface
public slots:
    QVariant read(const int index, const BlackChirp::PulseSetting s);
    double readRepRate();

    bool set(const int index, const BlackChirp::PulseSetting s, const QVariant val);
    bool setRepRate(double d);

    void sleep(bool b);

private:
    bool pGenWriteCmd(QString cmd);
    bool d_forceExtClock;
};

#endif // QC9528_H
