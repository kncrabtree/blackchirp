#ifndef VIRTUALPULSEGENERATOR_H
#define VIRTUALPULSEGENERATOR_H

#include "pulsegenerator.h"

class VirtualPulseGenerator : public PulseGenerator
{
    Q_OBJECT
public:
    explicit VirtualPulseGenerator(QObject *parent = nullptr);
    ~VirtualPulseGenerator();

    // HardwareObject interface
public slots:
    void readSettings();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();

    // PulseGenerator interface
    QVariant read(const int index, const BlackChirp::PulseSetting s);
    double readRepRate();

    bool set(const int index, const BlackChirp::PulseSetting s, const QVariant val);
    bool setRepRate(double d);

protected:
    bool testConnection();
    void initialize();
};

#endif // VIRTUALPULSEGENERATOR_H
