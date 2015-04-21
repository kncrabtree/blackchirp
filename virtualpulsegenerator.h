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
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    // PulseGenerator interface
public slots:
    QVariant read(const int index, const PulseGenConfig::Setting s);

    void set(const int index, const PulseGenConfig::Setting s, const QVariant val);
    void setRepRate(double d);

};

#endif // VIRTUALPULSEGENERATOR_H
