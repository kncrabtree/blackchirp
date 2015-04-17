#ifndef PULSEGENERATOR_H
#define PULSEGENERATOR_H

#include "rs232instrument.h"
#include "pulsegenconfig.h"

class PulseGenerator : public Rs232Instrument
{
    Q_OBJECT
public:
    PulseGenerator(QObject *parent = 0);
    ~PulseGenerator();



    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    PulseGenConfig config() const;
    QVariant read(const int index, const PulseGenConfig::Setting s);
    PulseGenConfig::ChannelConfig read(const int index);

    void set(const int index, const PulseGenConfig::Setting s, const QVariant val);
    void setChannel(const int index, const PulseGenConfig::ChannelConfig cc);
    void setAll(const PulseGenConfig cc);
    void setRepRate(double d);

signals:
    void settingUpdate(int,PulseGenConfig::Setting,QVariant);
    void configUpdate(const PulseGenConfig);
    void repRateUpdate(double);

private:
    PulseGenConfig d_config;

    void readAll();
};

#endif // PULSEGENERATOR_H
