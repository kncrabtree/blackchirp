#ifndef QC9528_H
#define QC9528_H

#include "pulsegenerator.h"


class Qc9528 : public PulseGenerator
{
public:
    explicit Qc9528(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    void beginAcquisition() override;
    void endAcquisition() override;
    void sleep(bool b) override;

    // PulseGenerator interface
    QVariant read(const int index, const BlackChirp::PulseSetting s) override;
    double readRepRate() override;

    bool set(const int index, const BlackChirp::PulseSetting s, const QVariant val) override;
    bool setRepRate(double d) override;


protected:
    bool testConnection() override;
    void initializePGen() override;


private:
    bool pGenWriteCmd(QString cmd);
    bool d_forceExtClock;
};

#endif // QC9528_H
