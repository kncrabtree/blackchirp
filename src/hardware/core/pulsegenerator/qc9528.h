#ifndef QC9528_H
#define QC9528_H

#include <src/hardware/core/pulsegenerator/pulsegenerator.h>

namespace BC::Key {
static const QString qc9528("qc9528");
static const QString qc9528Name("Pulse Generator QC 9528");
}

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
    void readSettings() override final;


protected:
    bool testConnection() override;
    void initializePGen() override;


private:
    bool pGenWriteCmd(QString cmd);
    bool d_forceExtClock;
};

#endif // QC9528_H
