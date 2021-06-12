#ifndef QC9518_H
#define QC9518_H

#include <src/hardware/core/pulsegenerator/pulsegenerator.h>

namespace BC::Key {
static const QString qc9518("QC9518");
static const QString qc9518Name("Pulse Generator QC 9518");
}

class Qc9518 : public PulseGenerator
{
public:
    explicit Qc9518(QObject *parent = nullptr);

	// HardwareObject interface
public slots:
    void sleep(bool b) override;
    void beginAcquisition() override;
    void endAcquisition() override;

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

};

#endif // QC9518_H
