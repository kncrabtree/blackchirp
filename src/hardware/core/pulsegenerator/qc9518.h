#ifndef QC9518_H
#define QC9518_H

#include <src/hardware/core/pulsegenerator/pulsegenerator.h>

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
