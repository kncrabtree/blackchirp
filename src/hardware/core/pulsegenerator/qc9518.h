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

protected:
    bool testConnection() override;
    void initializePGen() override;
    bool setChWidth(const int index, const double width) override;
    bool setChDelay(const int index, const double delay) override;
    bool setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level) override;
    bool setChEnabled(const int index, const bool en) override;
    bool setHwRepRate(double rr) override;
    double readChWidth(const int index) override;
    double readChDelay(const int index) override;
    PulseGenConfig::ActiveLevel readChActiveLevel(const int index) override;
    bool readChEnabled(const int index) override;
    double readHwRepRate() override;


private:
	bool pGenWriteCmd(QString cmd);
};

#endif // QC9518_H
