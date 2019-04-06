#ifndef QC9518_H
#define QC9518_H

#include "pulsegenerator.h"

class Qc9518 : public PulseGenerator
{
public:
    explicit Qc9518(QObject *parent = nullptr);

	// HardwareObject interface
public slots:
    void readSettings();
	void sleep(bool b);
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


private:
	bool pGenWriteCmd(QString cmd);

};

#endif // QC9518_H
