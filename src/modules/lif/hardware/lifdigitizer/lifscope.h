#ifndef LIFSCOPE_H
#define LIFSCOPE_H

#include <hardware/core/hardwareobject.h>
#include <modules/lif/hardware/lifdigitizer/lifdigitizerconfig.h>
#include <modules/lif/data/lifconfig.h>

namespace BC::Key::LifDigi {
static const QString lifScope{"LifDigitizer"};
}

class LifScope : public HardwareObject, protected LifDigitizerConfig
{
    Q_OBJECT
public:
    LifScope(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent = nullptr, bool threaded=true,bool critical=true);
    virtual ~LifScope();

signals:
    void waveformRead(QVector<qint8>);
    void configAcqComplete(QPrivateSignal);

public slots:
    bool prepareForExperiment(Experiment &exp) override final;
    virtual void startConfigurationAcquisition(const LifConfig &c);

    virtual void readWaveform() =0;
    virtual bool configure(const LifDigitizerConfig &c) =0;

private:
    void writeSettings();

};

#endif // LIFSCOPE_H
