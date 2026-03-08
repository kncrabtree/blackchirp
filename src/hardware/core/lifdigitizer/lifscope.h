#ifndef LIFSCOPE_H
#define LIFSCOPE_H

#include <hardware/core/hardwareobject.h>
#include <data/lif/lifdigitizerconfig.h>
#include <data/lif/lifconfig.h>

class LifScope : public HardwareObject, protected LifDigitizerConfig
{
    Q_OBJECT
public:
    LifScope(const QString& impl, const QString& label, QObject *parent = nullptr);
    virtual ~LifScope();

signals:
    void waveformRead(QVector<qint8>);
    void configAcqComplete(QPrivateSignal);

public slots:
    bool prepareForExperiment(Experiment &exp) override final;
    virtual void startConfigurationAcquisition(const LifConfig &c);
    void setAcquisitionGated(bool gated);
    virtual void flushAcquisitionBuffer() {}

    virtual void readWaveform() =0;
    virtual bool configure(const LifDigitizerConfig &c) =0;

protected:
    void emitWaveform(const QVector<qint8> &data);
    bool d_acquisitionGated{false};
    int d_discardCount{0};

private:
    void writeSettings();

};

#endif // LIFSCOPE_H
