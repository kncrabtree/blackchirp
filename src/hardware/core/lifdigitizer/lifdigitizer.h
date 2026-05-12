#ifndef LIFDIGITIZER_H
#define LIFDIGITIZER_H

#include <hardware/core/hardwareobject.h>
#include <data/lif/lifdigitizerconfig.h>
#include <data/lif/lifconfig.h>

class LifDigitizer : public HardwareObject, protected LifDigitizerConfig
{
    Q_OBJECT
public:
    LifDigitizer(const QString& impl, const QString& label, QObject *parent = nullptr);
    virtual ~LifDigitizer();

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
    void hwReadSettings() override final;
    /*!
     * \brief Driver hook called after LifDigitizer base settings are refreshed. Default is a no-op.
     */
    virtual void lifDigitizerReadSettings() {}

    void emitWaveform(const QVector<qint8> &data);
    bool d_acquisitionGated{false};
    int d_discardCount{0};

private:
    void writeSettings();

};

#endif // LIFDIGITIZER_H
