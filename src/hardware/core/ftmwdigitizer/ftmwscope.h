#ifndef FTMWSCOPE_H
#define FTMWSCOPE_H

#include <hardware/core/hardwareobject.h>
#include <data/settings/hardwarekeys.h>

#include <QByteArray>

#include <data/experiment/ftmwconfig.h>
#include <data/experiment/hardware/core/ftmwdigitizerconfig.h>
#include <data/storage/waveformbuffer.h>
#include <memory>

class FtmwScope : public HardwareObject, protected FtmwDigitizerConfig
{
    Q_OBJECT
public:
    explicit FtmwScope(const QString& impl, const QString& label, QObject *parent = nullptr);
    virtual ~FtmwScope();

signals:
    void shotAcquired(const QByteArray data);

public slots:
    virtual void readWaveform() =0;
    virtual bool hwPrepareForExperiment(Experiment &exp) override final;
    void setAcquisitionGated(bool gated);
    virtual void flushAcquisitionBuffer() {}

    // HardwareObject interface
public slots:
    QStringList forbiddenKeys() const override;

public:
    WaveformBuffer* waveformBuffer() const { return pu_waveformBuffer.get(); }

protected:
    void emitShot(const QByteArray &data);
    bool d_acquisitionGated{false};
    int d_discardCount{0};

private:
    std::unique_ptr<WaveformBuffer> pu_waveformBuffer;
    void writeSettings();
};

#endif // FTMWSCOPE_H
