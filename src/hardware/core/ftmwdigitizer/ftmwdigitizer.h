#ifndef FTMWDIGITIZER_H
#define FTMWDIGITIZER_H

#include <hardware/core/hardwareobject.h>
#include <data/settings/hardwarekeys.h>

#include <QByteArray>

#include <QVector>

#include <data/experiment/ftmwconfig.h>
#include <data/experiment/hardware/core/ftmwdigitizerconfig.h>
#include <data/storage/waveformbuffer.h>
#include <memory>

class FtmwDigitizer : public HardwareObject, protected FtmwDigitizerConfig
{
    Q_OBJECT
public:
    explicit FtmwDigitizer(const QString& impl, const QString& label, QObject *parent = nullptr);
    virtual ~FtmwDigitizer();

signals:
    void shotAcquired(const QByteArray data);

public slots:
    virtual void readWaveform() =0;
    virtual bool hwPrepareForExperiment(Experiment &exp) override final;
    void setAcquisitionGated(bool gated);
    virtual void flushAcquisitionBuffer() {}

public:
    WaveformBuffer* waveformBuffer() const { return pu_waveformBuffer.get(); }

protected:
    void hwReadSettings() override final;
    /*!
     * \brief Driver hook called after FtmwDigitizer base settings are refreshed. Default is a no-op.
     */
    virtual void ftmwReadSettings() {}

    void emitShot(const QByteArray &data);
    bool d_acquisitionGated{false};
    int d_discardCount{0};

private:
    std::unique_ptr<WaveformBuffer> pu_waveformBuffer;

    // Pre-accumulation state (producer thread only, no synchronization needed)
    bool d_preAccumulating{false};
    quint64 d_preAccumShots{0};
    QVector<qint64> d_preAccumData;
    quint8 d_bitShift{0};

    void parseAndAccumulate(const QByteArray &data);
    bool flushPreAccumulated();
    void resetPreAccumulation();

    void writeSettings();
};

#endif // FTMWDIGITIZER_H
