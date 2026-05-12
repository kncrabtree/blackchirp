#ifndef M412220X8_H
#define M412220X8_H

#include <hardware/core/ftmwdigitizer/ftmwdigitizer.h>
#include <hardware/library/spectrumlibrary.h>
#include <hardware/library/spectrumconstants.h>

#include <QTimer>

class M4i2220x8 : public FtmwDigitizer
{
    Q_OBJECT
public:
    explicit M4i2220x8(const QString& label, QObject *parent = nullptr);
    ~M4i2220x8();

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

    // FtmwDigitizer interface
    void readWaveform() override;

protected:
    bool testConnection() override;
    void initialize() override;

private:
    void* p_handle;

    qint64 d_waveformBytes;
    char *p_m4iBuffer;
    int d_bufferSize;

    QTimer *p_timer;

};

#endif // M412220X8_H
