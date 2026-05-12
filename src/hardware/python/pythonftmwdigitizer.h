#ifndef PYTHONFTMWDIGITIZER_H
#define PYTHONFTMWDIGITIZER_H

#include <hardware/core/ftmwdigitizer/ftmwdigitizer.h>

#include "pythonhardwarebase.h"

/*!
 * \brief FtmwDigitizer subclass that dispatches virtual methods to a Python subprocess via IPC
 *
 * PythonFtmwDigitizer launches a Python subprocess (via PythonProcess) that loads a
 * user-written FTMW digitizer driver script. The pure virtual methods required by
 * FtmwDigitizer — prepareForExperiment() — are translated to JSON requests sent over
 * stdin/stdout pipes.
 *
 * Waveform acquisition is push-driven: the Python script runs its own acquisition
 * loop (in a background thread) and calls self.digi.emit_shot() when data is ready.
 * PythonProcess emits waveformReceived(), which PythonFtmwDigitizer connects to
 * onWaveformReceived() to call emitShot() for accumulation.
 *
 * The FtmwDigitizer base class handles hwPrepareForExperiment() (creates the
 * WaveformBuffer), pre-accumulation, and shot dispatching.
 */
class PythonFtmwDigitizer : public FtmwDigitizer, public PythonHardwareBase
{
    Q_OBJECT
public:
    explicit PythonFtmwDigitizer(const QString &label, QObject *parent = nullptr);

public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;
    void readWaveform() override {}  // no-op: waveform is pushed by Python

protected:
    void initialize() override;
    bool testConnection() override;
    void ftmwReadSettings() override;
    void sleep(bool b) override;

private slots:
    void onWaveformReceived(const QByteArray &data, quint64 shotCount);

private:
    QJsonObject configToJson(const FtmwDigitizerConfig &config) const;
    bool jsonToConfig(const QJsonObject &obj, FtmwDigitizerConfig &config) const;
};

#endif // PYTHONFTMWDIGITIZER_H
