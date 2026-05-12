#ifndef PYTHONLIFDIGITIZER_H
#define PYTHONLIFDIGITIZER_H

#include <hardware/core/lifdigitizer/lifdigitizer.h>
#include <hardware/core/hardwareregistry.h>

#include "pythonhardwarebase.h"

/*!
 * \brief LifDigitizer subclass that dispatches virtual methods to a Python subprocess via IPC
 *
 * PythonLifDigitizer launches a Python subprocess (via PythonProcess) that loads a
 * user-written LIF digitizer driver script. The pure virtual methods required by
 * LifDigitizer — configure() and readWaveform() — are translated to JSON requests
 * sent over stdin/stdout pipes.
 *
 * LifDigitizer::prepareForExperiment() is final and calls configure() internally.
 * PythonLifDigitizer overrides configure() to serialize the config to JSON IPC
 * and deserialize the validated config back into *this.
 *
 * Waveform acquisition is push-driven: the Python script runs its own acquisition
 * loop (in a background thread) and calls self.digi.emit_shot() when data is
 * ready. PythonProcess emits waveformReceived(), which PythonLifDigitizer connects to
 * onWaveformReceived() to convert and forward via emitWaveform().
 */
class PythonLifDigitizer : public LifDigitizer, public PythonHardwareBase
{
    Q_OBJECT
public:
    explicit PythonLifDigitizer(const QString &label, QObject *parent = nullptr);

public slots:
    bool configure(const LifDigitizerConfig &c) override;
    void beginAcquisition() override;
    void endAcquisition() override;
    void readWaveform() override {}  // no-op: waveform is pushed by Python

protected:
    void initialize() override;
    bool testConnection() override;
    void lifDigitizerReadSettings() override;
    void sleep(bool b) override;

private slots:
    void onWaveformReceived(const QByteArray &data, quint64 shotCount);

private:
    QJsonObject configToJson(const LifDigitizerConfig &config) const;
    bool jsonToConfig(const QJsonObject &obj, LifDigitizerConfig &config) const;
};

#endif // PYTHONLIFDIGITIZER_H
