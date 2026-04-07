#ifndef PYTHONLIFSCOPE_H
#define PYTHONLIFSCOPE_H

#ifdef BC_PYTHON_HARDWARE

#include <hardware/core/lifdigitizer/lifscope.h>
#include <hardware/core/hardwareregistry.h>

#include "pythonhardwarebase.h"

/*!
 * \brief LifScope subclass that dispatches virtual methods to a Python subprocess via IPC
 *
 * PythonLifScope launches a Python subprocess (via PythonProcess) that loads a
 * user-written LIF digitizer driver script. The pure virtual methods required by
 * LifScope — configure() and readWaveform() — are translated to JSON requests
 * sent over stdin/stdout pipes.
 *
 * LifScope::prepareForExperiment() is final and calls configure() internally.
 * PythonLifScope overrides configure() to serialize the config to JSON IPC
 * and deserialize the validated config back into *this.
 *
 * Waveform acquisition is push-driven: the Python script runs its own acquisition
 * loop (in a background thread) and calls self.scope.emit_shot() when data is
 * ready. PythonProcess emits waveformReceived(), which PythonLifScope connects to
 * onWaveformReceived() to convert and forward via emitWaveform().
 */
class PythonLifScope : public LifScope, public PythonHardwareBase
{
    Q_OBJECT
public:
    explicit PythonLifScope(const QString &label, QObject *parent = nullptr);

    static QVector<HwConfigParam> configParams();

public slots:
    bool configure(const LifDigitizerConfig &c) override;
    void beginAcquisition() override;
    void endAcquisition() override;
    void readWaveform() override {}  // no-op: waveform is pushed by Python

protected:
    void initialize() override;
    bool testConnection() override;
    void readSettings() override;
    void sleep(bool b) override;
    QStringList forbiddenKeys() const override;

private slots:
    void onWaveformReceived(const QByteArray &data, quint64 shotCount);

private:
    QJsonObject configToJson(const LifDigitizerConfig &config) const;
    bool jsonToConfig(const QJsonObject &obj, LifDigitizerConfig &config) const;
};

#endif // BC_PYTHON_HARDWARE
#endif // PYTHONLIFSCOPE_H
