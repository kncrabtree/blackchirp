#include "pythonawg.h"

#include <QJsonArray>
#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonAwg, "Python AWG (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonAwg, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_SETTINGS(PythonAwg,
    {BC::Key::AWG::rate,      "Sample Rate (Hz)", "DAC output sample rate",
     16e9, 1e6, 100e9, HwSettingPriority::Important},
    {BC::Key::AWG::samples,   "Max Samples",      "Maximum waveform sample count",
     2e9, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::AWG::min,       "Min Freq (MHz)",    "Minimum chirp frequency",
     100.0, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::AWG::max,       "Max Freq (MHz)",    "Maximum chirp frequency",
     6250.0, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::AWG::prot,      "Protection Pulse",  "AWG outputs a protection pulse",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::AWG::amp,       "Amp Enable Pulse",  "AWG outputs an amplifier enable pulse",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::AWG::rampOnly,  "Ramp Only",         "Restrict to linear ramp chirps",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::AWG::triggered, "Triggered",         "AWG waits for external trigger",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

// ============================================================================
// Constructor / Destructor
// ============================================================================
PythonAwg::PythonAwg(const QString &label, QObject *parent) :
    AWG(QString(PythonAwg::staticMetaObject.className()), label, parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;
    d_critical = false;
}

// ============================================================================
// initialize()
// ============================================================================
void PythonAwg::initialize()
{
    initPythonProcess(p_comm,
        [this](const QString &key, const QVariant &defaultVal) -> QVariant {
            return get(key, defaultVal);
        },
        [this](const QString &key, const QVariant &val) {
            set(key, val, true);
        }
    );

    connect(pu_process.get(), &PythonProcess::logMessage,
            this, &PythonAwg::logMessage);
}

// ============================================================================
// testConnection()
// ============================================================================
bool PythonAwg::testConnection()
{
    if (!testPythonConnection(p_comm)) {
        d_errorString = pythonErrorString();
        return false;
    }
    return true;
}

// ============================================================================
// readAuxData()
// ============================================================================
AuxDataStorage::AuxDataMap PythonAwg::readAuxData()
{
    if (!pu_process || !pu_process->isRunning())
        return {};

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("read_aux_data");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return {};

    return parseAuxDataResult(resp);
}

// ============================================================================
// readValidationData()
// ============================================================================
AuxDataStorage::AuxDataMap PythonAwg::readValidationData()
{
    if (!pu_process || !pu_process->isRunning())
        return {};

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("read_validation_data");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return {};

    return parseAuxDataResult(resp);
}

// ============================================================================
// prepareForExperiment()
// ============================================================================
bool PythonAwg::prepareForExperiment(Experiment &exp)
{
    if (!pu_process || !pu_process->isRunning())
        return true;

    auto auxData = readAuxData();
    for (auto it = auxData.cbegin(); it != auxData.cend(); ++it)
        exp.auxData()->registerKey(d_key, it->first);

    auto valData = readValidationData();
    for (auto it = valData.cbegin(); it != valData.cend(); ++it)
        exp.auxData()->registerKey(d_key, it->first);

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("prepare_for_experiment");

    QJsonObject config;
    config[QStringLiteral("number")] = exp.d_number;
    config[QStringLiteral("ftmw_enabled")] = exp.ftmwEnabled();

    if (exp.ftmwEnabled()) {
        const auto &rfConfig = exp.ftmwConfig()->d_rfConfig;
        const auto &cc = rfConfig.d_chirpConfig;

        // Chirp segments: compact representation sufficient for DDS-style AWGs.
        // For memory-based AWGs, use _compute_waveform() in the Python template.
        QJsonArray segmentsArray;
        for (const auto &chirp : cc.chirpList()) {
            QJsonArray chirpArray;
            for (const auto &seg : chirp) {
                QJsonObject segObj;
                segObj[QStringLiteral("start_freq_mhz")] = seg.startFreqMHz;
                segObj[QStringLiteral("end_freq_mhz")] = seg.endFreqMHz;
                segObj[QStringLiteral("duration_us")] = seg.durationUs;
                segObj[QStringLiteral("alpha_us")] = seg.alphaUs;
                segObj[QStringLiteral("empty")] = seg.empty;
                chirpArray.append(segObj);
            }
            segmentsArray.append(chirpArray);
        }

        QJsonObject chirpObj;
        chirpObj[QStringLiteral("segments")] = segmentsArray;
        chirpObj[QStringLiteral("num_chirps")] = cc.numChirps();
        chirpObj[QStringLiteral("chirp_interval_us")] = cc.chirpInterval();
        chirpObj[QStringLiteral("pre_chirp_protection_us")] = cc.preChirpProtectionDelay();
        chirpObj[QStringLiteral("post_chirp_protection_us")] = cc.postChirpProtectionDelay();
        chirpObj[QStringLiteral("pre_chirp_gate_us")] = cc.preChirpGateDelay();
        chirpObj[QStringLiteral("post_chirp_gate_us")] = cc.postChirpGateDelay();
        chirpObj[QStringLiteral("sample_rate_hz")] = get<double>(BC::Key::AWG::rate);
        config[QStringLiteral("chirp")] = chirpObj;

        // RF chain parameters and clock assignments
        QJsonObject rfObj;
        rfObj[QStringLiteral("awg_mult")] = rfConfig.d_awgMult;
        rfObj[QStringLiteral("chirp_mult")] = rfConfig.d_chirpMult;
        rfObj[QStringLiteral("up_mix_sideband")] = static_cast<int>(rfConfig.d_upMixSideband);
        rfObj[QStringLiteral("down_mix_sideband")] = static_cast<int>(rfConfig.d_downMixSideband);

        QJsonObject clocksObj;
        const auto clocks = rfConfig.getClocks();
        for (auto it = clocks.cbegin(); it != clocks.cend(); ++it) {
            QJsonObject clkObj;
            clkObj[QStringLiteral("freq_mhz")] = rfConfig.clockFrequency(it.key());
            clkObj[QStringLiteral("hw_key")] = it.value().hwKey;
            clkObj[QStringLiteral("output")] = it.value().output;
            clocksObj[QString::number(static_cast<int>(it.key()))] = clkObj;
        }
        rfObj[QStringLiteral("clocks")] = clocksObj;
        config[QStringLiteral("rf_config")] = rfObj;
    }

    req[QStringLiteral("config")] = config;

    auto resp = pu_process->sendRequest(req);
    if (resp.contains(QStringLiteral("error"))) {
        d_errorString = resp[QStringLiteral("error")].toString();
        emit logMessage(QString("PythonAwg (%1): prepareForExperiment error: %2")
                            .arg(d_key, d_errorString),
                        LogHandler::Error);
        return false;
    }
    return resp[QStringLiteral("result")].toBool(true);
}

// ============================================================================
// beginAcquisition()
// ============================================================================
void PythonAwg::beginAcquisition()
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("begin_acquisition");
    pu_process->sendRequest(req);
}

// ============================================================================
// endAcquisition()
// ============================================================================
void PythonAwg::endAcquisition()
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req[QStringLiteral("method")] = QStringLiteral("end_acquisition");
    pu_process->sendRequest(req);
}

// ============================================================================
// sleep()
// ============================================================================
void PythonAwg::sleep(bool b)
{
    pythonSleep(b);
}

// ============================================================================
// readSettings()
// ============================================================================
void PythonAwg::readSettings()
{
    pythonReadSettings();
}

// ============================================================================
// parseAuxDataResult()
// ============================================================================
AuxDataStorage::AuxDataMap PythonAwg::parseAuxDataResult(const QJsonObject &response)
{
    AuxDataStorage::AuxDataMap out;
    QJsonObject result = response[QStringLiteral("result")].toObject();
    for (auto it = result.begin(); it != result.end(); ++it)
        out[it.key()] = QVariant(it.value().toDouble());
    return out;
}
