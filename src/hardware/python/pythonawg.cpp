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
    {BC::Key::AWG::markerCount, "Marker Count", "Number of physical marker output channels",
     0, 0, QVariant{}, HwSettingPriority::Required}
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
    req["method"_L1] = "read_aux_data"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
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
    req["method"_L1] = "read_validation_data"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
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
    req["method"_L1] = "prepare_for_experiment"_L1;

    QJsonObject config;
    config["number"_L1] = exp.d_number;
    config["ftmw_enabled"_L1] = exp.ftmwEnabled();

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
                segObj["start_freq_mhz"_L1] = seg.startFreqMHz;
                segObj["end_freq_mhz"_L1] = seg.endFreqMHz;
                segObj["duration_us"_L1] = seg.durationUs;
                segObj["alpha_us"_L1] = seg.alphaUs;
                segObj["empty"_L1] = seg.empty;
                chirpArray.append(segObj);
            }
            segmentsArray.append(chirpArray);
        }

        QJsonObject chirpObj;
        chirpObj["segments"_L1] = segmentsArray;
        chirpObj["num_chirps"_L1] = cc.numChirps();
        chirpObj["chirp_interval_us"_L1] = cc.chirpInterval();
        QJsonArray markersArray;
        for(const auto &m : cc.markerChannels())
        {
            QJsonObject mObj;
            mObj["name"_L1] = m.name;
            mObj["role"_L1] = static_cast<int>(m.role);
            mObj["start_us"_L1] = m.startTime;
            mObj["end_us"_L1] = m.endTime;
            mObj["enabled"_L1] = m.enabled;
            markersArray.append(mObj);
        }
        chirpObj["markers"_L1] = markersArray;
        chirpObj["sample_rate_hz"_L1] = get<double>(BC::Key::AWG::rate);
        config["chirp"_L1] = chirpObj;

        // RF chain parameters and clock assignments
        QJsonObject rfObj;
        rfObj["awg_mult"_L1] = rfConfig.d_awgMult;
        rfObj["chirp_mult"_L1] = rfConfig.d_chirpMult;
        rfObj["up_mix_sideband"_L1] = static_cast<int>(rfConfig.d_upMixSideband);
        rfObj["down_mix_sideband"_L1] = static_cast<int>(rfConfig.d_downMixSideband);

        QJsonObject clocksObj;
        const auto clocks = rfConfig.getClocks();
        for (auto it = clocks.cbegin(); it != clocks.cend(); ++it) {
            QJsonObject clkObj;
            clkObj["freq_mhz"_L1] = rfConfig.clockFrequency(it.key());
            clkObj["hw_key"_L1] = it.value().hwKey;
            clkObj["output"_L1] = it.value().output;
            clocksObj[QString::number(static_cast<int>(it.key()))] = clkObj;
        }
        rfObj["clocks"_L1] = clocksObj;
        config["rf_config"_L1] = rfObj;
    }

    req["config"_L1] = config;

    auto resp = pu_process->sendRequest(req);
    if (resp.contains("error"_L1)) {
        d_errorString = resp["error"_L1].toString();
        hwError(u"prepareForExperiment error: %1"_s.arg(d_errorString));
        return false;
    }
    return resp["result"_L1].toBool(true);
}

// ============================================================================
// beginAcquisition()
// ============================================================================
void PythonAwg::beginAcquisition()
{
    if (!pu_process || !pu_process->isRunning())
        return;

    QJsonObject req;
    req["method"_L1] = "begin_acquisition"_L1;
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
    req["method"_L1] = "end_acquisition"_L1;
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
    QJsonObject result = response["result"_L1].toObject();
    for (auto it = result.begin(); it != result.end(); ++it)
        out[it.key()] = QVariant(it.value().toDouble());
    return out;
}
