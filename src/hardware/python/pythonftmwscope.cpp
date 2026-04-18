#include "pythonftmwscope.h"

#include <QJsonArray>
#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>
#include <data/bcglobals.h>
#include <data/settings/hardwarekeys.h>

using namespace BC::Key::FtmwScope;
using namespace BC::Key::Digi;

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonFtmwScope, "Python FTMW Digitizer (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonFtmwScope, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_SETTINGS(PythonFtmwScope,
    {numAnalogChannels,  "Analog Channels",  "Number of analog inputs",
     4, 1, 32, HwSettingPriority::Required},
    {numDigitalChannels, "Digital Channels",  "Number of digital inputs",
     0, 0, 32, HwSettingPriority::Required},
    {hasAuxTriggerChannel, "Aux Trigger Channel", "Has auxiliary trigger input",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {minFullScale,       "Min Full Scale (V)", "Minimum full scale voltage",
     5e-2, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxFullScale,       "Max Full Scale (V)", "Maximum full scale voltage",
     2.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {minVOffset,         "Min V Offset (V)",   "Minimum voltage offset",
     -2.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxVOffset,         "Max V Offset (V)",   "Maximum voltage offset",
     2.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {isTriggered,        "Triggered",          "Digitizer uses external trigger",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {minTrigDelay,       "Min Trig Delay (us)", "Minimum trigger delay",
     -10.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxTrigDelay,       "Max Trig Delay (us)", "Maximum trigger delay",
     10.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {minTrigLevel,       "Min Trig Level (V)",  "Minimum trigger level",
     -5.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxTrigLevel,       "Max Trig Level (V)",  "Maximum trigger level",
     5.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxRecordLength,    "Max Record Length",   "Maximum record length in samples",
     100000000, 0, QVariant{}, HwSettingPriority::Optional},
    {canBlockAverage,    "Block Average",       "Supports block averaging",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxAverages,        "Max Averages",        "Maximum number of averages",
     100, 1, QVariant{}, HwSettingPriority::Optional},
    {canMultiRecord,     "Multi Record",        "Supports multi-record acquisition",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxRecords,         "Max Records",         "Maximum number of records",
     100, 1, QVariant{}, HwSettingPriority::Optional},
    {multiBlock,         "Multi Block",         "Can block average and multi-record simultaneously",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxBytes,           "Max Bytes/Point",     "Maximum bytes per data point",
     2, 1, 8, HwSettingPriority::Optional},
    {bandwidth,          "Bandwidth (MHz)",     "Analog bandwidth",
     16000.0, QVariant{}, QVariant{}, HwSettingPriority::Important}
)
REGISTER_HARDWARE_ARRAY(PythonFtmwScope, sampleRates,
    "Sample Rates", "Available digitizer sample rates",
    HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(PythonFtmwScope, sampleRates,
    {{srText, "2 GSa/s"}, {srValue, 2e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(PythonFtmwScope, sampleRates,
    {{srText, "5 GSa/s"}, {srValue, 5e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(PythonFtmwScope, sampleRates,
    {{srText, "10 GSa/s"}, {srValue, 10e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(PythonFtmwScope, sampleRates,
    {{srText, "20 GSa/s"}, {srValue, 20e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(PythonFtmwScope, sampleRates,
    {{srText, "50 GSa/s"}, {srValue, 50e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(PythonFtmwScope, sampleRates,
    {{srText, "100 GSa/s"}, {srValue, 100e9}})

// ============================================================================
// Constructor
// ============================================================================
PythonFtmwScope::PythonFtmwScope(const QString &label, QObject *parent) :
    FtmwScope(QString(PythonFtmwScope::staticMetaObject.className()), label, parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;
}

// ============================================================================
// initialize()
// ============================================================================
void PythonFtmwScope::initialize()
{
    initPythonProcess(p_comm,
        [this](const QString &key, const QVariant &defaultVal) -> QVariant {
            return get(key, defaultVal);
        },
        [this](const QString &key, const QVariant &val) {
            set(key, val, true);
        }
    );

    pu_process->setEnabledProxies({"scope"_L1});

    connect(pu_process.get(), &PythonProcess::waveformReceived,
            this, &PythonFtmwScope::onWaveformReceived);
}

// ============================================================================
// testConnection()
// ============================================================================
bool PythonFtmwScope::testConnection()
{
    if (!testPythonConnection(p_comm)) {
        d_errorString = pythonErrorString();
        return false;
    }

    // Send current config to Python so it knows the initial state
    QJsonObject req = configToJson(static_cast<const FtmwDigitizerConfig&>(*this));
    req["method"_L1] = "configure"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1)) {
        d_errorString = "configure() failed after successful connection: "_L1
                        + resp["error"_L1].toString();
        return false;
    }

    auto resultObj = resp["result"_L1].toObject();
    if (!resultObj["success"_L1].toBool(false)) {
        d_errorString = "configure() returned failure after successful connection"_L1;
        return false;
    }

    return true;
}

// ============================================================================
// prepareForExperiment()
// ============================================================================
bool PythonFtmwScope::prepareForExperiment(Experiment &exp)
{
    if (!exp.ftmwEnabled())
        return true;

    auto desiredConfig = exp.ftmwConfig()->scopeConfig();

    QJsonObject req = configToJson(desiredConfig);
    req["method"_L1] = "configure"_L1;
    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1)) {
        hwError(u"configure failed: %1"_s.arg(resp["error"_L1].toString()));
        return false;
    }

    auto resultObj = resp["result"_L1].toObject();
    if (!resultObj["success"_L1].toBool(false)) {
        hwError("configure returned failure"_L1);
        return false;
    }

    // Apply validated config returned from Python
    if (resultObj.contains("config"_L1))
        jsonToConfig(resultObj["config"_L1].toObject(), desiredConfig);

    static_cast<FtmwDigitizerConfig&>(*this) = desiredConfig;

    return true;
}

// ============================================================================
// beginAcquisition()
// ============================================================================
void PythonFtmwScope::beginAcquisition()
{
    if (pu_process && pu_process->isRunning()) {
        QJsonObject req;
        req["method"_L1] = "begin_acquisition"_L1;
        pu_process->sendRequest(req);
    }
}

// ============================================================================
// endAcquisition()
// ============================================================================
void PythonFtmwScope::endAcquisition()
{
    if (pu_process && pu_process->isRunning()) {
        QJsonObject req;
        req["method"_L1] = "end_acquisition"_L1;
        pu_process->sendRequest(req);
    }
}

// ============================================================================
// onWaveformReceived()
// ============================================================================
void PythonFtmwScope::onWaveformReceived(const QByteArray &data, quint64 /*shotCount*/)
{
    emitShot(data);
}

// ============================================================================
// readSettings()
// ============================================================================
void PythonFtmwScope::readSettings()
{
    pythonReadSettings();
}

// ============================================================================
// sleep()
// ============================================================================
void PythonFtmwScope::sleep(bool b)
{
    pythonSleep(b);
}

// ============================================================================
// configToJson()
// ============================================================================
QJsonObject PythonFtmwScope::configToJson(const FtmwDigitizerConfig &config) const
{
    QJsonObject obj;

    // Analog channels
    QJsonObject anObj;
    for (auto const &[k, ch] : config.d_analogChannels) {
        QJsonObject chObj;
        chObj["enabled"_L1]    = ch.enabled;
        chObj["full_scale"_L1] = ch.fullScale;
        chObj["offset"_L1]     = ch.offset;
        anObj[QString::number(k)] = chObj;
    }
    obj["analog_channels"_L1] = anObj;

    // Digital channels
    QJsonObject digObj;
    for (auto const &[k, ch] : config.d_digitalChannels) {
        QJsonObject chObj;
        chObj["enabled"_L1] = ch.enabled;
        chObj["input"_L1]   = ch.input;
        chObj["role"_L1]    = ch.role;
        digObj[QString::number(k)] = chObj;
    }
    obj["digital_channels"_L1] = digObj;

    // Trigger settings
    QJsonObject trigObj;
    trigObj["channel"_L1]  = config.d_triggerChannel;
    trigObj["slope"_L1]    = static_cast<int>(config.d_triggerSlope);
    trigObj["delay_us"_L1] = config.d_triggerDelayUSec;
    trigObj["level"_L1]    = config.d_triggerLevel;
    obj["trigger"_L1] = trigObj;

    // Horizontal / data transfer
    obj["sample_rate"_L1]     = config.d_sampleRate;
    obj["record_length"_L1]   = config.d_recordLength;
    obj["bytes_per_point"_L1] = config.d_bytesPerPoint;
    obj["byte_order"_L1]      = static_cast<int>(config.d_byteOrder);

    // Averaging
    obj["block_average"_L1] = config.d_blockAverage;
    obj["num_averages"_L1]  = config.d_numAverages;

    // Multi-record
    obj["multi_record"_L1] = config.d_multiRecord;
    obj["num_records"_L1]  = config.d_numRecords;

    // FtmwDigitizerConfig extra field
    obj["fid_channel"_L1] = config.d_fidChannel;

    return obj;
}

// ============================================================================
// jsonToConfig()
// ============================================================================
bool PythonFtmwScope::jsonToConfig(const QJsonObject &obj, FtmwDigitizerConfig &config) const
{
    // Analog channels
    if (obj.contains("analog_channels"_L1)) {
        auto anObj = obj["analog_channels"_L1].toObject();
        config.d_analogChannels.clear();
        for (auto it = anObj.begin(); it != anObj.end(); ++it) {
            int idx = it.key().toInt();
            auto chObj = it.value().toObject();
            DigitizerConfig::AnalogChannel ch;
            ch.enabled   = chObj["enabled"_L1].toBool();
            ch.fullScale = chObj["full_scale"_L1].toDouble();
            ch.offset    = chObj["offset"_L1].toDouble();
            config.d_analogChannels[idx] = ch;
        }
    }

    // Digital channels
    if (obj.contains("digital_channels"_L1)) {
        auto digObj = obj["digital_channels"_L1].toObject();
        config.d_digitalChannels.clear();
        for (auto it = digObj.begin(); it != digObj.end(); ++it) {
            int idx = it.key().toInt();
            auto chObj = it.value().toObject();
            DigitizerConfig::DigitalChannel ch;
            ch.enabled = chObj["enabled"_L1].toBool();
            ch.input   = chObj["input"_L1].toBool(true);
            ch.role    = chObj["role"_L1].toInt(-1);
            config.d_digitalChannels[idx] = ch;
        }
    }

    // Trigger
    if (obj.contains("trigger"_L1)) {
        auto trigObj = obj["trigger"_L1].toObject();
        config.d_triggerChannel   = trigObj["channel"_L1].toInt();
        config.d_triggerSlope     = static_cast<DigitizerConfig::TriggerSlope>(
                                        trigObj["slope"_L1].toInt());
        config.d_triggerDelayUSec = trigObj["delay_us"_L1].toDouble();
        config.d_triggerLevel     = trigObj["level"_L1].toDouble();
    }

    // Horizontal / data transfer
    if (obj.contains("sample_rate"_L1))
        config.d_sampleRate = obj["sample_rate"_L1].toDouble();
    if (obj.contains("record_length"_L1))
        config.d_recordLength = obj["record_length"_L1].toInt();
    if (obj.contains("bytes_per_point"_L1))
        config.d_bytesPerPoint = obj["bytes_per_point"_L1].toInt();
    if (obj.contains("byte_order"_L1))
        config.d_byteOrder = static_cast<DigitizerConfig::ByteOrder>(
                                 obj["byte_order"_L1].toInt());

    // Averaging
    if (obj.contains("block_average"_L1))
        config.d_blockAverage = obj["block_average"_L1].toBool();
    if (obj.contains("num_averages"_L1))
        config.d_numAverages = obj["num_averages"_L1].toInt();

    // Multi-record
    if (obj.contains("multi_record"_L1))
        config.d_multiRecord = obj["multi_record"_L1].toBool();
    if (obj.contains("num_records"_L1))
        config.d_numRecords = obj["num_records"_L1].toInt();

    // FtmwDigitizerConfig extra field
    if (obj.contains("fid_channel"_L1))
        config.d_fidChannel = obj["fid_channel"_L1].toInt();

    return true;
}
