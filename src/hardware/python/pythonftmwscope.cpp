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

    pu_process->setEnabledProxies({QStringLiteral("scope")});

    connect(pu_process.get(), &PythonProcess::logMessage,
            this, &PythonFtmwScope::logMessage);
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
    req[QStringLiteral("method")] = QStringLiteral("configure");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error"))) {
        d_errorString = QStringLiteral("configure() failed after successful connection: ")
                        + resp[QStringLiteral("error")].toString();
        return false;
    }

    auto resultObj = resp[QStringLiteral("result")].toObject();
    if (!resultObj[QStringLiteral("success")].toBool(false)) {
        d_errorString = QStringLiteral("configure() returned failure after successful connection");
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
    req[QStringLiteral("method")] = QStringLiteral("configure");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error"))) {
        hwError(u"configure failed: %1"_s.arg(resp[QStringLiteral("error")].toString()));
        return false;
    }

    auto resultObj = resp[QStringLiteral("result")].toObject();
    if (!resultObj[QStringLiteral("success")].toBool(false)) {
        hwError("configure returned failure"_L1);
        return false;
    }

    // Apply validated config returned from Python
    if (resultObj.contains(QStringLiteral("config")))
        jsonToConfig(resultObj[QStringLiteral("config")].toObject(), desiredConfig);

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
        req[QStringLiteral("method")] = QStringLiteral("begin_acquisition");
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
        req[QStringLiteral("method")] = QStringLiteral("end_acquisition");
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
        chObj[QStringLiteral("enabled")]    = ch.enabled;
        chObj[QStringLiteral("full_scale")] = ch.fullScale;
        chObj[QStringLiteral("offset")]     = ch.offset;
        anObj[QString::number(k)] = chObj;
    }
    obj[QStringLiteral("analog_channels")] = anObj;

    // Digital channels
    QJsonObject digObj;
    for (auto const &[k, ch] : config.d_digitalChannels) {
        QJsonObject chObj;
        chObj[QStringLiteral("enabled")] = ch.enabled;
        chObj[QStringLiteral("input")]   = ch.input;
        chObj[QStringLiteral("role")]    = ch.role;
        digObj[QString::number(k)] = chObj;
    }
    obj[QStringLiteral("digital_channels")] = digObj;

    // Trigger settings
    QJsonObject trigObj;
    trigObj[QStringLiteral("channel")]  = config.d_triggerChannel;
    trigObj[QStringLiteral("slope")]    = static_cast<int>(config.d_triggerSlope);
    trigObj[QStringLiteral("delay_us")] = config.d_triggerDelayUSec;
    trigObj[QStringLiteral("level")]    = config.d_triggerLevel;
    obj[QStringLiteral("trigger")] = trigObj;

    // Horizontal / data transfer
    obj[QStringLiteral("sample_rate")]     = config.d_sampleRate;
    obj[QStringLiteral("record_length")]   = config.d_recordLength;
    obj[QStringLiteral("bytes_per_point")] = config.d_bytesPerPoint;
    obj[QStringLiteral("byte_order")]      = static_cast<int>(config.d_byteOrder);

    // Averaging
    obj[QStringLiteral("block_average")] = config.d_blockAverage;
    obj[QStringLiteral("num_averages")]  = config.d_numAverages;

    // Multi-record
    obj[QStringLiteral("multi_record")] = config.d_multiRecord;
    obj[QStringLiteral("num_records")]  = config.d_numRecords;

    // FtmwDigitizerConfig extra field
    obj[QStringLiteral("fid_channel")] = config.d_fidChannel;

    return obj;
}

// ============================================================================
// jsonToConfig()
// ============================================================================
bool PythonFtmwScope::jsonToConfig(const QJsonObject &obj, FtmwDigitizerConfig &config) const
{
    // Analog channels
    if (obj.contains(QStringLiteral("analog_channels"))) {
        auto anObj = obj[QStringLiteral("analog_channels")].toObject();
        config.d_analogChannels.clear();
        for (auto it = anObj.begin(); it != anObj.end(); ++it) {
            int idx = it.key().toInt();
            auto chObj = it.value().toObject();
            DigitizerConfig::AnalogChannel ch;
            ch.enabled   = chObj[QStringLiteral("enabled")].toBool();
            ch.fullScale = chObj[QStringLiteral("full_scale")].toDouble();
            ch.offset    = chObj[QStringLiteral("offset")].toDouble();
            config.d_analogChannels[idx] = ch;
        }
    }

    // Digital channels
    if (obj.contains(QStringLiteral("digital_channels"))) {
        auto digObj = obj[QStringLiteral("digital_channels")].toObject();
        config.d_digitalChannels.clear();
        for (auto it = digObj.begin(); it != digObj.end(); ++it) {
            int idx = it.key().toInt();
            auto chObj = it.value().toObject();
            DigitizerConfig::DigitalChannel ch;
            ch.enabled = chObj[QStringLiteral("enabled")].toBool();
            ch.input   = chObj[QStringLiteral("input")].toBool(true);
            ch.role    = chObj[QStringLiteral("role")].toInt(-1);
            config.d_digitalChannels[idx] = ch;
        }
    }

    // Trigger
    if (obj.contains(QStringLiteral("trigger"))) {
        auto trigObj = obj[QStringLiteral("trigger")].toObject();
        config.d_triggerChannel   = trigObj[QStringLiteral("channel")].toInt();
        config.d_triggerSlope     = static_cast<DigitizerConfig::TriggerSlope>(
                                        trigObj[QStringLiteral("slope")].toInt());
        config.d_triggerDelayUSec = trigObj[QStringLiteral("delay_us")].toDouble();
        config.d_triggerLevel     = trigObj[QStringLiteral("level")].toDouble();
    }

    // Horizontal / data transfer
    if (obj.contains(QStringLiteral("sample_rate")))
        config.d_sampleRate = obj[QStringLiteral("sample_rate")].toDouble();
    if (obj.contains(QStringLiteral("record_length")))
        config.d_recordLength = obj[QStringLiteral("record_length")].toInt();
    if (obj.contains(QStringLiteral("bytes_per_point")))
        config.d_bytesPerPoint = obj[QStringLiteral("bytes_per_point")].toInt();
    if (obj.contains(QStringLiteral("byte_order")))
        config.d_byteOrder = static_cast<DigitizerConfig::ByteOrder>(
                                 obj[QStringLiteral("byte_order")].toInt());

    // Averaging
    if (obj.contains(QStringLiteral("block_average")))
        config.d_blockAverage = obj[QStringLiteral("block_average")].toBool();
    if (obj.contains(QStringLiteral("num_averages")))
        config.d_numAverages = obj[QStringLiteral("num_averages")].toInt();

    // Multi-record
    if (obj.contains(QStringLiteral("multi_record")))
        config.d_multiRecord = obj[QStringLiteral("multi_record")].toBool();
    if (obj.contains(QStringLiteral("num_records")))
        config.d_numRecords = obj[QStringLiteral("num_records")].toInt();

    // FtmwDigitizerConfig extra field
    if (obj.contains(QStringLiteral("fid_channel")))
        config.d_fidChannel = obj[QStringLiteral("fid_channel")].toInt();

    return true;
}
