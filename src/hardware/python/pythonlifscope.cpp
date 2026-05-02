#include "pythonlifscope.h"

#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>
#include <data/bcglobals.h>
#include <data/settings/hardwarekeys.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonLifScope, "Python LIF Digitizer (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonLifScope, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Gpib, CommunicationProtocol::Custom, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_SETTINGS(PythonLifScope,
    {BC::Key::Digi::canBlockAverage, "Block Average", "Supports block averaging mode", true,  QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxAverages,     "Max Averages",  "Maximum number of block averages", 100, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)
REGISTER_HARDWARE_ARRAY(PythonLifScope, BC::Key::Digi::sampleRates,
    "Sample Rates", "Available digitizer sample rates", HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(PythonLifScope, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "78.125 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/32}})
REGISTER_HARDWARE_ARRAY_ENTRY(PythonLifScope, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "156.25 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/16}})
REGISTER_HARDWARE_ARRAY_ENTRY(PythonLifScope, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "312.5 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/8}})
REGISTER_HARDWARE_ARRAY_ENTRY(PythonLifScope, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "625 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/4}})
REGISTER_HARDWARE_ARRAY_ENTRY(PythonLifScope, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "1250 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/2}})

// ============================================================================
// Constructor
// ============================================================================
PythonLifScope::PythonLifScope(const QString &label, QObject *parent) :
    LifScope(QString(PythonLifScope::staticMetaObject.className()), label, parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;

    using namespace BC::Key::Digi;
    setDefault(canMultiRecord, false);

    if(!containsArray(sampleRates))
        setArray(sampleRates,{
                     {{srText,"78.125 MSa/s"},{srValue,2.5e9/32}},
                     {{srText,"156.25 MSa/s"},{srValue,2.5e9/16}},
                     {{srText,"312.5 MSa/s"},{srValue,2.5e9/8}},
                     {{srText,"625 MSa/s"},{srValue,2.5e9/4}},
                     {{srText,"1250 MSa/s"},{srValue,2.5e9/2}},
                 });

    save();
}

// ============================================================================
// initialize()
// ============================================================================
void PythonLifScope::initialize()
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
            this, &PythonLifScope::onWaveformReceived);
}

// ============================================================================
// testConnection()
// ============================================================================
bool PythonLifScope::testConnection()
{
    if (!testPythonConnection(p_comm)) {
        d_errorString = pythonErrorString();
        return false;
    }

    // Send current config to Python so it knows the initial state
    QJsonObject req = configToJson(static_cast<const LifDigitizerConfig&>(*this));
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

    if (resultObj.contains("config"_L1))
        jsonToConfig(resultObj["config"_L1].toObject(),
                     static_cast<LifDigitizerConfig&>(*this));

    return true;
}

// ============================================================================
// configure()
// ============================================================================
bool PythonLifScope::configure(const LifDigitizerConfig &c)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req = configToJson(c);
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

    // Apply validated config returned from Python into *this so the base
    // class can write it back to the experiment and settings storage.
    if (resultObj.contains("config"_L1))
        jsonToConfig(resultObj["config"_L1].toObject(),
                     static_cast<LifDigitizerConfig&>(*this));
    else
        static_cast<LifDigitizerConfig&>(*this) = c;

    return true;
}

// ============================================================================
// beginAcquisition()
// ============================================================================
void PythonLifScope::beginAcquisition()
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
void PythonLifScope::endAcquisition()
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
void PythonLifScope::onWaveformReceived(const QByteArray &data, quint64 /*shotCount*/)
{
    QVector<qint8> vec(data.size());
    memcpy(vec.data(), data.constData(), static_cast<std::size_t>(data.size()));
    emitWaveform(vec);
}

// ============================================================================
// readSettings()
// ============================================================================
void PythonLifScope::readSettings()
{
    pythonReadSettings();
}

// ============================================================================
// sleep()
// ============================================================================
void PythonLifScope::sleep(bool b)
{
    pythonSleep(b);
}

// ============================================================================
// configToJson()
// ============================================================================
QJsonObject PythonLifScope::configToJson(const LifDigitizerConfig &config) const
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

    // LifDigitizerConfig extra fields
    obj["lif_channel"_L1]    = config.d_lifChannel;
    obj["ref_channel"_L1]    = config.d_refChannel;
    obj["ref_enabled"_L1]    = config.d_refEnabled;
    obj["channel_order"_L1]  = static_cast<int>(config.d_channelOrder);

    return obj;
}

// ============================================================================
// jsonToConfig()
// ============================================================================
bool PythonLifScope::jsonToConfig(const QJsonObject &obj, LifDigitizerConfig &config) const
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

    // LifDigitizerConfig extra fields
    if (obj.contains("lif_channel"_L1))
        config.d_lifChannel = obj["lif_channel"_L1].toInt();
    if (obj.contains("ref_channel"_L1))
        config.d_refChannel = obj["ref_channel"_L1].toInt();
    if (obj.contains("ref_enabled"_L1))
        config.d_refEnabled = obj["ref_enabled"_L1].toBool();
    if (obj.contains("channel_order"_L1))
        config.d_channelOrder = static_cast<LifDigitizerConfig::ChannelOrder>(
                                    obj["channel_order"_L1].toInt());

    return true;
}
