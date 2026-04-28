#include "pythonioboard.h"

#include <QJsonArray>
#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonIOBoard, "Python IO Board (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonIOBoard, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_SETTINGS(PythonIOBoard,
    {BC::Key::Digi::numAnalogChannels, "Analog Channels", "Number of analog input channels",
     0, 0, 128, HwSettingPriority::Required},
    {BC::Key::Digi::numDigitalChannels, "Digital Channels", "Number of digital input channels",
     0, 0, 128, HwSettingPriority::Required},
    {BC::Key::Digi::minFullScale, "Min Full Scale (V)", "Minimum full-scale voltage range",
     0.1, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxFullScale, "Max Full Scale (V)", "Maximum full-scale voltage range",
     10.0, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

// ============================================================================
// Constructor / Destructor
// ============================================================================
PythonIOBoard::PythonIOBoard(const QString &label, QObject *parent) :
    IOBoard(QString(PythonIOBoard::staticMetaObject.className()), label, parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;

    save();
}

// ============================================================================
// initialize()
// ============================================================================
void PythonIOBoard::initialize()
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
bool PythonIOBoard::testConnection()
{
    if (!testPythonConnection(p_comm)) {
        d_errorString = pythonErrorString();
        return false;
    }

    auto config = getConfig();
    if (!configure(config)) {
        d_errorString = "configure() failed after successful connection"_L1;
        return false;
    }
    static_cast<IOBoardConfig&>(*this) = config;

    return true;
}

// ============================================================================
// configure()
// ============================================================================
bool PythonIOBoard::configure(IOBoardConfig &config)
{
    if (!pu_process || !pu_process->isRunning())
        return false;

    QJsonObject req = configToJson(config);
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

    // If Python returned a validated config, apply it
    if (resultObj.contains("config"_L1))
        jsonToConfig(resultObj["config"_L1].toObject(), config);

    return true;
}

// ============================================================================
// configToJson()
// ============================================================================
QJsonObject PythonIOBoard::configToJson(const IOBoardConfig &config) const
{
    QJsonObject obj;

    // Analog channels
    QJsonObject anObj;
    for (auto const &[k, ch] : config.d_analogChannels) {
        QJsonObject chObj;
        chObj["enabled"_L1] = ch.enabled;
        chObj["full_scale"_L1] = ch.fullScale;
        chObj["offset"_L1] = ch.offset;
        auto name = config.analogName(k);
        if (!name.isEmpty())
            chObj["name"_L1] = name;
        anObj[QString::number(k)] = chObj;
    }
    obj["analog_channels"_L1] = anObj;

    // Digital channels
    QJsonObject digObj;
    for (auto const &[k, ch] : config.d_digitalChannels) {
        QJsonObject chObj;
        chObj["enabled"_L1] = ch.enabled;
        chObj["input"_L1] = ch.input;
        chObj["role"_L1] = ch.role;
        auto name = config.digitalName(k);
        if (!name.isEmpty())
            chObj["name"_L1] = name;
        digObj[QString::number(k)] = chObj;
    }
    obj["digital_channels"_L1] = digObj;

    // Trigger settings
    QJsonObject trigObj;
    trigObj["channel"_L1] = config.d_triggerChannel;
    trigObj["slope"_L1] = static_cast<int>(config.d_triggerSlope);
    trigObj["delay_us"_L1] = config.d_triggerDelayUSec;
    trigObj["level"_L1] = config.d_triggerLevel;
    obj["trigger"_L1] = trigObj;

    // Horizontal / data transfer
    obj["sample_rate"_L1] = config.d_sampleRate;
    obj["record_length"_L1] = config.d_recordLength;
    obj["bytes_per_point"_L1] = config.d_bytesPerPoint;
    obj["byte_order"_L1] = static_cast<int>(config.d_byteOrder);

    // Averaging
    obj["block_average"_L1] = config.d_blockAverage;
    obj["num_averages"_L1] = config.d_numAverages;

    // Multi-record
    obj["multi_record"_L1] = config.d_multiRecord;
    obj["num_records"_L1] = config.d_numRecords;

    return obj;
}

// ============================================================================
// jsonToConfig()
// ============================================================================
bool PythonIOBoard::jsonToConfig(const QJsonObject &obj, IOBoardConfig &config) const
{
    // Analog channels
    if (obj.contains("analog_channels"_L1)) {
        auto anObj = obj["analog_channels"_L1].toObject();
        config.d_analogChannels.clear();
        for (auto it = anObj.begin(); it != anObj.end(); ++it) {
            int idx = it.key().toInt();
            auto chObj = it.value().toObject();
            DigitizerConfig::AnalogChannel ch;
            ch.enabled = chObj["enabled"_L1].toBool();
            ch.fullScale = chObj["full_scale"_L1].toDouble();
            ch.offset = chObj["offset"_L1].toDouble();
            config.d_analogChannels[idx] = ch;
            if (chObj.contains("name"_L1))
                config.setAnalogName(idx, chObj["name"_L1].toString());
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
            ch.input = chObj["input"_L1].toBool(true);
            ch.role = chObj["role"_L1].toInt(-1);
            config.d_digitalChannels[idx] = ch;
            if (chObj.contains("name"_L1))
                config.setDigitalName(idx, chObj["name"_L1].toString());
        }
    }

    // Trigger
    if (obj.contains("trigger"_L1)) {
        auto trigObj = obj["trigger"_L1].toObject();
        config.d_triggerChannel = trigObj["channel"_L1].toInt();
        config.d_triggerSlope = static_cast<DigitizerConfig::TriggerSlope>(
            trigObj["slope"_L1].toInt());
        config.d_triggerDelayUSec = trigObj["delay_us"_L1].toDouble();
        config.d_triggerLevel = trigObj["level"_L1].toDouble();
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

    return true;
}

// ============================================================================
// readAnalogChannels()
// ============================================================================
std::map<int, double> PythonIOBoard::readAnalogChannels()
{
    if (!pu_process || !pu_process->isRunning())
        return {};

    QJsonObject req;
    req["method"_L1] = "read_analog_channels"_L1;

    QJsonArray channels;
    for (auto const &[k, ch] : d_analogChannels) {
        if (ch.enabled)
            channels.append(k);
    }
    req["channels"_L1] = channels;

    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return {};

    std::map<int, double> out;
    QJsonObject result = resp["result"_L1].toObject();
    for (auto it = result.begin(); it != result.end(); ++it)
        out[it.key().toInt()] = it.value().toDouble();

    return out;
}

// ============================================================================
// readDigitalChannels()
// ============================================================================
std::map<int, bool> PythonIOBoard::readDigitalChannels()
{
    if (!pu_process || !pu_process->isRunning())
        return {};

    QJsonObject req;
    req["method"_L1] = "read_digital_channels"_L1;

    QJsonArray channels;
    for (auto const &[k, ch] : d_digitalChannels) {
        if (ch.enabled)
            channels.append(k);
    }
    req["channels"_L1] = channels;

    auto resp = pu_process->sendRequest(req);

    if (resp.contains("error"_L1))
        return {};

    std::map<int, bool> out;
    QJsonObject result = resp["result"_L1].toObject();
    for (auto it = result.begin(); it != result.end(); ++it)
        out[it.key().toInt()] = it.value().toBool();

    return out;
}

// ============================================================================
// sleep()
// ============================================================================
void PythonIOBoard::sleep(bool b)
{
    pythonSleep(b);
}

// ============================================================================
// readSettings()
// ============================================================================
void PythonIOBoard::readSettings()
{
    pythonReadSettings();

    auto config = getConfig();
    if (configure(config))
        static_cast<IOBoardConfig&>(*this) = config;
}

