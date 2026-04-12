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
    {BC::Key::Digi::hasAuxTriggerChannel, "Aux Trigger Channel", "Has auxiliary trigger input",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minFullScale, "Min Full Scale (V)", "Minimum full-scale voltage range",
     0.1, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxFullScale, "Max Full Scale (V)", "Maximum full-scale voltage range",
     10.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minVOffset, "Min V Offset (V)", "Minimum vertical offset",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxVOffset, "Max V Offset (V)", "Maximum vertical offset",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minTrigDelay, "Min Trig Delay (us)", "Minimum trigger delay in microseconds",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxTrigDelay, "Max Trig Delay (us)", "Maximum trigger delay in microseconds",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minTrigLevel, "Min Trig Level (V)", "Minimum trigger threshold voltage",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxTrigLevel, "Max Trig Level (V)", "Maximum trigger threshold voltage",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxRecordLength, "Max Record Length", "Maximum record length in samples",
     1, 1, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::canBlockAverage, "Block Average", "Supports block averaging",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::canMultiRecord, "Multi Record", "Supports multi-record acquisition",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::multiBlock, "Multi Block", "Can simultaneously block average and multi-record",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxBytes, "Max Bytes/Point", "Maximum bytes per sample",
     2, 1, 8, HwSettingPriority::Optional}
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

    connect(pu_process.get(), &PythonProcess::logMessage,
            this, &PythonIOBoard::logMessage);
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
        d_errorString = QStringLiteral("configure() failed after successful connection");
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
    req[QStringLiteral("method")] = QStringLiteral("configure");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error"))) {
        emit logMessage(QString("PythonIOBoard (%1) configure failed: %2")
                        .arg(d_key, resp[QStringLiteral("error")].toString()),
                        LogHandler::Error);
        return false;
    }

    auto resultObj = resp[QStringLiteral("result")].toObject();
    if (!resultObj[QStringLiteral("success")].toBool(false)) {
        emit logMessage(QString("PythonIOBoard (%1) configure returned failure").arg(d_key),
                        LogHandler::Error);
        return false;
    }

    // If Python returned a validated config, apply it
    if (resultObj.contains(QStringLiteral("config")))
        jsonToConfig(resultObj[QStringLiteral("config")].toObject(), config);

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
        chObj[QStringLiteral("enabled")] = ch.enabled;
        chObj[QStringLiteral("full_scale")] = ch.fullScale;
        chObj[QStringLiteral("offset")] = ch.offset;
        auto name = config.analogName(k);
        if (!name.isEmpty())
            chObj[QStringLiteral("name")] = name;
        anObj[QString::number(k)] = chObj;
    }
    obj[QStringLiteral("analog_channels")] = anObj;

    // Digital channels
    QJsonObject digObj;
    for (auto const &[k, ch] : config.d_digitalChannels) {
        QJsonObject chObj;
        chObj[QStringLiteral("enabled")] = ch.enabled;
        chObj[QStringLiteral("input")] = ch.input;
        chObj[QStringLiteral("role")] = ch.role;
        auto name = config.digitalName(k);
        if (!name.isEmpty())
            chObj[QStringLiteral("name")] = name;
        digObj[QString::number(k)] = chObj;
    }
    obj[QStringLiteral("digital_channels")] = digObj;

    // Trigger settings
    QJsonObject trigObj;
    trigObj[QStringLiteral("channel")] = config.d_triggerChannel;
    trigObj[QStringLiteral("slope")] = static_cast<int>(config.d_triggerSlope);
    trigObj[QStringLiteral("delay_us")] = config.d_triggerDelayUSec;
    trigObj[QStringLiteral("level")] = config.d_triggerLevel;
    obj[QStringLiteral("trigger")] = trigObj;

    // Horizontal / data transfer
    obj[QStringLiteral("sample_rate")] = config.d_sampleRate;
    obj[QStringLiteral("record_length")] = config.d_recordLength;
    obj[QStringLiteral("bytes_per_point")] = config.d_bytesPerPoint;
    obj[QStringLiteral("byte_order")] = static_cast<int>(config.d_byteOrder);

    // Averaging
    obj[QStringLiteral("block_average")] = config.d_blockAverage;
    obj[QStringLiteral("num_averages")] = config.d_numAverages;

    // Multi-record
    obj[QStringLiteral("multi_record")] = config.d_multiRecord;
    obj[QStringLiteral("num_records")] = config.d_numRecords;

    return obj;
}

// ============================================================================
// jsonToConfig()
// ============================================================================
bool PythonIOBoard::jsonToConfig(const QJsonObject &obj, IOBoardConfig &config) const
{
    // Analog channels
    if (obj.contains(QStringLiteral("analog_channels"))) {
        auto anObj = obj[QStringLiteral("analog_channels")].toObject();
        config.d_analogChannels.clear();
        for (auto it = anObj.begin(); it != anObj.end(); ++it) {
            int idx = it.key().toInt();
            auto chObj = it.value().toObject();
            DigitizerConfig::AnalogChannel ch;
            ch.enabled = chObj[QStringLiteral("enabled")].toBool();
            ch.fullScale = chObj[QStringLiteral("full_scale")].toDouble();
            ch.offset = chObj[QStringLiteral("offset")].toDouble();
            config.d_analogChannels[idx] = ch;
            if (chObj.contains(QStringLiteral("name")))
                config.setAnalogName(idx, chObj[QStringLiteral("name")].toString());
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
            ch.input = chObj[QStringLiteral("input")].toBool(true);
            ch.role = chObj[QStringLiteral("role")].toInt(-1);
            config.d_digitalChannels[idx] = ch;
            if (chObj.contains(QStringLiteral("name")))
                config.setDigitalName(idx, chObj[QStringLiteral("name")].toString());
        }
    }

    // Trigger
    if (obj.contains(QStringLiteral("trigger"))) {
        auto trigObj = obj[QStringLiteral("trigger")].toObject();
        config.d_triggerChannel = trigObj[QStringLiteral("channel")].toInt();
        config.d_triggerSlope = static_cast<DigitizerConfig::TriggerSlope>(
            trigObj[QStringLiteral("slope")].toInt());
        config.d_triggerDelayUSec = trigObj[QStringLiteral("delay_us")].toDouble();
        config.d_triggerLevel = trigObj[QStringLiteral("level")].toDouble();
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
    req[QStringLiteral("method")] = QStringLiteral("read_analog_channels");

    QJsonArray channels;
    for (auto const &[k, ch] : d_analogChannels) {
        if (ch.enabled)
            channels.append(k);
    }
    req[QStringLiteral("channels")] = channels;

    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return {};

    std::map<int, double> out;
    QJsonObject result = resp[QStringLiteral("result")].toObject();
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
    req[QStringLiteral("method")] = QStringLiteral("read_digital_channels");

    QJsonArray channels;
    for (auto const &[k, ch] : d_digitalChannels) {
        if (ch.enabled)
            channels.append(k);
    }
    req[QStringLiteral("channels")] = channels;

    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error")))
        return {};

    std::map<int, bool> out;
    QJsonObject result = resp[QStringLiteral("result")].toObject();
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

// ============================================================================
// forbiddenKeys()
// ============================================================================
QStringList PythonIOBoard::forbiddenKeys() const
{
    auto keys = pythonForbiddenKeys();
    keys << BC::Key::Digi::numAnalogChannels << BC::Key::Digi::numDigitalChannels;
    return keys;
}
