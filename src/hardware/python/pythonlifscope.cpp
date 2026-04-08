#include "pythonlifscope.h"

#include <QJsonObject>

#include <hardware/core/hardwareregistration.h>
#include <data/bcglobals.h>
#include <data/settings/hardwarekeys.h>

// ============================================================================
// Registration
// ============================================================================
REGISTER_HARDWARE_META(PythonLifScope, "Python LIF Digitizer (user-defined Python script)")
REGISTER_HARDWARE_PROTOCOLS(PythonLifScope, CommunicationProtocol::Rs232, CommunicationProtocol::Tcp, CommunicationProtocol::Virtual)
REGISTER_HARDWARE_PARAMS(PythonLifScope)

// ============================================================================
// Constructor
// ============================================================================
PythonLifScope::PythonLifScope(const QString &label, QObject *parent) :
    LifScope(QString(PythonLifScope::staticMetaObject.className()), label, parent),
    PythonHardwareBase(d_key, d_model)
{
    d_threaded = true;

    using namespace BC::Key::Digi;
    setDefault(numAnalogChannels, 2);
    setDefault(numDigitalChannels, 0);
    setDefault(hasAuxTriggerChannel, true);
    setDefault(minFullScale, 5e-2);
    setDefault(maxFullScale, 2.0);
    setDefault(minVOffset, -2.0);
    setDefault(maxVOffset, 2.0);
    setDefault(isTriggered, true);
    setDefault(minTrigDelay, -10.0);
    setDefault(maxTrigDelay, 10.0);
    setDefault(minTrigLevel, -5.0);
    setDefault(maxTrigLevel, 5.0);
    setDefault(maxRecordLength, 100000000);
    setDefault(canBlockAverage, true);
    setDefault(maxAverages, 100);
    setDefault(canMultiRecord, false);
    setDefault(maxBytes, 2);

    if(!containsArray(sampleRates))
        setArray(sampleRates,{
                     {{srText,"78.125 MSa/s"},{srValue,2.5e9/32}},
                     {{srText,"156.25 MSa/s"},{srValue,2.5e9/16}},
                     {{srText,"312.5 MSa/s"},{srValue,2.5e9/8}},
                     {{srText,"625 MSa/s"},{srValue,2.5e9/4}},
                     {{srText,"1250 GSa/s"},{srValue,2.5e9/2}},
                 });

    save();
}

// ============================================================================
// configParams()
// ============================================================================
QVector<HwConfigParam> PythonLifScope::configParams()
{
    using namespace BC::Key::Digi;
    return {
        { numAnalogChannels,  QStringLiteral("Analog Channels"),  QVariant(0), QVariant(0), QVariant(32) },
        { numDigitalChannels, QStringLiteral("Digital Channels"), QVariant(0), QVariant(0), QVariant(32) },
    };
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

    pu_process->setEnabledProxies({QStringLiteral("scope")});

    connect(pu_process.get(), &PythonProcess::logMessage,
            this, &PythonLifScope::logMessage);
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

    if (resultObj.contains(QStringLiteral("config")))
        jsonToConfig(resultObj[QStringLiteral("config")].toObject(),
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
    req[QStringLiteral("method")] = QStringLiteral("configure");
    auto resp = pu_process->sendRequest(req);

    if (resp.contains(QStringLiteral("error"))) {
        emit logMessage(QString("PythonLifScope (%1) configure failed: %2")
                        .arg(d_key, resp[QStringLiteral("error")].toString()),
                        LogHandler::Error);
        return false;
    }

    auto resultObj = resp[QStringLiteral("result")].toObject();
    if (!resultObj[QStringLiteral("success")].toBool(false)) {
        emit logMessage(QString("PythonLifScope (%1) configure returned failure").arg(d_key),
                        LogHandler::Error);
        return false;
    }

    // Apply validated config returned from Python into *this so the base
    // class can write it back to the experiment and settings storage.
    if (resultObj.contains(QStringLiteral("config")))
        jsonToConfig(resultObj[QStringLiteral("config")].toObject(),
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
        req[QStringLiteral("method")] = QStringLiteral("begin_acquisition");
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
        req[QStringLiteral("method")] = QStringLiteral("end_acquisition");
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
// forbiddenKeys()
// ============================================================================
QStringList PythonLifScope::forbiddenKeys() const
{
    auto keys = LifScope::forbiddenKeys();
    keys << pythonForbiddenKeys();
    return keys;
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

    // LifDigitizerConfig extra fields
    obj[QStringLiteral("lif_channel")]    = config.d_lifChannel;
    obj[QStringLiteral("ref_channel")]    = config.d_refChannel;
    obj[QStringLiteral("ref_enabled")]    = config.d_refEnabled;
    obj[QStringLiteral("channel_order")]  = static_cast<int>(config.d_channelOrder);

    return obj;
}

// ============================================================================
// jsonToConfig()
// ============================================================================
bool PythonLifScope::jsonToConfig(const QJsonObject &obj, LifDigitizerConfig &config) const
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

    // LifDigitizerConfig extra fields
    if (obj.contains(QStringLiteral("lif_channel")))
        config.d_lifChannel = obj[QStringLiteral("lif_channel")].toInt();
    if (obj.contains(QStringLiteral("ref_channel")))
        config.d_refChannel = obj[QStringLiteral("ref_channel")].toInt();
    if (obj.contains(QStringLiteral("ref_enabled")))
        config.d_refEnabled = obj[QStringLiteral("ref_enabled")].toBool();
    if (obj.contains(QStringLiteral("channel_order")))
        config.d_channelOrder = static_cast<LifDigitizerConfig::ChannelOrder>(
                                    obj[QStringLiteral("channel_order")].toInt());

    return true;
}
