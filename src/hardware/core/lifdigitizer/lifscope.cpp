#include <hardware/core/lifdigitizer/lifscope.h>

#include <data/bcglobals.h>
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key::Digi;

REGISTER_HARDWARE_BASE(LifScope,
    {numAnalogChannels,    "Analog Channels",        "Number of analog input channels",           2,      1,         128,       HwSettingPriority::Required},
    {numDigitalChannels,   "Digital Channels",       "Number of digital input channels",          0,      0,         128,       HwSettingPriority::Required},
    {hasAuxTriggerChannel, "Aux Trigger Channel",    "Has auxiliary trigger input",               true,   QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {minFullScale,         "Min Full Scale (V)",     "Minimum full-scale voltage range",          0.05,   QVariant{}, QVariant{}, HwSettingPriority::Important},
    {maxFullScale,         "Max Full Scale (V)",     "Maximum full-scale voltage range",          2.0,    QVariant{}, QVariant{}, HwSettingPriority::Important},
    {minVOffset,           "Min V Offset (V)",       "Minimum vertical offset",                   -2.0,   QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxVOffset,           "Max V Offset (V)",       "Maximum vertical offset",                   2.0,    QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {isTriggered,          "Externally Triggered",   "Digitizer uses external trigger signal",    true,   QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {minTrigDelay,         "Min Trig Delay (us)",    "Minimum trigger delay in microseconds",     -10.0,  QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxTrigDelay,         "Max Trig Delay (us)",    "Maximum trigger delay in microseconds",     10.0,   QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {minTrigLevel,         "Min Trig Level (V)",     "Minimum trigger threshold voltage",         -5.0,   QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxTrigLevel,         "Max Trig Level (V)",     "Maximum trigger threshold voltage",         5.0,    QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {canBlockAverage,      "Block Average",          "Supports block averaging mode",             false,  QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxBytes,             "Max Bytes/Point",        "Maximum bytes per sample",                  2,      1,         4,         HwSettingPriority::Optional},
    {maxRecordLength,      "Max Record Length",      "Maximum record length in samples",          100000000, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxAverages,          "Max Averages",           "Maximum number of block averages",          10000,  QVariant{}, QVariant{}, HwSettingPriority::Optional}
)
REGISTER_HARDWARE_BASE_ARRAY(LifScope, sampleRates,
    "Sample Rates", "Available digitizer sample rates", HwSettingPriority::Important)

LifScope::LifScope(const QString& impl, const QString& label, QObject *parent) :
    HardwareObject(QString(LifScope::staticMetaObject.className()), impl, label, parent),
    LifDigitizerConfig(BC::Key::hwKey(QString(LifScope::staticMetaObject.className()), label))
{
    d_threaded = true;

    using namespace BC::Key::Digi;
    using namespace BC::Store::Digi;

    for(int i=0; i<get(numAnalogChannels,0); ++i)
    {
        auto idx = getArrayValue(dwAnChannels,i,chIndex,i+1);
        bool b = getArrayValue(dwAnChannels,i,en,false);
        auto fullScale = getArrayValue(dwAnChannels,i,fs,0.0);
        auto off = getArrayValue(dwAnChannels,i,offset,0.0);
        d_analogChannels.insert({idx,{b,fullScale,off}});
    }

    for(int i=0; i<get(numDigitalChannels,0); ++i)
    {
        auto idx = getArrayValue(dwDigChannels,i,chIndex,i+1);
        bool b = getArrayValue(dwDigChannels,i,en,false);
        bool in = getArrayValue(dwDigChannels,i,digInp,true);
        int role = getArrayValue(dwDigChannels,i,digRole,-1);
        d_digitalChannels.insert({idx,{b,in,role}});
    }

    d_triggerChannel = get(trigCh,0);
    d_triggerSlope = get(trigSlope,RisingEdge);
    d_triggerDelayUSec = get(trigDelay,0.0);
    d_triggerLevel = get(trigLevel,0.0);
    d_bytesPerPoint = get(bpp,1);
    d_byteOrder = get(bo,LittleEndian);
    d_sampleRate = get(sRate,0.0);
    d_recordLength = get(recLen,0);
    d_blockAverage = get(blockAvg,false);
    d_numAverages = get(numAvg,1);
    d_multiRecord = get(multiRec,false);
    d_numRecords = get(multiRecNum,1);
    d_lifChannel = get(lifChannel,1);
    d_refChannel = get(lifRefChannel,2);
    d_refEnabled = get(lifRefEnabled,false);
    d_channelOrder = get(lifChannelOrder,Interleaved);

}

LifScope::~LifScope()
{

}

bool LifScope::prepareForExperiment(Experiment &exp)
{
    d_enabledForExperiment = exp.lifEnabled();
    if(!d_enabledForExperiment)
        return true;

    if(configure(exp.lifConfig()->scopeConfig()))
    {
        exp.lifConfig()->scopeConfig() = static_cast<LifDigitizerConfig&>(*this);
        writeSettings();
        save();
        return true;
    }
    return false;
}

void LifScope::startConfigurationAcquisition(const LifConfig &c)
{
    if(configure(c.scopeConfig()))
    {
        writeSettings();
        save();
        emit configAcqComplete(QPrivateSignal());
        beginAcquisition();
    }
}

void LifScope::setAcquisitionGated(bool gated)
{
    d_acquisitionGated = gated;
    if(!gated)
        d_discardCount = 1;
}

void LifScope::emitWaveform(const QVector<qint8> &data)
{
    if(d_acquisitionGated)
        return;
    if(d_discardCount > 0)
    {
        --d_discardCount;
        return;
    }
    emit waveformRead(data);
}

void LifScope::writeSettings()
{
    using namespace BC::Key::Digi;
    using namespace BC::Store::Digi;

    int i=0;
    for(auto const &[k,ch] : d_analogChannels)
    {
        SettingsMap m{
            {en,ch.enabled},
            {chIndex,k},
            {fs,ch.fullScale},
            {offset,ch.offset}
        };

        if((std::size_t)i == getArraySize(dwAnChannels))
            appendArrayMap(dwAnChannels,m);
        else
        {
            for(auto &[kk,v] : m)
                setArrayValue(dwAnChannels,i,kk,v);
        }
        i++;
    }

    i=0;
    for(auto const &[k,ch] : d_digitalChannels)
    {
        SettingsMap m{
            {en,ch.enabled},
            {chIndex,k},
            {digInp,ch.input},
            {digRole,ch.role}
        };

        if((std::size_t) i == getArraySize(dwDigChannels))
            appendArrayMap(dwDigChannels,m);
        else
        {
            for(auto &[kk,v] : m)
                setArrayValue(dwDigChannels,i,kk,v);
        }
        i++;
    }

    set(trigCh,d_triggerChannel);
    set(trigSlope,static_cast<int>(d_triggerSlope));
    set(trigDelay,d_triggerDelayUSec);
    set(trigLevel,d_triggerLevel);
    set(bpp,d_bytesPerPoint);
    set(bo,static_cast<int>(d_byteOrder));
    set(sRate,d_sampleRate);
    set(recLen,d_recordLength);
    set(blockAvg,d_blockAverage);
    set(numAvg,d_numAverages);
    set(multiRec,d_multiRecord);
    set(multiRecNum,d_numRecords);
    set(lifChannel,d_lifChannel);
    set(lifRefChannel,d_refChannel);
    set(lifRefEnabled,d_refEnabled);
    set(lifChannelOrder,static_cast<int>(d_channelOrder));
    save();
}
