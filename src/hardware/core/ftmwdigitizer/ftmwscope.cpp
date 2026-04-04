#include <hardware/core/ftmwdigitizer/ftmwscope.h>

FtmwScope::FtmwScope(const QString& impl, const QString& label, QObject *parent) :
    HardwareObject(QString(FtmwScope::staticMetaObject.className()), impl, label, parent),
    FtmwDigitizerConfig(BC::Key::hwKey(QString(FtmwScope::staticMetaObject.className()), label))
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
    d_fidChannel = get(fidCh,0);
}

FtmwScope::~FtmwScope()
{

}

bool FtmwScope::hwPrepareForExperiment(Experiment &exp)
{
    auto out = HardwareObject::hwPrepareForExperiment(exp);
    if(out)
    {
        writeSettings();
        qint64 waveformBytes = static_cast<qint64>(d_recordLength) * d_bytesPerPoint * d_numRecords;
        pu_waveformBuffer = std::make_unique<WaveformBuffer>(10, waveformBytes);
        if(exp.ftmwEnabled())
            exp.ftmwConfig()->setWaveformBuffer(pu_waveformBuffer.get());
    }

    return out;
}





QStringList FtmwScope::forbiddenKeys() const
{
    return {BC::Key::Digi::numAnalogChannels, BC::Key::Digi::numDigitalChannels};
}

void FtmwScope::setAcquisitionGated(bool gated)
{
    d_acquisitionGated = gated;
    if(!gated)
        d_discardCount = 1;
}

void FtmwScope::emitShot(const QByteArray &data)
{
    if(d_acquisitionGated)
        return;
    if(d_discardCount > 0)
    {
        --d_discardCount;
        return;
    }
    if(pu_waveformBuffer)
    {
        quint64 shots = d_blockAverage ? d_numAverages : 1;
        pu_waveformBuffer->write(data, shots);
    }
}

void FtmwScope::writeSettings()
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
    set(fidCh,d_fidChannel);
    save();
}
