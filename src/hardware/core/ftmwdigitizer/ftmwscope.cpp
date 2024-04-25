#include <hardware/core/ftmwdigitizer/ftmwscope.h>

FtmwScope::FtmwScope(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical) :
    HardwareObject(BC::Key::FtmwScope::ftmwScope,subKey,name,commType,parent,threaded,critical),
    FtmwDigitizerConfig(subKey)
{
    using namespace BC::Key::Digi;
    using namespace BC::Store::Digi;

    for(int i=0; i<get(numAnalogChannels,0); ++i)
    {
        bool b = getArrayValue(dwAnChannels,i,en,false);
        if(b)
        {
            auto fullScale = getArrayValue(dwAnChannels,i,fs,0.0);
            auto off = getArrayValue(dwAnChannels,i,offset,0.0);
            d_analogChannels.insert({i,{fullScale,off}});
        }
    }

    for(int i=0; i<get(numDigitalChannels,0); ++i)
    {
        bool in = getArrayValue(dwDigChannels,i,digInp,true);
        int role = getArrayValue(dwDigChannels,i,digRole,-1);
        d_digitalChannels.insert({i,{in,role}});
    }

    d_triggerChannel = get(trigCh,0);
    d_triggerSlope = get(trigSlope,RisingEdge);
    d_triggerDelayUSec = get(trigDelay,0.0);
    d_triggerLevel = get(trigLevel,0.0);
    d_bytesPerPoint = get(bpp,1);
    d_byteOrder = get(bpp,LittleEndian);
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
    using namespace BC::Key::Digi;
    using namespace BC::Store::Digi;

    for(int i=0; i < get(numAnalogChannels,0); ++i)
    {
        SettingsMap m;
        auto it = d_analogChannels.find(i);
        if(it != d_analogChannels.end())
        {
            m = {
                {en,true},
                {fs,it->second.fullScale},
                {offset,it->second.offset}
            };
        }

        if((std::size_t)i == getArraySize(dwAnChannels))
            appendArrayMap(dwAnChannels,m);
        else
        {
            for(auto &[k,v] : m)
                setArrayValue(dwAnChannels,i,k,v);
        }
    }

    for(int i=0; i<get(numDigitalChannels,0); ++i)
    {
        SettingsMap m;
        auto it = d_digitalChannels.find(i);
        if(it != d_digitalChannels.end())
        {
            m = {
                {digInp,it->second.input},
                {digRole,it->second.role}
            };
        }

        if((std::size_t) i == getArraySize(dwDigChannels))
            appendArrayMap(dwDigChannels,m);
        else
        {
            for(auto &[k,v] : m)
                setArrayValue(dwDigChannels,i,k,v);
        }
    }

    set(trigCh,d_triggerChannel);
    set(trigSlope,d_triggerSlope);
    set(trigDelay,d_triggerDelayUSec);
    set(trigLevel,d_triggerLevel);
    set(bpp,d_bytesPerPoint);
    set(bo,d_byteOrder);
    set(sRate,d_sampleRate);
    set(recLen,d_recordLength);
    set(blockAvg,d_blockAverage);
    set(numAvg,d_numAverages);
    set(multiRec,d_multiRecord);
    set(multiRecNum,d_numRecords);
    set(fidCh,d_fidChannel);
}





QStringList FtmwScope::forbiddenKeys() const
{
    return {BC::Key::Digi::numAnalogChannels, BC::Key::Digi::numDigitalChannels};
}
