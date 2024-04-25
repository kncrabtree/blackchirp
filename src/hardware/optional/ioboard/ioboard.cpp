#include <hardware/optional/ioboard/ioboard.h>

IOBoard::IOBoard(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical)  :
    HardwareObject(BC::Key::IOB::ioboard, subKey, name, commType, parent, threaded, critical,d_count), IOBoardConfig(subKey,d_count)
{
    using namespace BC::Key::Digi;
    using namespace BC::Store::Digi;
    setDefault(isTriggered,false);

    d_count++;

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

}

IOBoard::~IOBoard()
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
}

QStringList IOBoard::validationKeys() const
{
    using namespace BC::Key::Digi;
    int an = get(numAnalogChannels,0);
    int dn = get(numDigitalChannels,0);

    QStringList out;
    out.reserve(an+dn);
    for(int i=0; i<an; ++i)
        out.append(BC::Aux::IOB::ain.arg(i+1));
    for(int i=0; i<dn; ++i)
        out.append(BC::Aux::IOB::din.arg(i+1));

    return out;
}

AuxDataStorage::AuxDataMap IOBoard::readAuxData()
{
    AuxDataStorage::AuxDataMap out;
    auto m = readAnalogChannels();
    for(auto it = m.cbegin(); it != m.cend(); ++it)
    {
        auto name = analogName(it->first);
        if(!name.isEmpty())
            out.insert({name+"."+BC::Aux::IOB::ain.arg(it->first),it->second});
        else
            out.insert({BC::Aux::IOB::ain.arg(it->first),it->second});
    }

    return out;
}

AuxDataStorage::AuxDataMap IOBoard::readValidationData()
{
    AuxDataStorage::AuxDataMap out;
    auto m = readDigitalChannels();
    for(auto it = m.cbegin(); it != m.cend(); ++it)
    {
        auto name = digitalName(it->first);
        if(!name.isEmpty())
            out.insert({name+"."+BC::Aux::IOB::din.arg(it->first),it->second});
        else
            out.insert({BC::Aux::IOB::din.arg(it->first),it->second});
    }

    return out;
}

bool IOBoard::prepareForExperiment(Experiment &exp)
{
    auto wp = exp.getOptHwConfig<IOBoardConfig>(d_headerKey);
    auto cfg = static_cast<IOBoardConfig*>(this);
    if(auto p = wp.lock())
        *cfg = *p;
    else
        exp.addOptHwConfig(*cfg);

    for(auto it = d_analogChannels.cbegin();it!=d_analogChannels.cend();++it)
    {
        auto name = analogName(it->first);
        if(!name.isEmpty())
            exp.auxData()->registerKey(d_key,d_subKey,name+"."+BC::Aux::IOB::ain.arg(it->first));
        else
            exp.auxData()->registerKey(d_key,d_subKey,BC::Aux::IOB::ain.arg(it->first));
    }

    //note: digital channels should not be registered because they do not need to be plotted and
    //saved to disk. their only purpose is to potentially abort the experiment if they are set
    //as a validation condition.

    return true;
}


QStringList IOBoard::forbiddenKeys() const
{
    return {BC::Key::Digi::numAnalogChannels, BC::Key::Digi::numDigitalChannels};
}
