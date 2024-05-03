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
        auto idx = getArrayValue(dwAnChannels,i,chIndex,i+1);
        bool b = getArrayValue(dwAnChannels,i,en,false);
        auto fullScale = getArrayValue(dwAnChannels,i,fs,0.0);
        auto off = getArrayValue(dwAnChannels,i,offset,0.0);
        auto n = getArrayValue(dwAnChannels,i,chName,QString(""));
        d_analogChannels.insert({idx,{b,fullScale,off}});
        setAnalogName(idx,n);
    }

    for(int i=0; i<get(numDigitalChannels,0); ++i)
    {
        auto idx = getArrayValue(dwDigChannels,i,chIndex,i+1);
        bool b = getArrayValue(dwDigChannels,i,en,false);
        bool in = getArrayValue(dwDigChannels,i,digInp,true);
        int role = getArrayValue(dwDigChannels,i,digRole,-1);
        auto n = getArrayValue(dwDigChannels,i,chName,QString(""));
        d_digitalChannels.insert({idx,{b,in,role}});
        setDigitalName(idx,n);
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

}

IOBoard::~IOBoard()
{
    writeSettings();
}

QStringList IOBoard::validationKeys() const
{
    using namespace BC::Key::Digi;
    int an = get(numAnalogChannels,0);
    int dn = get(numDigitalChannels,0);

    QStringList out;
    out.reserve(an+dn);
    for(int i=0; i<an; ++i)
    {
        auto idx = getArrayValue(dwAnChannels,i,BC::Store::Digi::chIndex,i+1);
        out.append(BC::Aux::IOB::ain.arg(idx));
    }
    for(int i=0; i<dn; ++i)
    {
        auto idx = getArrayValue(dwDigChannels,i,BC::Store::Digi::chIndex,i+1);
        out.append(BC::Aux::IOB::din.arg(idx));
    }

    return out;
}

AuxDataStorage::AuxDataMap IOBoard::readAuxData()
{
    AuxDataStorage::AuxDataMap out;
    auto m = readAnalogChannels();
    for(auto const &[k,v] : m)
    {
        auto name = analogName(k);
        if(!name.isEmpty())
            out.insert({name+"."+BC::Aux::IOB::ain.arg(k),v});
        else
            out.insert({BC::Aux::IOB::ain.arg(k),v});
    }

    return out;
}

AuxDataStorage::AuxDataMap IOBoard::readValidationData()
{
    AuxDataStorage::AuxDataMap out;
    auto m = readDigitalChannels();
    for(auto const &[k,v] : m)
    {
        auto name = digitalName(k);
        if(!name.isEmpty())
            out.insert({name+"."+BC::Aux::IOB::din.arg(k),v});
        else
            out.insert({BC::Aux::IOB::din.arg(k),v});
    }

    return out;
}

void IOBoard::writeSettings()
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
            {offset,ch.offset},
            {chName,analogName(k)}
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
            {digRole,ch.role},
            {chName,digitalName(k)}
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

    save();
}

bool IOBoard::hwPrepareForExperiment(Experiment &exp)
{
    auto out = HardwareObject::hwPrepareForExperiment(exp);
    if(out)
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
        writeSettings();
    }

    return out;
}


QStringList IOBoard::forbiddenKeys() const
{
    return {BC::Key::Digi::numAnalogChannels, BC::Key::Digi::numDigitalChannels};
}
