#include "digitizerconfig.h"

#include <cmath>

DigitizerConfig::DigitizerConfig(const QString key, const QString subKey) : HeaderStorage(key,subKey)
{

}

double DigitizerConfig::xIncr() const
{
    if(d_sampleRate > 0.0)
        return 1.0/d_sampleRate;

    return nan("");
}

double DigitizerConfig::yMult(int ch) const
{
    auto it = d_analogChannels.find(ch);
    if(it == d_analogChannels.end())
        return nan("");

    return it->second.fullScale / static_cast<double>( 1 << (d_bytesPerPoint*8 - 1));
}


void DigitizerConfig::storeValues()
{
    using namespace BC::Store::Digi;
    int i=0;
    for(auto const &[k,ch] : d_analogChannels)
    {
        storeArrayValue(an,i,chIndex,k);
        storeArrayValue(an,i,en,ch.enabled);
        storeArrayValue(an,i,fs,ch.fullScale,BC::Unit::V);
        storeArrayValue(an,i,offset,ch.offset,BC::Unit::V);
        i++;
    }
    i=0;
    for(auto const &[k,ch] : d_digitalChannels)
    {
        storeArrayValue(dig,i,chIndex,k);
        storeArrayValue(dig,i,en,ch.enabled);
        storeArrayValue(dig,i,digInp,ch.input);
        storeArrayValue(dig,i,digRole,ch.role);
        i++;
    }

    store(trigCh,d_triggerChannel);
    store(trigSlope,d_triggerSlope);
    store(trigLevel,d_triggerLevel,BC::Unit::V);
    store(trigDelay,d_triggerDelayUSec,BC::Unit::us);
    store(bpp,d_bytesPerPoint);
    store(bo,d_byteOrder);
    store(sRate,d_sampleRate,BC::Unit::Hz);
    store(recLen,d_recordLength);
    store(blockAvg,d_blockAverage);
    store(numAvg,d_numAverages);
    store(multiRec,d_multiRecord);
    store(multiRecNum,d_numRecords);
}

void DigitizerConfig::retrieveValues()
{
    //note: all channels are 1-indexed to allow for aux channel to be 0
    //but channels are stored as 0-indexed
    using namespace BC::Store::Digi;
    int n = arrayStoreSize(an);
    for(int i=0; i<n; ++i)
    {
        bool e = retrieveArrayValue(an,i,en,false);
        auto idx = retrieveArrayValue(an,i,chIndex,i+1);
        AnalogChannel c{ e,retrieveArrayValue(an,i,fs,0.0),
                    retrieveArrayValue(an,i,offset,0.0)};
        d_analogChannels.insert_or_assign(idx,c);
    }
    n = arrayStoreSize(dig);
    for(int i=0; i<n; ++i)
    {
        bool e = retrieveArrayValue(dig,i,en,false);
        auto idx = retrieveArrayValue(dig,i,chIndex,i+1);
        DigitalChannel c{e,retrieveArrayValue(dig,i,digInp,false),retrieveArrayValue(dig,i,digRole,-1)};
            d_digitalChannels.insert_or_assign(idx,c);
    }

    d_triggerChannel = retrieve(trigCh,0);
    d_triggerSlope = retrieve(trigSlope,RisingEdge);
    d_triggerLevel = retrieve(trigLevel,0.0);
    d_triggerDelayUSec = retrieve(trigDelay,0.0);
    d_bytesPerPoint = retrieve(bpp,1);
    d_byteOrder = retrieve(bo,DigitizerConfig::LittleEndian);
    d_sampleRate = retrieve(sRate,0.0);
    d_recordLength = retrieve(recLen,0);
    d_blockAverage = retrieve(blockAvg,false);
    d_numAverages = retrieve(numAvg,1);
    d_multiRecord = retrieve(multiRec,false);
    d_numRecords = retrieve(multiRecNum,1);
}
