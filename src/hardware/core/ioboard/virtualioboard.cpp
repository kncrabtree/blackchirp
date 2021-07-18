#include "virtualioboard.h"

VirtualIOBoard::VirtualIOBoard(QObject *parent) :
    IOBoard(BC::Key::hwVirtual,BC::Key::IOB::viobName,CommunicationProtocol::Virtual,parent)
{
    using namespace BC::Key::Digi;

    setDefault(numAnalogChannels,8);
    setDefault(numDigitalChannels,8);
    setDefault(hasAuxTriggerChannel,false);
    setDefault(minFullScale,2.44);
    setDefault(maxFullScale,2.44);
    setDefault(minVOffset,0.0);
    setDefault(maxVOffset,0.0);
    setDefault(isTriggered,false);
    setDefault(minTrigDelay,0.0);
    setDefault(maxTrigDelay,0.0);
    setDefault(minTrigLevel,0.0);
    setDefault(maxTrigLevel,0.0);
    setDefault(maxRecordLength,1);
    setDefault(canBlockAverage,false);
    setDefault(canMultiRecord,false);
    setDefault(multiBlock,false);
    setDefault(maxBytes,2);

    if(!containsArray(sampleRates))
        setArray(sampleRates,{
                     {{srText,"N/A"},{srValue,0.0}},
                 });
}




bool VirtualIOBoard::testConnection()
{
    return true;
}

void VirtualIOBoard::initialize()
{
}


std::map<int, double> VirtualIOBoard::readAnalogChannels()
{
    std::map<int,double> out;
    for(auto it = d_analogChannels.begin(); it != d_analogChannels.cend(); ++it)
        out.insert({it->first,static_cast<double>(qrand() % (2 << d_bytesPerPoint*8))/
                    (2 << d_bytesPerPoint*8)*it->second.fullScale});

    return out;
}

std::map<int, bool> VirtualIOBoard::readDigitalChannels()
{
    std::map<int,bool> out;
    for(auto it = d_digitalChannels.begin(); it != d_digitalChannels.cend(); ++it)
        out.insert({it->first,static_cast<bool>(qrand()%2)});

    return out;
}
