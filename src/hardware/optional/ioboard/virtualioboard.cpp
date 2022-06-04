#include "virtualioboard.h"

#include <QRandomGenerator>

VirtualIOBoard::VirtualIOBoard(QObject *parent) :
    IOBoard(BC::Key::Comm::hwVirtual,BC::Key::IOB::viobName,CommunicationProtocol::Virtual,parent)
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

    save();
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
    auto qr = QRandomGenerator::global();
    std::map<int,double> out;
    for(auto it = d_analogChannels.cbegin(); it != d_analogChannels.cend(); ++it)
        out.insert({it->first,static_cast<double>(qr->bounded(0,2 << d_bytesPerPoint*8))/
                    (2 << d_bytesPerPoint*8)*it->second.fullScale});

    return out;
}

std::map<int, bool> VirtualIOBoard::readDigitalChannels()
{
    auto qr = QRandomGenerator::global();
    std::map<int,bool> out;
    for(auto it = d_digitalChannels.cbegin(); it != d_digitalChannels.cend(); ++it)
        out.insert({it->first,static_cast<bool>(qr->bounded(1))});

    return out;
}
