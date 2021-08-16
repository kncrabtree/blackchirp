#include <hardware/optional/ioboard/ioboardconfig.h>



IOBoardConfig::IOBoardConfig() : DigitizerConfig(BC::Store::Digi::iob)
{
}

void IOBoardConfig::setAnalogName(int ch, const QString name)
{
    d_analogNames.insert_or_assign(ch,name);
}

void IOBoardConfig::setDigitalName(int ch, const QString name)
{
    d_digitalNames.insert_or_assign(ch,name);
}

QString IOBoardConfig::analogName(int ch) const
{
    QString out;
    auto it = d_analogNames.find(ch);
    if(it != d_analogNames.end())
        out = it->second;

    return out;
}

QString IOBoardConfig::digitalName(int ch) const
{
    QString out;
    auto it = d_digitalNames.find(ch);
    if(it != d_digitalNames.end())
        out = it->second;

    return out;
}


void IOBoardConfig::storeValues()
{
    using namespace BC::Store::Digi;

    for(auto it = d_analogNames.cbegin(); it != d_analogNames.cend(); ++it)
        storeArrayValue(an,it->first,name,it->second);
    for(auto it = d_digitalNames.cbegin(); it != d_digitalNames.cend(); ++it)
        storeArrayValue(dig,it->first,name,it->second);

    DigitizerConfig::storeValues();
}

void IOBoardConfig::retrieveValues()
{
    using namespace BC::Store::Digi;

    DigitizerConfig::retrieveValues();

    for(auto it = d_analogChannels.cbegin(); it != d_analogChannels.cend(); ++it)
    {
        auto s = retrieveArrayValue(an,it->first,name,QString(""));
        if(!s.isEmpty())
            d_analogNames.insert_or_assign(it->first,s);
    }

    for(auto it = d_digitalChannels.cbegin(); it != d_digitalChannels.cend(); ++it)
    {
        auto s = retrieveArrayValue(dig,it->first,name,QString(""));
        if(!s.isEmpty())
            d_digitalNames.insert_or_assign(it->first,s);
    }
}
