#include <hardware/optional/ioboard/ioboardconfig.h>

#include <hardware/optional/ioboard/ioboard.h>

IOBoardConfig::IOBoardConfig(const QString subKey, int index) : DigitizerConfig(BC::Key::hwKey(BC::Key::IOB::ioboard,index),subKey)
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

    int i=0;
    for(auto const &[k,n] : d_analogNames)
    {
        storeArrayValue(an,i,chName,n);
        i++;
    }

    i=0;
    for(auto const &[k,n] : d_digitalNames)
    {
        storeArrayValue(dig,i,chName,n);
        i++;
    }

    DigitizerConfig::storeValues();
}

void IOBoardConfig::retrieveValues()
{
    using namespace BC::Store::Digi;

    DigitizerConfig::retrieveValues();

    for(uint i=0; i<arrayStoreSize(an); i++)
    {
        auto s = retrieveArrayValue(an,i,chName,QString(""));
        auto k = retrieveArrayValue(an,i,chIndex,i+1);
        d_analogNames.insert_or_assign(k,s);
    }

    for(uint i=0; i<arrayStoreSize(dig); i++)
    {
        auto s = retrieveArrayValue(dig,i,chName,QString(""));
        auto k = retrieveArrayValue(dig,i,chIndex,i+1);
        d_digitalNames.insert_or_assign(k,s);
    }
}
