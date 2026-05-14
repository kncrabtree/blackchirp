#include "ioboardconfigwidget.h"

#include <QTableWidget>

#include <hardware/optional/ioboard/ioboard.h>

#include <data/settings/hardwarekeys.h>

IOBoardConfigWidget::IOBoardConfigWidget(IOBoardConfig &cfg, QWidget *parent) :
    DigitizerConfigWidget("IOBoardConfigWidget",cfg.headerKey(),true,parent)
{
    setFromConfig(cfg);
}

IOBoardConfigWidget::~IOBoardConfigWidget()
{
}

void IOBoardConfigWidget::setFromConfig(const IOBoardConfig &cfg)
{
    DigitizerConfigWidget::setFromConfig(static_cast<const DigitizerConfig&>(cfg));

    SettingsStorage s(d_hwKey,Hardware);

    using namespace BC::Key::Digi;
    int ac = s.get(numAnalogChannels,4);
    for(int i=0; i<ac; ++i)
        setAnalogChannelName(i+1,cfg.analogName(i+1));

    int dc = s.get(numDigitalChannels,0);
    for(int i=0; i<dc; ++i)
        setDigitalChannelName(i+1,cfg.digitalName(i+1));
}

void IOBoardConfigWidget::toConfig(IOBoardConfig &cfg)
{
    DigitizerConfigWidget::toConfig(static_cast<DigitizerConfig&>(cfg));

    SettingsStorage s(d_hwKey,Hardware);
    using namespace BC::Key::Digi;

    int ac = s.get(numAnalogChannels,4);
    for(int i=0; i<ac; ++i)
    {
        auto name = analogChannelName(i+1);
        if(!name.isEmpty())
            cfg.setAnalogName(i+1,name);
    }

    int dc = s.get(numDigitalChannels,0);
    for(int i=0; i<dc; ++i)
    {
        auto name = digitalChannelName(i+1);
        if(!name.isEmpty())
            cfg.setDigitalName(i+1,name);
    }
}

std::map<int, QString> IOBoardConfigWidget::getAnalogNames() const
{
    std::map<int,QString> out;
    if(!p_anTable)
        return out;
    for(int i=0; i<p_anTable->rowCount(); ++i)
        out.insert_or_assign(i+1,analogChannelName(i+1));
    return out;
}

std::map<int, QString> IOBoardConfigWidget::getDigitalNames() const
{
    std::map<int,QString> out;
    if(!p_digTable)
        return out;
    for(int i=0; i<p_digTable->rowCount(); ++i)
        out.insert_or_assign(i+1,digitalChannelName(i+1));
    return out;
}
