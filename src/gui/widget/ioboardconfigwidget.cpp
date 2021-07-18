#include "ioboardconfigwidget.h"

#include <hardware/core/ioboard/ioboard.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>



IOBoardConfigWidget::IOBoardConfigWidget(QWidget *parent) :
    DigitizerConfigWidget("IOBoardConfigWidget",BC::Key::IOB::ioboard,parent)
{
    using namespace BC::Key::Digi;
    using namespace BC::Key::DigiWidget;

    auto l = static_cast<QHBoxLayout*>(layout());

    auto v = new QVBoxLayout;

    SettingsStorage s(d_hwKey,Hardware);

    int ac = s.get(numAnalogChannels,4);
    if(ac > 0)
    {
        p_analogNameWidget = new QTableWidget(ac,1);
        p_analogNameWidget->setHorizontalHeaderLabels({"Analog Channel Name"});

        QStringList hdr;
        for(int i=0;i<ac;++i)
        {
            hdr.append(QString::number(i+1));
            p_analogNameWidget->setItem(i,0,new QTableWidgetItem(getArrayValue(dwAnChannels,i,channelName,QString(""))));
        }
        p_analogNameWidget->setVerticalHeaderLabels(hdr);
        p_analogNameWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

        v->addWidget(p_analogNameWidget,ac);
    }

    int dc = s.get(numDigitalChannels,0);
    if(dc > 0)
    {
        p_digitalNameWidget = new QTableWidget(dc,1);
        p_digitalNameWidget->setHorizontalHeaderLabels({"Digital Channel Name"});

        QStringList hdr;
        for(int i=0;i<ac;++i)
        {
            hdr.append(QString::number(i+1));
            p_digitalNameWidget->setItem(i,0,new QTableWidgetItem(getArrayValue(dwDigChannels,i,channelName,QString(""))));
        }

        p_digitalNameWidget->setVerticalHeaderLabels(hdr);
        p_digitalNameWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        v->addWidget(p_digitalNameWidget,dc);
    }


    l->addLayout(v,1);



}

IOBoardConfigWidget::~IOBoardConfigWidget()
{
    using namespace BC::Key::Digi;
    using namespace BC::Key::DigiWidget;
    SettingsStorage s(d_hwKey,Hardware);


    for(int i=0; i<p_analogNameWidget->rowCount(); ++i)
    {
        auto text = p_analogNameWidget->item(i,0)->text();
        if((std::size_t) i == getArraySize(dwAnChannels))
            appendArrayMap(dwAnChannels,{{channelName,text}});
        else
            setArrayValue(dwAnChannels,i,channelName,text);
    }

    for(int i=0; i<p_digitalNameWidget->rowCount(); ++i)
    {
        auto text = p_digitalNameWidget->item(i,0)->text();
        if((std::size_t) i == getArraySize(dwDigChannels))
            appendArrayMap(dwDigChannels,{{channelName,text}});
        else
            setArrayValue(dwDigChannels,i,channelName,text);
    }

}

void IOBoardConfigWidget::setFromConfig(const IOBoardConfig &cfg)
{
    DigitizerConfigWidget::setFromConfig(static_cast<const DigitizerConfig&>(cfg));

    SettingsStorage s(d_hwKey,Hardware);

    using namespace BC::Key::Digi;
    int ac = s.get(numAnalogChannels,4);
    for(int i=0; i<ac; ++i)
        p_analogNameWidget->item(i,0)->setText(cfg.analogName(i+1));

    int dc = s.get(numDigitalChannels,4);
    for(int i=0; i<dc; ++i)
        p_digitalNameWidget->item(i,0)->setText(cfg.digitalName(i+1));

}

void IOBoardConfigWidget::toConfig(IOBoardConfig &cfg)
{
    DigitizerConfigWidget::toConfig(static_cast<DigitizerConfig&>(cfg));
    if(p_analogNameWidget != nullptr)
    {
        for(int i=0; i<p_analogNameWidget->rowCount(); ++i)
        {
            auto s = p_analogNameWidget->item(i,0)->text();
            setArrayValue(BC::Key::DigiWidget::dwAnChannels,i,BC::Key::DigiWidget::channelName,s,false);
            if(!s.isEmpty())
                cfg.setAnalogName(i+1,s);
        }
    }
    if(p_digitalNameWidget != nullptr)
    {
        for(int i=0; i<p_digitalNameWidget->rowCount(); ++i)
        {
            auto s = p_digitalNameWidget->item(i,0)->text();
            setArrayValue(BC::Key::DigiWidget::dwDigChannels,i,BC::Key::DigiWidget::channelName,s,false);
            if(!s.isEmpty())
                cfg.setDigitalName(i+1,s);
        }
    }
}

std::map<int, QString> IOBoardConfigWidget::getAnalogNames() const
{
    std::map<int,QString> out;

    if(p_analogNameWidget != nullptr)
    {
        for(auto i=0; i<p_analogNameWidget->rowCount(); ++i)
            out.insert_or_assign(i+1,p_analogNameWidget->item(i,0)->text());
    }

    return out;

}

std::map<int, QString> IOBoardConfigWidget::getDigitalNames() const
{
    std::map<int,QString> out;

    if(p_digitalNameWidget != nullptr)
    {
        for(auto i=0; i<p_digitalNameWidget->rowCount(); ++i)
            out.insert_or_assign(i+1,p_digitalNameWidget->item(i,0)->text());
    }

    return out;
}
