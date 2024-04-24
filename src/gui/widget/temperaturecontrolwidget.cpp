#include "temperaturecontrolwidget.h"

#include <hardware/optional/tempcontroller/temperaturecontroller.h>

#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QGridLayout>

TemperatureControlWidget::TemperatureControlWidget(const TemperatureControllerConfig &cfg, QWidget *parent) :
    QWidget(parent),
    SettingsStorage(BC::Key::widgetKey(BC::Key::TCW::key,cfg.headerKey(),cfg.hwSubKey())),
    d_config{cfg}
{
    auto gl = new QGridLayout(this);

    gl->addWidget(new QLabel("Ch"),0,0);
    gl->addWidget(new QLabel("Name"),0,1);
    gl->addWidget(new QLabel("Enabled"),0,2);
    gl->setColumnStretch(0,0);
    gl->setColumnStretch(1,1);
    gl->setColumnStretch(2,0);
    gl->setMargin(3);
    gl->setSpacing(3);

    auto numChannels = cfg.numChannels();
    for(uint i=0; i<numChannels; ++i)
    {
        auto le = new QLineEdit(this);
        le->setText(getArrayValue(BC::Key::TCW::channels,i,BC::Key::TCW::chName,QString("")));
        connect(le,&QLineEdit::editingFinished,[this,le,i]{
            emit channelNameChanged(d_config.headerKey(),i,le->text());
        });

        auto b = new QPushButton("Off",this);
        b->setCheckable(true);
        b->setChecked(false);
        connect(b,&QPushButton::toggled,[this,i](bool en){
            emit channelEnableChanged(d_config.headerKey(),i,en);
        });

        gl->addWidget(new QLabel(QString::number(i+1)),i+1,0);
        gl->addWidget(le,i+1,1);
        gl->addWidget(b,i+1,2);

        d_channelWidgets.push_back({le,b});
    }

    setFromConfig(cfg);
}

TemperatureControlWidget::~TemperatureControlWidget()
{
    using namespace BC::Key::TCW;
    setArray(channels,{});
    std::vector<SettingsMap> v;
    for(std::size_t i=0; i<d_channelWidgets.size(); ++i)
        v.push_back({{chName,d_channelWidgets[i].le->text()}});
    setArray(channels,v);
}

TemperatureControllerConfig &TemperatureControlWidget::toConfig()
{
    for(uint i=0; i<d_config.numChannels(); i++)
    {
        if( (std::size_t)i < d_channelWidgets.size())
        {
            d_config.setName(i,d_channelWidgets.at(i).le->text());
            d_config.setEnabled(i,d_channelWidgets.at(i).button->isChecked());
        }
    }

    return d_config;
}

void TemperatureControlWidget::setFromConfig(const TemperatureControllerConfig &cfg)
{
    if(cfg.headerKey() != d_config.headerKey())
        return;

    for(uint i=0; i<cfg.numChannels(); ++i)
    {
        if( (std::size_t)i < d_channelWidgets.size())
            setChannelEnabled(cfg.headerKey(),i,cfg.channelEnabled(i));
    }

    d_config = cfg;
}

void TemperatureControlWidget::setChannelEnabled(const QString key, uint ch, bool en)
{
    if(key != d_config.headerKey())
        return;

    if(ch >= d_channelWidgets.size())
        return;

    auto b = d_channelWidgets[ch].button;
    if(en)
        b->setText("On");
    else
        b->setText("Off");
    b->blockSignals(true);
    b->setChecked(en);
    b->blockSignals(false);
}
