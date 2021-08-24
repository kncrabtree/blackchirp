#include "temperaturecontrolwidget.h"

#include <hardware/optional/tempcontroller/temperaturecontroller.h>

#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QGridLayout>

TemperatureControlWidget::TemperatureControlWidget(QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::TCW::key)
{
    auto gl = new QGridLayout(this);

    gl->addWidget(new QLabel("Ch",0,0));
    gl->addWidget(new QLabel("Name"),0,1);
    gl->addWidget(new QLabel("Enabled"),0,2);
    gl->setColumnStretch(0,0);
    gl->setColumnStretch(1,1);
    gl->setColumnStretch(2,0);
    gl->setMargin(3);
    gl->setSpacing(3);

    SettingsStorage tc(BC::Key::TC::key,Hardware);
    auto numChannels = tc.get(BC::Key::TC::numChannels,4);
    for(int i=0; i<numChannels; ++i)
    {
        auto le = new QLineEdit(this);
        le->setText(getArrayValue(BC::Key::TCW::channels,i,BC::Key::TCW::chName,QString("")));
        connect(le,&QLineEdit::editingFinished,[this,le,i]{ emit channelNameChanged(i,le->text()); });

        auto b = new QPushButton("Off",this);
        b->setCheckable(true);
        b->setChecked(false);
        connect(b,&QPushButton::toggled,[this,i](bool en){
            emit channelEnableChanged(i,en);
        });

        gl->addWidget(new QLabel(QString::number(i+1)),i+1,0);
        gl->addWidget(le,i+1,1);
        gl->addWidget(b,i+1,2);

        d_channelWidgets.push_back({le,b});
    }
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

void TemperatureControlWidget::setFromConfig(const TemperatureControllerConfig &cfg)
{
    for(int i=0; i<cfg.numChannels(); ++i)
    {
        if( (std::size_t)i < d_channelWidgets.size())
            setChannelEnabled(i,cfg.channelEnabled(i));
    }
}

void TemperatureControlWidget::setChannelEnabled(int ch, bool en)
{
    if(ch < 0 || (std::size_t)ch >= d_channelWidgets.size())
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
