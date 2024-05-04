#include "pulsestatusbox.h"

#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <gui/widget/pulseconfigwidget.h>
#include <gui/widget/led.h>

#include <QGridLayout>
#include <QLabel>
#include <QMetaEnum>

PulseStatusBox::PulseStatusBox(QString key, QWidget *parent) :
    HardwareStatusBox(key,parent)
{
    QGridLayout *gl = new QGridLayout;

    SettingsStorage pg(key,SettingsStorage::Hardware);
    int channels = pg.get(BC::Key::PGen::numChannels,8);
    d_ledList.reserve(channels);
    for(int i=0; i<channels; i++)
    {       
        QLabel *lbl = new QLabel(QString("Ch%1").arg(i+1),this);
        lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);

        Led *led = new Led(this);
        gl->addWidget(lbl,i/4,(2*i)%8,1,1,Qt::AlignVCenter);
        gl->addWidget(led,i/4,((2*i)%8)+1,1,1,Qt::AlignVCenter);

        d_ledList.push_back({lbl,led});
    }
    for(int i=0; i<8; i++)
        gl->setColumnStretch(i,(i%2)+1);

    auto r = gl->rowCount();
    p_repLabel = new QLabel(QString("Disconnected"),this);
    gl->addWidget(p_repLabel,r,0,1,7,Qt::AlignRight);

    p_enLed = new Led(this);
    gl->addWidget(p_enLed,r,7);

    gl->setMargin(3);
    gl->setContentsMargins(3,3,3,3);
    gl->setSpacing(3);

    setLayout(gl);

}

void PulseStatusBox::updatePulseLeds(const QString k, const PulseGenConfig &cc)
{
    if(k != d_key)
        return;

    d_config = cc;
    updateAll();
}

void PulseStatusBox::updatePulseSetting(const QString k, int index, PulseGenConfig::Setting s, QVariant val)
{
    if(k != d_key)
        return;

    if(index < 0 || (std::size_t)index >= d_ledList.size())
        return;

    d_config.setCh(index,s,val);

    switch(s) {
    case PulseGenConfig::NameSetting:
        d_ledList.at(index).first->setText(val.toString());
        break;
    case PulseGenConfig::EnabledSetting:
        d_ledList.at(index).second->setState(val.toBool());
        break;
    case PulseGenConfig::ModeSetting:
        if(val.value<PulseGenConfig::ChannelMode>() == PulseGenConfig::Normal)
            d_ledList.at(index).second->setColor(Led::Green);
        else
            d_ledList.at(index).second->setColor(Led::Yellow);
        break;
    case PulseGenConfig::RepRateSetting:
        d_config.d_repRate = val.toDouble();
        if(d_config.d_mode == PulseGenConfig::Continuous)
            p_repLabel->setText(QString("Rep Rate: %1 Hz").arg(d_config.d_repRate,0,'f',2));
        break;
    case PulseGenConfig::PGenModeSetting:
        d_config.d_mode = val.value<PulseGenConfig::PGenMode>();
        updateAll();
        break;
    case PulseGenConfig::PGenEnabledSetting:
        d_config.d_pulseEnabled = val.toBool();
        p_enLed->setState(d_config.d_pulseEnabled);
        break;
    default:
        break;
    }
}

void PulseStatusBox::updateAll()
{

    if(d_config.d_mode != PulseGenConfig::Continuous)
    {
        p_repLabel->setText(QString("External Trigger"));
        p_enLed->setColor(Led::Yellow);
    }
    else
    {
        p_repLabel->setText(QString("Rep Rate: %1 Hz").arg(d_config.d_repRate,0,'f',2));
        p_enLed->setColor(Led::Green);
    }

    p_enLed->setState(d_config.d_pulseEnabled);

    for(std::size_t i=0; i<d_ledList.size() && (int)i < d_config.size(); ++i)
    {
        if(d_config.d_channels.at(i).channelName.isEmpty())
        {
            if(d_config.d_channels.at(i).role != PulseGenConfig::None)
            {
                auto me = QMetaEnum::fromType<PulseGenConfig::Role>();
                d_ledList.at(i).first->setText(QString(me.valueToKey(d_config.d_channels.at(i).role)));
            }
            else
                d_ledList.at(i).first->setText(QString("Ch%1").arg(i+1));
        }
        else
            d_ledList.at(i).first->setText(d_config.d_channels.at(i).channelName);
        if(d_config.d_channels.at(i).mode == PulseGenConfig::Normal)
            d_ledList.at(i).second->setColor(Led::Green);
        else
            d_ledList.at(i).second->setColor(Led::Yellow);

        d_ledList.at(i).second->setState(d_config.d_channels.at(i).enabled);
    }
}
