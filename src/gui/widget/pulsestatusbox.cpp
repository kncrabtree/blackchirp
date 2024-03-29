#include "pulsestatusbox.h"

#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <gui/widget/pulseconfigwidget.h>
#include <gui/widget/led.h>

#include <QGridLayout>
#include <QLabel>
#include <QMetaEnum>

PulseStatusBox::PulseStatusBox(QWidget *parent) : QGroupBox(parent)
{
    setTitle(QString("Pulse Status"));

    QGridLayout *gl = new QGridLayout;

    SettingsStorage pg(BC::Key::PGen::key,SettingsStorage::Hardware);
    int channels = pg.get(BC::Key::PGen::numChannels,8);
    d_ledList.reserve(channels);
    for(int i=0; i<channels; i++)
    {       
        QLabel *lbl = new QLabel(this);
        lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);

        Led *led = new Led(this);
        gl->addWidget(lbl,i/4,(2*i)%8,1,1,Qt::AlignVCenter);
        gl->addWidget(led,i/4,((2*i)%8)+1,1,1,Qt::AlignVCenter);

        d_ledList.push_back({lbl,led});
    }
    for(int i=0; i<8; i++)
        gl->setColumnStretch(i,(i%2)+1);

    auto r = gl->rowCount();
    p_repLabel = new QLabel(this);
    gl->addWidget(p_repLabel,r,0,1,7,Qt::AlignRight);

    p_enLed = new Led(this);
    gl->addWidget(p_enLed,r,7);

    gl->setMargin(3);
    gl->setContentsMargins(3,3,3,3);
    gl->setSpacing(3);

    setLayout(gl);

}

void PulseStatusBox::updatePulseLeds(const PulseGenConfig &cc)
{
    static_cast<PulseGenConfig&>(*this) = cc;
    updateAll();
}

void PulseStatusBox::updatePulseLed(int index, Setting s, QVariant val)
{
    if(index < 0 || (std::size_t)index >= d_ledList.size())
        return;

    setCh(index,s,val);

    switch(s) {
    case NameSetting:
        d_ledList.at(index).first->setText(val.toString());
        break;
    case EnabledSetting:
        d_ledList.at(index).second->setState(val.toBool());
        break;
    case ModeSetting:
        if(val.value<ChannelMode>() == Normal)
            d_ledList.at(index).second->setColor(Led::Green);
        else
            d_ledList.at(index).second->setColor(Led::Yellow);
        break;
    default:
        break;
    }
}

void PulseStatusBox::updateRepRate(double rr)
{
    d_repRate = rr;
    if(d_mode == Continuous)
        p_repLabel->setText(QString("Rep Rate: %1 Hz").arg(d_repRate,0,'f',2));
}

void PulseStatusBox::updatePGenMode(PGenMode m)
{
    d_mode = m;
    updateAll();
}

void PulseStatusBox::updatePGenEnabled(bool en)
{
    d_pulseEnabled = en;
    p_enLed->setState(en);
}

void PulseStatusBox::updateAll()
{

    if(d_mode == Triggered)
    {
        p_repLabel->setText(QString("External Trigger"));
        p_enLed->setColor(Led::Yellow);
    }
    else
    {
        p_repLabel->setText(QString("Rep Rate: %1 Hz").arg(d_repRate,0,'f',2));
        p_enLed->setColor(Led::Green);
    }

    p_enLed->setState(d_pulseEnabled);

    for(std::size_t i=0; i<d_ledList.size() && (int)i < d_channels.size(); ++i)
    {
        if(d_channels.at(i).channelName.isEmpty())
        {
            if(d_channels.at(i).role != None)
            {
                auto me = QMetaEnum::fromType<Role>();
                d_ledList.at(i).first->setText(QString(me.valueToKey(d_channels.at(i).role)));
            }
            else
                d_ledList.at(i).first->setText(QString("Ch%1").arg(i+1));
        }
        else
            d_ledList.at(i).first->setText(d_channels.at(i).channelName);
        if(d_channels.at(i).mode == Normal)
            d_ledList.at(i).second->setColor(Led::Green);
        else
            d_ledList.at(i).second->setColor(Led::Yellow);

        d_ledList.at(i).second->setState(d_channels.at(i).enabled);
    }
}
