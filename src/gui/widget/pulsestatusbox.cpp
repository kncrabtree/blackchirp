#include "pulsestatusbox.h"

#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <gui/widget/pulseconfigwidget.h>
#include <gui/widget/led.h>

#include <QGridLayout>
#include <QLabel>

PulseStatusBox::PulseStatusBox(QWidget *parent) : QGroupBox(parent)
{
    setTitle(QString("Pulse Status"));

    QGridLayout *gl = new QGridLayout;

    SettingsStorage pg(BC::Key::PGen::key,SettingsStorage::Hardware);
    SettingsStorage pw(BC::Key::PulseWidget::key);
    int channels = pg.get(BC::Key::PGen::numChannels,8);
    d_ledList.reserve(channels);
    for(int i=0; i<channels; i++)
    {
        auto txt = pw.getArrayValue<QString>(BC::Key::PulseWidget::channels,i,
                                             BC::Key::PulseWidget::name,"Ch"+QString::number(i));
        QLabel *lbl = new QLabel(txt,this);
        lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);

        Led *led = new Led(this);
        gl->addWidget(lbl,i/4,(2*i)%8,1,1,Qt::AlignVCenter);
        gl->addWidget(led,i/4,((2*i)%8)+1,1,1,Qt::AlignVCenter);

        d_ledList.push_back({lbl,led});
    }
    for(int i=0; i<8; i++)
        gl->setColumnStretch(i,(i%2)+1);

    gl->setMargin(3);
    gl->setContentsMargins(3,3,3,3);
    gl->setSpacing(3);

    setLayout(gl);
}

void PulseStatusBox::updatePulseLeds(const PulseGenConfig &cc)
{
    for(std::size_t i=0; i<d_ledList.size() && (int)i < cc.size(); ++i)
        d_ledList.at(i).second->setState(cc.at(i).enabled);
}

void PulseStatusBox::updatePulseLed(int index, PulseGenConfig::Setting s, QVariant val)
{
    if(index < 0 || (std::size_t)index >= d_ledList.size())
        return;

    switch(s) {
    case PulseGenConfig::NameSetting:
        d_ledList.at(index).first->setText(val.toString());
        break;
    case PulseGenConfig::EnabledSetting:
        d_ledList.at(index).second->setState(val.toBool());
        break;
    default:
        break;
    }
}
