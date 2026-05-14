#include "pulsestatusbox.h"

#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <gui/widget/pulseconfigwidget.h>
#include <gui/widget/led.h>
#include <gui/util/numericformat.h>

#include <QFontMetrics>
#include <QGridLayout>
#include <QLabel>
#include <QMetaEnum>

using namespace Qt::Literals::StringLiterals;

PulseStatusBox::PulseStatusBox(const QString &key, QWidget *parent) :
    HardwareStatusBox(key,parent), d_config(key)
{
    QGridLayout *gl = new QGridLayout;

    // Per-channel label width budget: ~10 M-widths. Channel names longer
    // than this elide with an ellipsis; the full name is preserved in
    // the channel tooltip. Sized for the 2-channel-per-row status grid;
    // each row gets two label/LED pairs.
    d_labelMaxWidth = QFontMetrics(font()).horizontalAdvance(QString("M")) * 10;

    SettingsStorage pg(key,SettingsStorage::Hardware);
    int channels = pg.get(BC::Key::PGen::numChannels,8);
    d_ledList.reserve(channels);
    d_channelFullNames.reserve(channels);
    for(int i=0; i<channels; i++)
    {
        const auto defaultName = QString("Ch%1").arg(i+1);

        QLabel *lbl = new QLabel(defaultName,this);
        lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
        lbl->setMaximumWidth(d_labelMaxWidth);

        Led *led = new Led(this);
        gl->addWidget(lbl,i/2,(2*i)%4,1,1,Qt::AlignVCenter);
        gl->addWidget(led,i/2,((2*i)%4)+1,1,1,Qt::AlignVCenter);

        d_ledList.push_back({lbl,led});
        d_channelFullNames.push_back(defaultName);
    }
    int numCols = channels > 2 ? 4 : channels*2;
    for(int i=0; i<numCols; i++)
        gl->setColumnStretch(i,(i%2)+1);

    auto r = gl->rowCount();
    p_repLabel = new QLabel(QString("Disconnected"),this);
    gl->addWidget(p_repLabel,r,0,1,3,Qt::AlignRight);

    p_enLed = new Led(this);
    gl->addWidget(p_enLed,r,3);

    gl->setContentsMargins(3,3,3,3);
    gl->setSpacing(3);

    body()->setLayout(gl);

}

void PulseStatusBox::rebuild()
{
    auto gl = qobject_cast<QGridLayout*>(body()->layout());

    for(auto& [lbl, led] : d_ledList)
    {
        gl->removeWidget(lbl);
        gl->removeWidget(led);
        delete lbl;
        delete led;
    }
    d_ledList.clear();
    d_channelFullNames.clear();

    gl->removeWidget(p_repLabel);
    gl->removeWidget(p_enLed);

    SettingsStorage pg(d_key, SettingsStorage::Hardware);
    int channels = pg.get(BC::Key::PGen::numChannels, 8);
    d_ledList.reserve(channels);
    d_channelFullNames.reserve(channels);
    for(int i=0; i<channels; i++)
    {
        const auto defaultName = QString("Ch%1").arg(i+1);

        QLabel *lbl = new QLabel(defaultName,this);
        lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
        lbl->setMaximumWidth(d_labelMaxWidth);

        Led *led = new Led(this);
        gl->addWidget(lbl,i/2,(2*i)%4,1,1,Qt::AlignVCenter);
        gl->addWidget(led,i/2,((2*i)%4)+1,1,1,Qt::AlignVCenter);

        d_ledList.push_back({lbl,led});
        d_channelFullNames.push_back(defaultName);
    }

    int numCols = channels > 2 ? 4 : channels*2;
    for(int i=0; i<numCols; i++)
        gl->setColumnStretch(i,(i%2)+1);

    int r = (channels + 1) / 2;
    gl->addWidget(p_repLabel,r,0,1,3,Qt::AlignRight);
    gl->addWidget(p_enLed,r,3);

    updateAll();
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

    if(index>=0)
    {
        if((std::size_t)index >= d_ledList.size())
            return;
        d_config.setCh(index,s,val);
    }


    switch(s) {
    case PulseGenConfig::NameSetting:
        setChannelLabel(index, val.toString());
        break;
    case PulseGenConfig::EnabledSetting:
        d_ledList.at(index).second->setState(val.toBool());
        break;
    case PulseGenConfig::ModeSetting:
        if(val.value<PulseGenConfig::ChannelMode>() == PulseGenConfig::Normal)
            d_ledList.at(index).second->setColor(Led::Green);
        else
            d_ledList.at(index).second->setColor(Led::Yellow);
        updateChannelTooltip(index);
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
    case PulseGenConfig::DelaySetting:
    case PulseGenConfig::WidthSetting:
    case PulseGenConfig::LevelSetting:
    case PulseGenConfig::SyncSetting:
    case PulseGenConfig::DutyOnSetting:
    case PulseGenConfig::DutyOffSetting:
        updateChannelTooltip(index);
        break;
    default:
        break;
    }
}

void PulseStatusBox::updateChannelTooltip(int ch)
{
    if(ch < 0 || (std::size_t)ch >= d_ledList.size() || ch >= d_config.size())
        return;

    const auto &cc = d_config.d_channels.at(ch);

    QString syncStr;
    if(cc.syncCh == 0)
        syncStr = "T0"_L1;
    else
        syncStr = QString("Ch%1").arg(cc.syncCh);

    QString delayStr = BC::Gui::formatNumberForDisplay(cc.delay, 3) + u" μs"_s;
    QString widthStr = BC::Gui::formatNumberForDisplay(cc.width, 3) + u" μs"_s;
    QString levelStr = (cc.level == PulseGenConfig::ActiveHigh) ? "High"_L1 : "Low"_L1;

    const QString &fullName = d_channelFullNames.at(ch);

    QString tip = "<table>"_L1;
    if(!fullName.isEmpty())
        tip += "<tr><td><b>Name:</b></td><td>"_L1 + fullName.toHtmlEscaped() + "</td></tr>"_L1;
    tip +=
        "<tr><td><b>Sync:</b></td><td>"_L1 + syncStr + "</td></tr>"_L1
        "<tr><td><b>Delay:</b></td><td>"_L1 + delayStr + "</td></tr>"_L1
        "<tr><td><b>Width:</b></td><td>"_L1 + widthStr + "</td></tr>"_L1
        "<tr><td><b>Active:</b></td><td>"_L1 + levelStr + "</td></tr>"_L1;

    if(cc.mode == PulseGenConfig::DutyCycle)
    {
        tip += "<tr><td><b>Duty On:</b></td><td>"_L1 + QString::number(cc.dutyOn) + "</td></tr>"_L1;
        tip += "<tr><td><b>Duty Off:</b></td><td>"_L1 + QString::number(cc.dutyOff) + "</td></tr>"_L1;
    }

    tip += "</table>"_L1;

    d_ledList.at(ch).first->setToolTip(tip);
    d_ledList.at(ch).second->setToolTip(tip);
}

void PulseStatusBox::setChannelLabel(int ch, const QString &fullName)
{
    if(ch < 0 || (std::size_t)ch >= d_ledList.size())
        return;

    d_channelFullNames[ch] = fullName;

    auto lbl = d_ledList.at(ch).first;
    QFontMetrics fm(lbl->font());
    lbl->setText(fm.elidedText(fullName, Qt::ElideRight, d_labelMaxWidth));

    updateChannelTooltip(ch);
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
        QString fullName = d_config.d_channels.at(i).channelName;
        if(fullName.isEmpty())
        {
            if(d_config.d_channels.at(i).role != PulseGenConfig::None)
            {
                auto me = QMetaEnum::fromType<PulseGenConfig::Role>();
                fullName = QString(me.valueToKey(d_config.d_channels.at(i).role));
            }
            else
                fullName = QString("Ch%1").arg(i+1);
        }
        setChannelLabel(i, fullName);
        if(d_config.d_channels.at(i).mode == PulseGenConfig::Normal)
            d_ledList.at(i).second->setColor(Led::Green);
        else
            d_ledList.at(i).second->setColor(Led::Yellow);

        d_ledList.at(i).second->setState(d_config.d_channels.at(i).enabled);
    }
}
