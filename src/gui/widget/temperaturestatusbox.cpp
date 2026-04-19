#include "temperaturestatusbox.h"

#include <QGridLayout>
#include <QLabel>

#include <data/storage/settingsstorage.h>
#include <gui/style/themecolors.h>
#include <gui/util/numericformat.h>
#include <hardware/optional/tempcontroller/temperaturecontroller.h>

using namespace Qt::Literals::StringLiterals;

TemperatureStatusBox::TemperatureStatusBox(const QString key, QWidget *parent) :
    HardwareStatusBox(key,parent)
{
    auto gl = new QGridLayout;

    SettingsStorage tc(d_key,SettingsStorage::Hardware);
    auto nc = tc.get(BC::Key::TC::numChannels,4);

    for(int i=0; i<nc; ++i)
    {
        auto n = tc.getArrayValue(BC::Key::TC::channels,i,BC::Key::TC::chName,QString("Ch%1").arg(i+1));
        auto lbl = new QLabel(n);
        lbl->setMinimumWidth(70);
        lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
        gl->addWidget(lbl,i,0);

        auto val = new QLabel;
        val->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
        gl->addWidget(val,i,1);

        int decimals = tc.getArrayValue(BC::Key::TC::channels,i,BC::Key::TC::decimals,4);
        QString suffix = " "_L1 + tc.getArrayValue(BC::Key::TC::channels,i,BC::Key::TC::units,QString("K"));

        d_widgets.push_back({lbl,val,decimals,suffix,true});
    }

    p_noActiveLabel = new QLabel(tr("No active channels"));
    p_noActiveLabel->setAlignment(Qt::AlignCenter);
    p_noActiveLabel->setStyleSheet(QString("QLabel { color: %1; font-style: italic; }").arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
    gl->addWidget(p_noActiveLabel,nc,0,1,2);

    gl->setColumnStretch(0,0);
    gl->setColumnStretch(1,1);

    body()->setLayout(gl);

    loadFromSettings();
}

void TemperatureStatusBox::loadFromSettings()
{
    SettingsStorage tc(d_key,SettingsStorage::Hardware);
    for(std::size_t i=0; i<d_widgets.size(); ++i)
    {
        d_widgets[i].decimals = tc.getArrayValue(BC::Key::TC::channels,i,BC::Key::TC::decimals,4);
        d_widgets[i].suffix = " "_L1 + tc.getArrayValue(BC::Key::TC::channels,i,BC::Key::TC::units,QString("K"));
    }
}

void TemperatureStatusBox::setTemperature(const QString key, uint ch, double t)
{
    if(key != d_key)
        return;

    if(ch >= d_widgets.size())
        return;

    auto &w = d_widgets[ch];
    w.value->setText(BC::Gui::formatNumberForDisplay(t, w.decimals) + w.suffix);
}

void TemperatureStatusBox::setChannelName(const QString key, uint ch, const QString name)
{
    if(key != d_key)
        return;

    if(ch >= d_widgets.size())
        return;

    auto lbl = d_widgets[ch].label;
    if(name.isEmpty())
        lbl->setText(QString("Ch%1").arg(ch+1));
    else
        lbl->setText(name);
}

void TemperatureStatusBox::setChannelEnabled(const QString key, uint ch, bool en)
{
    if(key != d_key)
        return;

    if(ch >= d_widgets.size())
        return;

    auto &w = d_widgets[ch];
    w.label->setVisible(en);
    w.value->setVisible(en);
    w.active = en;

    updateNoActiveLabel();
}

void TemperatureStatusBox::updateNoActiveLabel()
{
    bool anyActive = false;
    for(const auto &chw : d_widgets)
        anyActive = anyActive || chw.active;

    p_noActiveLabel->setVisible(!anyActive);
}
