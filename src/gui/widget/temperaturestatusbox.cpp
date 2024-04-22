#include "temperaturestatusbox.h"

#include <QGridLayout>
#include <QLabel>
#include <QDoubleSpinBox>

#include <data/storage/settingsstorage.h>
#include <gui/widget/temperaturecontrolwidget.h>
#include <hardware/optional/tempcontroller/temperaturecontroller.h>

TemperatureStatusBox::TemperatureStatusBox(const QString key, QWidget *parent) :
    HardwareStatusBox(key,parent)
{
    auto gl = new QGridLayout;

    SettingsStorage tc(BC::Key::TC::key,SettingsStorage::Hardware);
    auto nc = tc.get(BC::Key::TC::numChannels,4);

    for(int i=0; i<nc; ++i)
    {
        auto lbl = new QLabel(QString("Ch%1").arg(i+1));
        lbl->setMinimumWidth(70);
        lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
        gl->addWidget(lbl,i,0);

        auto sb = new QDoubleSpinBox;
        sb->setRange(0.0,1000.0);
        sb->blockSignals(true);
        sb->setKeyboardTracking(false);
        sb->setButtonSymbols(QAbstractSpinBox::NoButtons);
        gl->addWidget(sb,i,1);

        d_widgets.push_back({lbl,sb,true});
    }


    gl->setColumnStretch(0,0);
    gl->setColumnStretch(1,1);

    setLayout(gl);

    loadFromSettings();

}

void TemperatureStatusBox::loadFromSettings()
{
    SettingsStorage tcw(BC::Key::TCW::key);
    SettingsStorage tc(BC::Key::TC::key,SettingsStorage::Hardware);
    for(std::size_t i=0; i<d_widgets.size(); ++i)
    {
        auto lbl = d_widgets[i].label;
        auto name = tcw.getArrayValue(BC::Key::TCW::channels,i,BC::Key::TCW::chName,QString(""));
        if(name.isEmpty())
            lbl->setText(QString("Ch%1").arg(i+1));
        else
            lbl->setText(name);

        d_widgets[i].box->setDecimals(tc.getArrayValue(BC::Key::TC::channels,i,BC::Key::TC::decimals,4));

        d_widgets[i].box->setSuffix(QString(" ") + tc.getArrayValue(BC::Key::TC::channels,i,
                                                     BC::Key::TC::units,QString("K")));
    }
}

void TemperatureStatusBox::setTemperature(const QString key, uint ch, double t)
{
    if(key != d_key)
        return;

    if(ch >= d_widgets.size())
        return;

    d_widgets[ch].box->setValue(t);
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
    w.box->setVisible(en);
    w.active = en;

    bool visible = false;
    for(auto &chw : d_widgets)
        visible = visible || chw.active;

    setVisible(visible);
}
