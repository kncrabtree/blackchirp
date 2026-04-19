#include "gasflowdisplaywidget.h"

#include <QGridLayout>
#include <QLabel>

#include <gui/widget/led.h>
#include <gui/widget/gascontrolwidget.h>
#include <gui/util/numericformat.h>
#include <data/storage/settingsstorage.h>
#include <hardware/optional/flowcontroller/flowcontroller.h>

using namespace BC::Key::Flow;
using namespace Qt::Literals::StringLiterals;

GasFlowDisplayBox::GasFlowDisplayBox(const QString key, QWidget *parent) : HardwareStatusBox(key,parent)
{
    auto gl = new QGridLayout;
    gl->setSpacing(3);
    gl->setContentsMargins(3,3,3,3);
    gl->setColumnStretch(0, 1);
    gl->setColumnStretch(1, 0);
    gl->setColumnStretch(2, 0);

    p_pressureLabel = new QLabel;
    p_pressureLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    p_pressureLed = new Led;

    gl->addWidget(new QLabel("Pressure"_L1), 0, 0, Qt::AlignRight | Qt::AlignVCenter);
    gl->addWidget(p_pressureLabel, 0, 1);
    gl->addWidget(p_pressureLed,   0, 2);

    SettingsStorage fc(key, SettingsStorage::Hardware);
    int n = fc.get(flowChannels, 4);
    for (int i = 0; i < n; ++i)
    {
        auto name = fc.getArrayValue(channels, i, BC::Key::Flow::chName, QString());
        auto nameLabel = new QLabel(name.isEmpty() ? QString("Ch%1"_L1).arg(i+1) : name);
        nameLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

        auto valueLabel = new QLabel;
        valueLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);

        auto led = new Led;

        d_flowWidgets.append({nameLabel, valueLabel, led});
        d_channelDecimals.append(2);
        d_channelSuffix.append(QString());
        d_setpoints.append(0.0);
        d_channelNames.append(name);
    }

    body()->setLayout(gl);

    addChannelsToGrid(gl);
    applySettings();
}

void GasFlowDisplayBox::addChannelsToGrid(QGridLayout *gl)
{
    for (int i = 0; i < d_flowWidgets.size(); ++i)
    {
        auto [nameLabel, valueLabel, led] = d_flowWidgets.at(i);
        gl->addWidget(nameLabel,  i + 1, 0, Qt::AlignRight | Qt::AlignVCenter);
        gl->addWidget(valueLabel, i + 1, 1);
        gl->addWidget(led,        i + 1, 2);
        updateChannelVisibility(i);
    }
}

void GasFlowDisplayBox::rebuild()
{
    auto gl = qobject_cast<QGridLayout*>(body()->layout());

    for (auto& fw : d_flowWidgets)
    {
        gl->removeWidget(std::get<0>(fw));
        gl->removeWidget(std::get<1>(fw));
        gl->removeWidget(std::get<2>(fw));
        delete std::get<0>(fw);
        delete std::get<1>(fw);
        delete std::get<2>(fw);
    }
    d_flowWidgets.clear();
    d_channelDecimals.clear();
    d_channelSuffix.clear();
    d_setpoints.clear();
    d_channelNames.clear();

    SettingsStorage fc(d_key, SettingsStorage::Hardware);
    int n = fc.get(flowChannels, 0);
    for (int i = 0; i < n; ++i)
    {
        auto name = fc.getArrayValue(channels, i, chName, QString());
        auto nameLabel = new QLabel(name.isEmpty() ? QString("Ch%1"_L1).arg(i+1) : name);
        nameLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

        auto valueLabel = new QLabel;
        valueLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);

        auto led = new Led;

        d_flowWidgets.append({nameLabel, valueLabel, led});
        d_channelDecimals.append(2);
        d_channelSuffix.append(QString());
        d_setpoints.append(0.0);
        d_channelNames.append(name);
    }

    addChannelsToGrid(gl);
    applySettings();
}

void GasFlowDisplayBox::applySettings()
{
    SettingsStorage fc(d_key, SettingsStorage::Hardware);

    d_pressureDecimals = fc.get(pDec, 3);
    d_pressureSuffix = QString(" "_L1) + fc.get(pUnits, QString());

    for (int i = 0; i < d_flowWidgets.size(); ++i)
    {
        d_channelDecimals[i] = fc.getArrayValue(channels, i, chDecimals, 2);
        d_channelSuffix[i] = QString(" "_L1) + fc.getArrayValue(channels, i, chUnits, QString());
    }
}

void GasFlowDisplayBox::updateChannelVisibility(int ch)
{
    bool visible = !qFuzzyCompare(1.0, d_setpoints.at(ch) + 1.0) || !d_channelNames.at(ch).isEmpty();
    auto [nameLabel, valueLabel, led] = d_flowWidgets.at(ch);
    nameLabel->setVisible(visible);
    valueLabel->setVisible(visible);
    led->setVisible(visible);
}

void GasFlowDisplayBox::updateFlow(const QString key, int ch, double val)
{
    if (key != d_key)
        return;

    if (ch < 0 || ch >= d_flowWidgets.size())
        return;

    auto label = std::get<1>(d_flowWidgets.at(ch));
    label->setText(BC::Gui::formatNumberForDisplay(val, d_channelDecimals.at(ch)) + d_channelSuffix.at(ch));
}

void GasFlowDisplayBox::updateFlowName(const QString key, int ch, const QString name)
{
    if (key != d_key)
        return;

    if (ch < 0 || ch >= d_flowWidgets.size())
        return;

    d_channelNames[ch] = name;

    auto lbl = std::get<0>(d_flowWidgets.at(ch));
    lbl->setText(name.isEmpty() ? QString("Ch%1"_L1).arg(ch+1) : name.mid(0, 9));

    updateChannelVisibility(ch);
}

void GasFlowDisplayBox::updateSetpointTooltip(int ch)
{
    double sp = d_setpoints.at(ch);
    QString tip = QString("Setpoint: %1%2"_L1)
                      .arg(BC::Gui::formatNumberForDisplay(sp, d_channelDecimals.at(ch)),
                           d_channelSuffix.at(ch));
    auto [nameLabel, valueLabel, led] = d_flowWidgets.at(ch);
    nameLabel->setToolTip(tip);
    valueLabel->setToolTip(tip);
    led->setToolTip(tip);
}

void GasFlowDisplayBox::updateFlowSetpoint(const QString key, int ch, double val)
{
    if (key != d_key)
        return;

    if (ch < 0 || ch >= d_flowWidgets.size())
        return;

    d_setpoints[ch] = val;
    updateSetpointTooltip(ch);

    bool active = !qFuzzyCompare(1.0, val + 1.0);
    std::get<2>(d_flowWidgets.at(ch))->setState(active);
    updateChannelVisibility(ch);
}

void GasFlowDisplayBox::updatePressureControl(const QString key, bool en)
{
    if (key != d_key)
        return;

    p_pressureLed->setState(en);
}

void GasFlowDisplayBox::updatePressure(const QString key, double p)
{
    if (key != d_key)
        return;

    p_pressureLabel->setText(BC::Gui::formatNumberForDisplay(p, d_pressureDecimals) + d_pressureSuffix);
}
