#include "gasflowdisplaywidget.h"

#include <QGridLayout>
#include <QLabel>
#include <QDoubleSpinBox>


#include <gui/widget/led.h>
#include <gui/widget/gascontrolwidget.h>
#include <data/storage/settingsstorage.h>
#include <hardware/optional/flowcontroller/flowcontroller.h>

using namespace BC::Key::Flow;

GasFlowDisplayBox::GasFlowDisplayBox(const QString key, QWidget *parent) : HardwareStatusBox(key,parent)
{
    QGridLayout *gl = new QGridLayout(this);
    gl->setSpacing(3);
    gl->setContentsMargins(3,3,3,3);

    p_pressureBox = new QDoubleSpinBox;
    p_pressureBox->setReadOnly(true);
    p_pressureBox->blockSignals(true);
    p_pressureBox->setButtonSymbols(QAbstractSpinBox::NoButtons);
    p_pressureBox->setFocusPolicy(Qt::ClickFocus);

    p_pressureLed = new Led;
    gl->addWidget(new QLabel("Pressure"),0,0,Qt::AlignRight);
    gl->addWidget(p_pressureBox,0,1);
    gl->addWidget(p_pressureLed,0,2);

    SettingsStorage fc(key,SettingsStorage::Hardware);
    int n = fc.get(flowChannels,4);
    for(int i=0; i<n; ++i)
    {
        auto n = fc.getArrayValue(channels,i,chName,QString("Ch%1").arg(i+1));
        auto nameLabel = new QLabel(n);
        nameLabel->setMinimumWidth(QFontMetrics(QFont(QString("sans-serif"))).horizontalAdvance(QString("MMMMMMMM")));
        nameLabel->setAlignment(Qt::AlignRight|Qt::AlignVCenter);

        auto led = new Led();

        auto displayBox = new QDoubleSpinBox();
        displayBox->blockSignals(true);
        displayBox->setReadOnly(true);
        displayBox->setFocusPolicy(Qt::ClickFocus);
        displayBox->setButtonSymbols(QAbstractSpinBox::NoButtons);

        d_flowWidgets.append({nameLabel,displayBox,led});

        gl->addWidget(nameLabel,i+1,0,1,1,Qt::AlignRight);
        gl->addWidget(displayBox,i+1,1,1,1);
        gl->addWidget(led,i+1,2,1,1);
    }

    setLayout(gl);

    applySettings();
}

void GasFlowDisplayBox::applySettings()
{
    SettingsStorage fc(d_key,SettingsStorage::Hardware);

    p_pressureBox->setDecimals(fc.get(pDec,3));
    p_pressureBox->setRange(-fc.get(pMax,10.0),fc.get(pMax,10.0));
    p_pressureBox->setSuffix(QString(" ") + fc.get(pUnits,QString("")));

    for(int i=0; i<d_flowWidgets.size(); ++i)
    {
        auto b = std::get<1>(d_flowWidgets.at(i));
        b->setDecimals(fc.getArrayValue(channels,i,chDecimals,2));
        b->setRange(-fc.getArrayValue(channels,i,chMax,10000.0),fc.getArrayValue(channels,i,chMax,10000.0));
        b->setSuffix(QString(" ")+fc.getArrayValue(channels,i,chUnits,QString("")));
    }

}

void GasFlowDisplayBox::updateFlow(const QString key, int ch, double val)
{
    if(key != d_key)
        return;

    if(ch < 0 || ch >= d_flowWidgets.size())
        return;

    std::get<1>(d_flowWidgets.at(ch))->setValue(val);
}

void GasFlowDisplayBox::updateFlowName(const QString key, int ch, const QString name)
{
    if(key != d_key)
        return;

    if(ch < 0 || ch >= d_flowWidgets.size())
        return;

    auto lbl = std::get<0>(d_flowWidgets.at(ch));

    if(name.isEmpty())
        lbl->setText(QString("Ch%1").arg(ch+1));
    else
        lbl->setText(name.mid(0,9));
}

void GasFlowDisplayBox::updateFlowSetpoint(const QString key, int ch, double val)
{
    if(key != d_key)
        return;

    if(ch < 0 || ch >= d_flowWidgets.size())
        return;

    auto led = std::get<2>(d_flowWidgets.at(ch));
    if(qFuzzyCompare(1.0,val+1.0))
        led->setState(false);
    else
        led->setState(true);
}

void GasFlowDisplayBox::updatePressureControl(const QString key, bool en)
{
    if(key != d_key)
        return;

    p_pressureLed->setState(en);
}

void GasFlowDisplayBox::updatePressure(const QString key, double p)
{
    if(key != d_key)
        return;

    p_pressureBox->setValue(p);
}
