#include "experimenttemperaturecontrollerconfigpage.h"

using namespace BC::Key::WizTC;

#include <QVBoxLayout>
#include <gui/widget/temperaturecontrolwidget.h>

ExperimentTemperatureControllerConfigPage::ExperimentTemperatureControllerConfigPage(const QString hwKey, const QString title, Experiment *exp, QWidget *parent) :
    ExperimentConfigPage(key,title,exp,parent)
{

    auto vbl = new QVBoxLayout;

    auto tc = exp->getOptHwConfig<TemperatureControllerConfig>(hwKey);
    p_tcw = new TemperatureControlWidget(*tc.lock(),this);
    vbl->addWidget(p_tcw);
    vbl->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::MinimumExpanding));

    setLayout(vbl);

    connect(p_tcw,&TemperatureControlWidget::channelEnableChanged,p_tcw,&TemperatureControlWidget::setChannelEnabled);
}


void ExperimentTemperatureControllerConfigPage::initialize()
{
}

bool ExperimentTemperatureControllerConfigPage::validate()
{
    if(!isEnabled())
        return true;

    auto &c = p_tcw->toConfig();
    auto en = false;
    for(uint i=0; i<c.numChannels(); i++)
    {
        auto chEn = c.channelEnabled(i);
        if(chEn)
        {
            auto n = c.channelName(i);
            if(n.isEmpty())
                emit warning(QString("Temperature channel %1 on %2 is enabled but has no name.").arg(i+1).arg(d_title));

        }
        en |= chEn;
    }

    if(!en)
        emit warning(QString("No channels enabled on %1.").arg(d_title));

    return true;
}

void ExperimentTemperatureControllerConfigPage::apply()
{
    p_exp->addOptHwConfig(p_tcw->toConfig());
}
