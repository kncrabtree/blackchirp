#include "experimentflowconfigpage.h"

using namespace BC::Key::WizFlow;

#include <QVBoxLayout>

#include <gui/widget/gascontrolwidget.h>

ExperimentFlowConfigPage::ExperimentFlowConfigPage(const QString hwKey, const QString title, Experiment *exp, QWidget *parent)
    : ExperimentConfigPage(key,title,exp,parent)
{
    auto *vbl = new QVBoxLayout;

    auto fc = exp->getOptHwConfig<FlowConfig>(hwKey);
    p_gcw = new GasControlWidget(*fc.lock(),this);
    vbl->addWidget(p_gcw);
    vbl->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::MinimumExpanding));

    setLayout(vbl);

}


void ExperimentFlowConfigPage::initialize()
{
}

bool ExperimentFlowConfigPage::validate()
{
    if(!isEnabled())
        return true;

    auto &c = p_gcw->toConfig();
    auto en = false;
    for(int i=0; i<c.size(); i++)
    {
        auto chEn = c.setting(i,FlowConfig::Enabled).toBool();
        if(chEn)
        {
            auto n = c.setting(i,FlowConfig::Name).toString();
            if(n.isEmpty())
                emit warning(QString("Gas channel %1 on %2 is enabled but has no name.").arg(i+1).arg(d_title));

        }
        en |= chEn;
    }

    if(!en)
        emit warning(QString("No gas flow channels enabled on %1.").arg(d_title));

    return true;
}

void ExperimentFlowConfigPage::apply()
{
    p_exp->addOptHwConfig(p_gcw->toConfig());
}
