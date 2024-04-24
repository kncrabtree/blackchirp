#include "experimentpressurecontrollerconfigpage.h"

using namespace BC::Key::WizPC;

#include <QVBoxLayout>
#include <gui/widget/pressurecontrolwidget.h>

ExperimentPressureControllerConfigPage::ExperimentPressureControllerConfigPage(const QString hwKey, const QString title, Experiment *exp, QWidget *parent) :
    ExperimentConfigPage(key,title,exp,parent)
{
    auto vbl = new QVBoxLayout;

    auto pc = exp->getOptHwConfig<PressureControllerConfig>(hwKey);
    p_pcw = new PressureControlWidget(*pc.lock(),this);
    vbl->addWidget(p_pcw);
    vbl->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::MinimumExpanding));

    setLayout(vbl);

    connect(p_pcw,&PressureControlWidget::pressureControlModeChanged,p_pcw,&PressureControlWidget::pressureControlModeUpdate);

}


void ExperimentPressureControllerConfigPage::initialize()
{
}

bool ExperimentPressureControllerConfigPage::validate()
{
    return true;
}

void ExperimentPressureControllerConfigPage::apply()
{
    p_exp->addOptHwConfig(p_pcw->toConfig());
}
