#include "experimentlifconfigpage.h"

#include <QVBoxLayout>

#include <modules/lif/gui/lifcontrolwidget.h>

using namespace BC::Key::WizLif;

ExperimentLifConfigPage::ExperimentLifConfigPage(Experiment *exp, QWidget *parent) :
    ExperimentConfigPage(key,title,exp,parent)
{
    p_lcw = new LifControlWidget;

    auto vbl = new QVBoxLayout;
    vbl->addWidget(p_lcw);

    setLayout(vbl);

    if(p_exp->d_number > 0 && p_exp->lifEnabled())
        p_lcw->setFromConfig(*p_exp->lifConfig());
}


void ExperimentLifConfigPage::initialize()
{
}

bool ExperimentLifConfigPage::validate()
{
    if(!p_exp->lifEnabled())
        return true;

    //consider smarter validation?
    return true;

}

void ExperimentLifConfigPage::apply()
{
    if(isEnabled() && p_exp->lifEnabled())
        p_lcw->toConfig(*p_exp->lifConfig());
}
