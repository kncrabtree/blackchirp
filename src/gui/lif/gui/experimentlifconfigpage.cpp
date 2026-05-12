#include "experimentlifconfigpage.h"

#include <QVBoxLayout>

#include <gui/lif/gui/lifcontrolwidget.h>
#include <data/experiment/hardwaredatacontainer.h>

using namespace BC::Key::WizLif;

ExperimentLifConfigPage::ExperimentLifConfigPage(Experiment *exp, QWidget *parent) :
    ExperimentConfigPage(key,title,exp,parent)
{
    // Look up LIF scope and laser hardware keys from experiment's hardware data
    QString digitizerHwKey;
    QString laserHwKey;
    for (auto it = exp->d_hardwareData.hardwareMap.cbegin();
         it != exp->d_hardwareData.hardwareMap.cend(); ++it) {
        if (it.value().type == BC::Data::HardwareType::LifDigitizer)
            digitizerHwKey = it.key();
        else if (it.value().type == BC::Data::HardwareType::LifLaser)
            laserHwKey = it.key();
    }

    p_lcw = new LifControlWidget(digitizerHwKey, laserHwKey);

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
