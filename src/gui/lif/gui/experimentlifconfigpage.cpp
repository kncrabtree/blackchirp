#include "experimentlifconfigpage.h"

#include <QTabWidget>
#include <QVBoxLayout>

#include <gui/lif/gui/lifcontrolwidget.h>
#include <gui/lif/gui/lifconversionwidget.h>
#include <data/experiment/hardwaredatacontainer.h>

using namespace BC::Key::WizLif;
using namespace Qt::StringLiterals;

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
    p_conversionWidget = new LifConversionWidget(false, this);

    auto tabs = new QTabWidget(this);
    tabs->addTab(p_lcw, "Acquisition"_L1);
    tabs->addTab(p_conversionWidget, "Conversion"_L1);

    auto vbl = new QVBoxLayout;
    vbl->addWidget(tabs);

    setLayout(vbl);

    // For an existing experiment, the conversion table shows the topology
    // actually recorded for it (loaded from liftopology.csv on disk, or
    // seeded from the current LIF preset when Experiment::enableLif() ran
    // for a repeat/new experiment earlier in the wizard flow). A genuinely
    // new experiment has no LifConfig yet at page-construction time (this
    // page is built before ExperimentTypePage::apply() first runs
    // enableLif()); LifConversionWidget's own constructor already seeds
    // itself from the current LIF preset in that case, exactly mirroring
    // FtmwConfigWidget/ExperimentFtmwConfigPage.
    if(p_exp->d_number > 0 && p_exp->lifEnabled())
    {
        p_lcw->setFromConfig(*p_exp->lifConfig());
        p_conversionWidget->setFromConfig(*p_exp->lifConfig());
    }

    connect(p_conversionWidget, &LifConversionWidget::edited,
            this, &ExperimentLifConfigPage::presetChanged);
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
    {
        p_lcw->toConfig(*p_exp->lifConfig());
        p_conversionWidget->toConfig(*p_exp->lifConfig());
    }
}
