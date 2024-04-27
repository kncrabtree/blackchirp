#include "experimentrfconfigpage.h"

#include <QVBoxLayout>

#include <gui/widget/rfconfigwidget.h>

using namespace BC::Key::WizRf;

ExperimentRfConfigPage::ExperimentRfConfigPage(Experiment *exp, const QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks, QWidget *parent) :
    ExperimentConfigPage(key,title,exp,parent), d_clocks{clocks}
{
    p_rfc = new RfConfigWidget(this);

    auto vbl = new QVBoxLayout;
    vbl->addWidget(p_rfc);

    setLayout(vbl);

    if(p_exp->d_number > 0 && p_exp->ftmwEnabled())
        p_rfc->setFromRfConfig(p_exp->ftmwConfig()->d_rfConfig);
    else
        p_rfc->setClocks(d_clocks);
}

void ExperimentRfConfigPage::initialize()
{
}

bool ExperimentRfConfigPage::validate()
{
    if(!p_exp->ftmwConfig())
        return true;

    auto out = true;

    if(p_exp->ftmwConfig()->d_type == FtmwConfig::LO_Scan)
    {
        if(p_rfc->getHwKey(RfConfig::UpLO).isEmpty())
        {
            emit error("No upconversion LO set for LO Scan.");
            out = false;
        }

        if(!p_rfc->commonLO())
        {
            if(p_rfc->getHwKey(RfConfig::DownLO).isEmpty())
            {
                emit error("No downconversion LO set for LO Scan.");
                out = false;
            }
        }
    }
    else if(p_exp->ftmwConfig()->d_type == FtmwConfig::DR_Scan)
    {
        if(p_rfc->getHwKey(RfConfig::DRClock).isEmpty())
        {
            emit error("No DR clock set for DR Scan.");
            out = false;
        }
    }
    else
    {
        if(p_rfc->getHwKey(RfConfig::UpLO).isEmpty())
            emit warning("No upconversion LO set; assuming 0 MHz.");

        if(!p_rfc->commonLO())
        {
            if(p_rfc->getHwKey(RfConfig::DownLO).isEmpty())
                emit warning("No downconversion LO set; assuming 0 MHz");
        }
    }

    return out;
}

void ExperimentRfConfigPage::apply()
{
    if(isEnabled() && p_exp->ftmwEnabled())
        p_rfc->toRfConfig(p_exp->ftmwConfig()->d_rfConfig);
}
