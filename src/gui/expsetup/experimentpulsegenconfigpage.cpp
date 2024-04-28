#include "experimentpulsegenconfigpage.h"

using namespace BC::Key::WizPulse;

#include <QVBoxLayout>

#include <gui/widget/pulseconfigwidget.h>

ExperimentPulseGenConfigPage::ExperimentPulseGenConfigPage(const QString hwKey, const QString title, Experiment *exp, QWidget *parent)
    : ExperimentConfigPage(key,title,exp,parent)
{
    QVBoxLayout *vbl = new QVBoxLayout();

    auto p = exp->getOptHwConfig<PulseGenConfig>(hwKey);
    p_pcw = new PulseConfigWidget(*p.lock(),this);
    p_pcw->configureForWizard();
    vbl->addWidget(p_pcw);

    setLayout(vbl);
}


void ExperimentPulseGenConfigPage::initialize()
{
}

bool ExperimentPulseGenConfigPage::validate()
{
    if(!isEnabled())
        return true;

    auto &c = p_pcw->getConfig();
    auto out = true;

    if(!c.d_pulseEnabled)
        emit warning(QString("%1 is disabled.").arg(d_title));

    auto gasCh = c.channelsForRole(PulseGenConfig::Gas);
    for(auto ch : gasCh)
    {
        if(!c.d_channels.at(ch).enabled)
            emit warning(QString("Gas channel on %1 is disabled.").arg(d_title));
    }

    if(p_exp->ftmwEnabled())
    {
        auto awgCh = c.channelsForRole(PulseGenConfig::AWG);
        for(auto ch : awgCh)
        {
            if(!c.d_channels.at(ch).enabled)
            emit warning(QString("AWG channel on %1 is disabled.").arg(d_title));
        }

        auto ampCh = c.channelsForRole(PulseGenConfig::Amp);
        for(auto ch : ampCh)
        {
            if(!c.d_channels.at(ch).enabled)
            emit warning(QString("Amp channel on %1 is disabled.").arg(d_title));
        }

        auto protCh = c.channelsForRole(PulseGenConfig::Prot);
        for(auto ch : protCh)
        {
            if(!c.d_channels.at(ch).enabled)
            {
                emit error(QString("Protection channel on %1 is disabled.").arg(d_title));
                out = false;
            }
            else
                emit warning(QString("Ensure timings on %1 protection channel are correct.").arg(d_title));

            if(p_exp->ftmwConfig()->d_rfConfig.d_chirpConfig.numChirps() > 1)
                emit warning(QString("Protection channel on %1 may not protect for all chirps."));
        }
    }

#ifdef BC_LIF
    if(p_exp->lifEnabled())
    {
        auto lifCh = c.channelsForRole(PulseGenConfig::LIF);
        for(auto ch : lifCh)
        {
            if(!c.d_channels.at(ch).enabled)
                emit warning(QString("LIF channel on %1 will be enabled automatically.").arg(d_title));
        }
    }
#endif

    return out;
}

void ExperimentPulseGenConfigPage::apply()
{
    p_exp->addOptHwConfig(p_pcw->getConfig());
}
