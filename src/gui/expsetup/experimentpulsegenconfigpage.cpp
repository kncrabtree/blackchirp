#include "experimentpulsegenconfigpage.h"

using namespace BC::Key::WizPulse;

#include <QVBoxLayout>

#include <gui/widget/pulseconfigwidget.h>

ExperimentPulseGenConfigPage::ExperimentPulseGenConfigPage(QString hwKey, QString title, Experiment *exp, QWidget *parent)
    : ExperimentConfigPage(key,title,exp,parent)
{
    QVBoxLayout *vbl = new QVBoxLayout();

    auto p = exp->getOptHwConfig<PulseGenConfig>(hwKey);
    p_pcw = new PulseConfigWidget(*p.lock(),this);
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

    auto gasCh = c.channelForRole(PulseGenConfig::Gas);
    if(gasCh >= 0 && !c.d_channels.at(gasCh).enabled)
        emit warning(QString("Gas channel on %1 is disabled.").arg(d_title));

    if(p_exp->ftmwEnabled())
    {
        auto awgCh = c.channelForRole(PulseGenConfig::AWG);
        if(awgCh >= 0 && !c.d_channels.at(awgCh).enabled)
            emit warning(QString("AWG channel on %1 is disabled.").arg(d_title));

        auto ampCh = c.channelForRole(PulseGenConfig::Amp);
        if(ampCh >= 0 && !c.d_channels.at(ampCh).enabled)
            emit warning(QString("Amp channel on %1 is disabled.").arg(d_title));

        auto protCh = c.channelForRole(PulseGenConfig::Prot);
        if(protCh >= 0)
        {
            if(!c.d_channels.at(protCh).enabled)
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
        auto lifCh = c.channelForRole(PulseGenConfig::LIF);
        if(lifCh >= 0 && !c.d_channels.at(lifCh).enabled)
        {
            emit error(QString("LIF channel on %1 is disabled.").arg(d_title));
            out = false;
        }
    }
#endif

    return out;
}

void ExperimentPulseGenConfigPage::apply()
{
    p_exp->addOptHwConfig(p_pcw->getConfig());
}
