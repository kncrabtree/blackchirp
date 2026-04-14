#include "experimentchirpconfigpage.h"

using namespace BC::Key::WizChirp;

#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QSpinBox>

#include <gui/widget/chirpconfigwidget.h>
#include <hardware/optional/chirpsource/awg.h>
#include <hardware/core/runtimehardwareconfig.h>

ExperimentChirpConfigPage::ExperimentChirpConfigPage(Experiment *exp, QWidget *parent) :
    ExperimentConfigPage(key,title,exp,parent)
{
    p_ccw = new ChirpConfigWidget(this);
    QVBoxLayout *vbl = new QVBoxLayout(this);
    vbl->addWidget(p_ccw);

    setLayout(vbl);

    if(exp->ftmwConfig())
    {
        if(exp->d_number > 0)
            p_ccw->setFromRfConfig(exp->ftmwConfig()->d_rfConfig);
        else
            p_ccw->initialize(exp->ftmwConfig()->d_rfConfig);
    }
}


void ExperimentChirpConfigPage::initialize()
{
    if(isEnabled() && p_exp->ftmwConfig())
        p_ccw->initialize(p_exp->ftmwConfig()->d_rfConfig);

    p_ccw->updateChirpPlot();
}

bool ExperimentChirpConfigPage::validate()
{
    if(!isEnabled() || !p_exp->ftmwConfig())
        return true;

    auto l = p_ccw->getChirps().chirpList();

    if(l.isEmpty())
    {
        emit error("No chirp configured.");
        return false;
    }

    auto out = true;
    for(int i=0; i<l.size(); i++)
    {
        if(l.at(i).isEmpty())
        {
            emit error(QString("Chirp %1 is not configured").arg(i+1));
            out = false;
        }
    }

    // Safety: warn if no protection marker is active
    int mCount = 0;
    auto awgKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<AWG>();
    if(!awgKeys.isEmpty())
    {
        SettingsStorage awg(awgKeys.first(), SettingsStorage::Hardware);
        mCount = awg.get(BC::Key::AWG::markerCount, 0);
    }
    if(mCount > 0)
    {
        const auto &cc = p_ccw->getChirps();
        const auto *prot = cc.findEnabledMarkerByRole(MarkerRole::Protection);
        const auto *gate = cc.findEnabledMarkerByRole(MarkerRole::Gate);

        if(!prot)
        {
            // Check whether a protection marker exists but is disabled
            bool hasProt = false;
            for(const auto &m : cc.markerChannels())
            {
                if(m.role == MarkerRole::Protection) { hasProt = true; break; }
            }
            if(!hasProt)
                emit warning("No protection marker is configured. Consider adding a Protection marker to prevent damage to sensitive hardware.");
            else
                emit warning("Protection marker is disabled while the chirp is active.");
        }
        else
        {
            // Protection exists and is enabled — check timing coverage
            if(prot->startTime >= 0.0)
                emit warning("Protection pulse starts at or after the chirp. startTime should be negative so protection opens before the chirp begins.");

            if(prot->endTime <= 0.0)
                emit warning("Protection pulse ends at or before the chirp. endTime should be positive so protection closes after the chirp ends.");

            // If an amp enable (gate) pulse is also active, protection must fully enclose it
            if(gate)
            {
                if(prot->startTime > gate->startTime)
                    emit warning("Protection pulse starts after the amp enable pulse. Protection must open at or before the amp enable opens.");

                if(prot->endTime < gate->endTime)
                    emit warning("Protection pulse ends before the amp enable pulse. Protection must close at or after the amp enable closes.");
            }
        }
    }

    return out;
}

void ExperimentChirpConfigPage::apply()
{
    if(isEnabled() && p_exp->ftmwConfig())
        p_exp->ftmwConfig()->d_rfConfig.d_chirpConfig = p_ccw->getChirps();
}
