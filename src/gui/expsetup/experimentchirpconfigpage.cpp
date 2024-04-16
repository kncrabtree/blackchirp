#include "experimentchirpconfigpage.h"

using namespace BC::Key::WizChirp;

#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QSpinBox>

#include <gui/widget/chirpconfigwidget.h>

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

    return out;
}

void ExperimentChirpConfigPage::apply()
{
    if(isEnabled() && p_exp->ftmwConfig())
        p_exp->ftmwConfig()->d_rfConfig.d_chirpConfig = p_ccw->getChirps();
}
