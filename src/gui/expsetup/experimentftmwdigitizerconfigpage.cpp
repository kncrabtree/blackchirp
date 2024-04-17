#include "experimentftmwdigitizerconfigpage.h"

using namespace BC::Key::WizFtDig;

#include <QVBoxLayout>
#include <QMessageBox>

#include <gui/widget/ftmwdigitizerconfigwidget.h>
#include <gui/wizard/experimentwizard.h>

ExperimentFtmwDigitizerConfigPage::ExperimentFtmwDigitizerConfigPage(Experiment *exp, QWidget *parent)
    : ExperimentConfigPage(key,title,exp,parent)
{
    QVBoxLayout *vbl = new QVBoxLayout(this);
    p_dc = new FtmwDigitizerConfigWidget(this);
    p_dc->d_maxAnalogEnabled = 1;

    vbl->addWidget(p_dc);

    setLayout(vbl);

    if(exp->ftmwConfig())
    {
        if(exp->d_number > 0)
            p_dc->setFromConfig(exp->ftmwConfig()->d_scopeConfig);
    }
}


void ExperimentFtmwDigitizerConfigPage::initialize()
{
}

bool ExperimentFtmwDigitizerConfigPage::validate()
{
    if(!isEnabled() || !p_exp->ftmwEnabled())
        return true;

    int numChirps = p_exp->ftmwConfig()->d_rfConfig.d_chirpConfig.numChirps();

    if(numChirps > 1)
    {
        bool ba = p_dc->blockAverageChecked();
        bool mr = p_dc->multiRecordChecked();
        int numAvg = p_dc->numAverages();
        int numRec = p_dc->numRecords();

        if(!ba && !mr)
            emit warning("Number of chirps is >1, but digitizer is not configured for multiple records or block averaging");

        if(mr && (numRec != numChirps))
            emit warning("Number of chirps does not match number of digitizer records.");

        if(ba && (numAvg != numChirps))
            emit warning("Number of chirps does not match number of block averages.");

    }

    if(p_dc->numAnalogChecked() != 1)
    {
        emit error("No FID channel selected.");
        return false;
    }

    return true;
}

void ExperimentFtmwDigitizerConfigPage::apply()
{
    p_dc->toConfig(p_exp->ftmwConfig()->d_scopeConfig);
    p_exp->ftmwConfig()->d_scopeConfig.d_fidChannel = p_exp->ftmwConfig()->d_scopeConfig.d_analogChannels.cbegin()->first;
}
