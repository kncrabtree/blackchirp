#include "experimentftmwdigitizerconfigpage.h"

using namespace BC::Key::WizFtDig;

#include <QVBoxLayout>
#include <QMessageBox>

#include <gui/widget/ftmwdigitizerconfigwidget.h>

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
            p_dc->setFromConfig(exp->ftmwConfig()->scopeConfig());
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

    bool ba = p_dc->blockAverageChecked();
    bool mr = p_dc->multiRecordChecked();
    int numAvg = p_dc->numAverages();
    int numRec = p_dc->numRecords();

    if(numChirps > 1)
    {

        if(!ba && !mr)
            emit warning("Number of chirps is >1, but digitizer is not configured for multiple records or block averaging");

        if(ba && (numAvg != numChirps))
            emit warning("Number of chirps does not match number of block averages.");

    }

    if(mr && (numRec != numChirps))
        emit warning("Number of chirps does not match number of digitizer records.");

    if(p_dc->numAnalogChecked() < 1)
    {
        emit error("No FID channel selected.");
        return false;
    }
    else if(p_dc->numAnalogChecked() > 1)
    {
        emit error("Only 1 FID channel may be selected.");
        return false;
    }

    return true;
}

void ExperimentFtmwDigitizerConfigPage::apply()
{
    if(isEnabled() && p_exp->ftmwEnabled())
    {
        p_dc->toConfig(p_exp->ftmwConfig()->scopeConfig());
        p_exp->ftmwConfig()->scopeConfig().d_fidChannel = p_exp->ftmwConfig()->scopeConfig().d_analogChannels.cbegin()->first;
    }
}
