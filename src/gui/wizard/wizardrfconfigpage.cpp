#include "wizardrfconfigpage.h"

#include <QVBoxLayout>

WizardRfConfigPage::WizardRfConfigPage(QWidget *parent) : ExperimentWizardPage(BC::Key::WizRf::key,parent)
{
    setTitle(QString("Configure Clocks"));
    setSubTitle(QString("Configure the clock setup. If the experiment involves multiple clock frequencies (e.g., an LO scan), the frequencies will be set automatically."));

    p_rfc = new RfConfigWidget(this);
    connect(p_rfc,&RfConfigWidget::edited,this,&QWizardPage::completeChanged);

    auto vbl = new QVBoxLayout(this);
    vbl->addWidget(p_rfc);

    setLayout(vbl);
}



void WizardRfConfigPage::initializePage()
{
    auto e = getExperiment();
    if(e->d_number > 0)
        p_rfc->setFromRfConfig(e->ftmwConfig()->d_rfConfig);
}

bool WizardRfConfigPage::validatePage()
{
    //Ensure clocks are set for scan type
    auto e = getExperiment();
    p_rfc->toRfConfig(e->ftmwConfig()->d_rfConfig);
    return true;
}

int WizardRfConfigPage::nextId() const
{
    return startingFtmwPage();
}

bool WizardRfConfigPage::isComplete() const
{
    auto e = getExperiment();
    if(e->ftmwConfig()->d_type == FtmwConfig::LO_Scan)
    {
        if(p_rfc->getHwKey(RfConfig::UpLO).isEmpty())
            return false;

        if(!p_rfc->commonLO())
        {
            if(p_rfc->getHwKey(RfConfig::DownLO).isEmpty())
                return false;
        }
    }
    else if(e->ftmwConfig()->d_type == FtmwConfig::DR_Scan)
    {
        if(p_rfc->getHwKey(RfConfig::DRClock).isEmpty())
            return false;
    }

    return true;
}
