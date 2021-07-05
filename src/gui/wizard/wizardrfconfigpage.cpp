#include "wizardrfconfigpage.h"

#include <QVBoxLayout>

WizardRfConfigPage::WizardRfConfigPage(QWidget *parent) : ExperimentWizardPage(BC::Key::WizRf::key,parent)
{
    setTitle(QString("Configure Clocks"));
    setSubTitle(QString("Configure the clock setup. If the experiment involves multiple clock frequencies (e.g., an LO scan), the frequencies will be set automatically."));

    p_rfc = new RfConfigWidget(this);

    auto vbl = new QVBoxLayout(this);
    vbl->addWidget(p_rfc);

    setLayout(vbl);
}



void WizardRfConfigPage::initializePage()
{
    auto e = getExperiment();
    p_rfc->setRfConfig(e->d_ftmwCfg.d_rfConfig);
}

bool WizardRfConfigPage::validatePage()
{
    //Ensure clocks are set for scan type
    auto rfc = p_rfc->getRfConfig();
    auto e = getExperiment();
    e->d_ftmwCfg.d_rfConfig = p_rfc->getRfConfig();
    


    if(e->d_ftmwCfg.type() == FtmwConfig::LO_Scan)
    {
        if(rfc.clockHardware(RfConfig::UpLO).isEmpty())
            return false;

        if(!rfc.d_commonUpDownLO)
        {
            if(rfc.clockHardware(RfConfig::DownLO).isEmpty())
                return false;
        }
    }

    if(e->d_ftmwCfg.type() == FtmwConfig::DR_Scan)
    {
        if(rfc.clockHardware(RfConfig::DRClock).isEmpty())
            return false;
    }

    e->setRfConfig(p_rfc->getRfConfig());


    return true;
}

int WizardRfConfigPage::nextId() const
{
    return startingFtmwPage();
}
