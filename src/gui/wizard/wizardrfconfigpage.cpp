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
    if(e->d_number > 0)
        p_rfc->setRfConfig(e->ftmwConfig()->d_rfConfig);
}

bool WizardRfConfigPage::validatePage()
{
    //Ensure clocks are set for scan type
    auto rfc = p_rfc->getRfConfig();
    auto e = getExperiment();
    e->ftmwConfig()->d_rfConfig = p_rfc->getRfConfig();
    


    if(e->ftmwConfig()->d_type == FtmwConfig::LO_Scan)
    {
        if(rfc.clockHardware(RfConfig::UpLO).isEmpty())
            return false;

        if(!rfc.d_commonUpDownLO)
        {
            if(rfc.clockHardware(RfConfig::DownLO).isEmpty())
                return false;
        }
    }

    if(e->ftmwConfig()->d_type == FtmwConfig::DR_Scan)
    {
        if(rfc.clockHardware(RfConfig::DRClock).isEmpty())
            return false;
    }

    e->ftmwConfig()->d_rfConfig = getRfConfig();


    return true;
}

int WizardRfConfigPage::nextId() const
{
    return startingFtmwPage();
}
