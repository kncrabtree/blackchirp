#include "wizardrfconfigpage.h"

#include <QVBoxLayout>

WizardRfConfigPage::WizardRfConfigPage(QWidget *parent) : ExperimentWizardPage(parent)
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
    p_rfc->setRfConfig(e.ftmwConfig().rfConfig());
}

bool WizardRfConfigPage::validatePage()
{
    //Ensure clocks are set for scan type
    auto rfc = p_rfc->getRfConfig();
    auto e = getExperiment();

    if(e.ftmwConfig().type() == BlackChirp::FtmwLoScan)
    {
        if(rfc.clockHardware(BlackChirp::UpConversionLO).isEmpty())
            return false;

        if(!rfc.commonLO())
        {
            if(rfc.clockHardware(BlackChirp::DownConversionLO).isEmpty())
                return false;
        }
    }

    if(e.ftmwConfig().type() == BlackChirp::FtmwDrScan)
    {
        if(rfc.clockHardware(BlackChirp::DRClock).isEmpty())
            return false;
    }

    e.setRfConfig(p_rfc->getRfConfig());
    emit experimentUpdate(e);
    return true;
}

int WizardRfConfigPage::nextId() const
{
    return startingFtmwPage();
}
