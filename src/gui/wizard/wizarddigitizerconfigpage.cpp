#include "wizarddigitizerconfigpage.h"

#include <QVBoxLayout>
#include <QMessageBox>

#include <gui/widget/ftmwdigitizerconfigwidget.h>
#include <gui/wizard/experimentwizard.h>

WizardDigitizerConfigPage::WizardDigitizerConfigPage(QWidget *parent) :
    ExperimentWizardPage(BC::Key::WizFtDig::key,parent)
{
    setTitle(QString("Configure FTMW Digitizer"));
    setSubTitle("Only one analog channel may be selected; this should be the channel to which the spectrometer output is connected.");

    QVBoxLayout *vbl = new QVBoxLayout(this);
    p_dc = new FtmwDigitizerConfigWidget(this);
    p_dc->d_maxAnalogEnabled = 1;

    connect(p_dc,&FtmwDigitizerConfigWidget::edited,this,&QWizardPage::completeChanged);

    vbl->addWidget(p_dc);

    setLayout(vbl);
}

WizardDigitizerConfigPage::~WizardDigitizerConfigPage()
{
}



void WizardDigitizerConfigPage::initializePage()
{
    ///TODO: Be more flexible here
    auto e = getExperiment();
    if(e->d_number > 0)
        p_dc->setFromConfig(e->ftmwConfig()->d_scopeConfig);

    int numChirps = e->ftmwConfig()->d_rfConfig.d_chirpConfig.numChirps();
    if(numChirps > 1)
        p_dc->configureForChirp(numChirps);

}

bool WizardDigitizerConfigPage::validatePage()
{
    auto e = getExperiment();
    int numChirps = e->ftmwConfig()->d_rfConfig.d_chirpConfig.numChirps();

    if(p_dc->numAnalogChecked() != 1)
        return false;

    p_dc->toConfig(e->ftmwConfig()->d_scopeConfig);
    e->ftmwConfig()->d_scopeConfig.d_fidChannel = e->ftmwConfig()->d_scopeConfig.d_analogChannels.cbegin()->first;
    if(numChirps > 1)
    {
        if(!e->ftmwConfig()->d_scopeConfig.d_blockAverage && !e->ftmwConfig()->d_scopeConfig.d_multiRecord)
        {
            QMessageBox msg;
            msg.setIcon(QMessageBox::Warning);
            msg.setText("Your configuration may be invalid.");
            msg.setInformativeText("Your AWG is generating multiple chirps, but the digitizer is not configured for block averaging or multiple records. Do you wish to proceed anyways?");
            msg.setStandardButtons(QMessageBox::Yes|QMessageBox::No);
            msg.setDefaultButton(QMessageBox::No);
            auto ret = msg.exec();
            if(ret == QMessageBox::Yes)
                return true;
            else
                return false;
        }
    }
    
    return true;
}

int WizardDigitizerConfigPage::nextId() const
{
    return static_cast<ExperimentWizard*>(wizard())->nextOptionalPage();
}

bool WizardDigitizerConfigPage::isComplete() const
{
    return p_dc->numAnalogChecked() == 1;
}
