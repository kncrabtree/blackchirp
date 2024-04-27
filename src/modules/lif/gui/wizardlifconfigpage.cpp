#include "wizardlifconfigpage.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QSettings>
#include <QApplication>
#include <QGroupBox>
#include <QLabel>
#include <QComboBox>

#include <modules/lif/gui/lifcontrolwidget.h>
#include <modules/lif/gui/liflasercontroldoublespinbox.h>
#include <gui/wizard/experimentwizard.h>

WizardLifConfigPage::WizardLifConfigPage(QWidget *parent) :
    ExperimentWizardPage(BC::Key::WizLif::key,parent)
{
    setTitle(QString("LIF Configuration"));
    setSubTitle(QString("Configure the parameters for the LIF Acquisition."));

    auto *vbl = new QVBoxLayout;

    // p_lifControl = new LifControlWidget(this);
    // vbl->addWidget(p_lifControl,1);

    setLayout(vbl);
}

WizardLifConfigPage::~WizardLifConfigPage()
{
}

void WizardLifConfigPage::initializePage()
{
    // auto e = getExperiment();
    // if(e->lifEnabled() && e->d_number > 0)
        // p_lifControl->setFromConfig(*e->lifConfig());
}

bool WizardLifConfigPage::validatePage()
{
    auto e = getExperiment();
    p_lifControl->toConfig(*e->lifConfig());
    
    return true;

}

int WizardLifConfigPage::nextId() const
{
    auto e = getExperiment();
    if(e->ftmwEnabled())
        return ExperimentWizard::RfConfigPage;

    return static_cast<ExperimentWizard*>(wizard())->nextOptionalPage();
}
