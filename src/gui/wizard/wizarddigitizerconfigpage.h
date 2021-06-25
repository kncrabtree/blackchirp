#ifndef WIZARDDIGITIZERCONFIGPAGE_H
#define WIZARDDIGITIZERCONFIGPAGE_H

#include <gui/wizard/experimentwizardpage.h>

class DigitizerConfigWidget;

namespace BC::Key::WizFtDig {
static const QString key("WizardFtmwDigitizerPage");
}

class WizardDigitizerConfigPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardDigitizerConfigPage(QWidget *parent = 0);
    ~WizardDigitizerConfigPage();

    // QWizardPage interface
public:
    void initializePage();
    bool validatePage();
    int nextId() const;

private:
    DigitizerConfigWidget *p_dc;
};

#endif // WIZARDDIGITIZERCONFIGPAGE_H
