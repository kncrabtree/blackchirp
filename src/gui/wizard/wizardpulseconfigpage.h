#ifndef WIZARDPULSECONFIGPAGE_H
#define WIZARDPULSECONFIGPAGE_H

#include "experimentwizardpage.h"

class PulseConfigWidget;


class WizardPulseConfigPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardPulseConfigPage(QWidget *parent = 0);
    ~WizardPulseConfigPage();

    // QWizardPage interface
    void initializePage();
    int nextId() const;
    bool validatePage();

private:
    PulseConfigWidget *p_pcw;
    bool d_firstInitialization;

};

#endif // WIZARDPULSECONFIGPAGE_H
