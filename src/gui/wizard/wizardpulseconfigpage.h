#ifndef WIZARDPULSECONFIGPAGE_H
#define WIZARDPULSECONFIGPAGE_H

#include <gui/wizard/experimentwizardpage.h>

class PulseConfigWidget;

namespace BC::Key::WizPulse {
static const QString key{"WizardPulsePage"};
}

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
