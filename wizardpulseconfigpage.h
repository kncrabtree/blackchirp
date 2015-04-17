#ifndef WIZARDPULSECONFIGPAGE_H
#define WIZARDPULSECONFIGPAGE_H

#include <QWizardPage>
#include "pulseconfigwidget.h"

class WizardPulseConfigPage : public QWizardPage
{
    Q_OBJECT
public:
    WizardPulseConfigPage(QWidget *parent = 0);
    ~WizardPulseConfigPage();

    void setConfig(const PulseGenConfig c);
    PulseGenConfig getConfig() const;

    // QWizardPage interface
    void initializePage();
    int nextId() const;
    bool validatePage();

private:
    PulseConfigWidget *p_pcw;

};

#endif // WIZARDPULSECONFIGPAGE_H
