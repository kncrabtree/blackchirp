#ifndef WIZARDRFCONFIGPAGE_H
#define WIZARDRFCONFIGPAGE_H

#include "experimentwizardpage.h"

#include "rfconfigwidget.h"

class WizardRfConfigPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardRfConfigPage(QWidget *parent = nullptr);

private:
    RfConfigWidget *p_rfc;

    // QWizardPage interface
public:
    RfConfig getRfConfig() { return p_rfc->getRfConfig(); }
    virtual void initializePage();
    virtual bool validatePage();
    virtual int nextId() const;
};

#endif // WIZARDRFCONFIGPAGE_H
