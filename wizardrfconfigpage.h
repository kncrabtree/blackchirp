#ifndef WIZARDRFCONFIGPAGE_H
#define WIZARDRFCONFIGPAGE_H

#include <QWizardPage>

#include "rfconfigwidget.h"

class WizardRfConfigPage : public QWizardPage
{
    Q_OBJECT
public:
    WizardRfConfigPage(QWidget *parent = nullptr);

private:
    RfConfigWidget *p_rfc;

    // QWizardPage interface
public:
    virtual void initializePage();
    virtual bool validatePage();
    virtual int nextId() const;
};

#endif // WIZARDRFCONFIGPAGE_H
