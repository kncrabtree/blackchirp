#ifndef WIZARDRFCONFIGPAGE_H
#define WIZARDRFCONFIGPAGE_H

#include <src/gui/wizard/experimentwizardpage.h>

#include <src/gui/widget/rfconfigwidget.h>

namespace BC::Key::WizRf {
static const QString key("WizardRfConfigPage");
}

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
