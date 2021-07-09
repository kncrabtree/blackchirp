#ifndef WIZARDRFCONFIGPAGE_H
#define WIZARDRFCONFIGPAGE_H

#include <gui/wizard/experimentwizardpage.h>

#include <gui/widget/rfconfigwidget.h>

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
    virtual void initializePage();
    virtual bool validatePage();
    virtual int nextId() const;
    virtual bool isComplete() const;
};

#endif // WIZARDRFCONFIGPAGE_H
