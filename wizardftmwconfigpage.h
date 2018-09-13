#ifndef WIZARDFTMWCONFIGPAGE_H
#define WIZARDFTMWCONFIGPAGE_H

#include "experimentwizardpage.h"

class FtmwConfig;
class FtmwConfigWidget;

class WizardFtmwConfigPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardFtmwConfigPage(QWidget *parent = 0);
    ~WizardFtmwConfigPage();

    // QWizardPage interface
public:
    void initializePage();
    bool validatePage();
    int nextId() const;

private:
    FtmwConfigWidget *p_ftc;
};

#endif // WIZARDFTMWCONFIGPAGE_H
