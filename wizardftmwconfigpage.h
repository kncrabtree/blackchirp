#ifndef WIZARDFTMWCONFIGPAGE_H
#define WIZARDFTMWCONFIGPAGE_H

#include <QWizardPage>
#include "ftmwconfigwidget.h"

class WizardFtmwConfigPage : public QWizardPage
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

    FtmwConfig getFtmwConfig() const;

private:
    FtmwConfigWidget *p_ftc;
};

#endif // WIZARDFTMWCONFIGPAGE_H
