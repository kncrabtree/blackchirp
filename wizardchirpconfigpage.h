#ifndef WIZARDCHIRPCONFIGPAGE_H
#define WIZARDCHIRPCONFIGPAGE_H

#include "experimentwizardpage.h"

class ChirpConfigWidget;
class RfConfig;

class WizardChirpConfigPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardChirpConfigPage(QWidget *parent = 0);
    ~WizardChirpConfigPage();

    // QWizardPage interface
    void initializePage();
    int nextId() const;
    bool validatePage();
    virtual bool isComplete() const;

private:
    ChirpConfigWidget *p_ccw;


};

#endif // WIZARDCHIRPCONFIGPAGE_H
