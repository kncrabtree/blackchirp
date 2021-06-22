#ifndef WIZARDCHIRPCONFIGPAGE_H
#define WIZARDCHIRPCONFIGPAGE_H

#include <src/gui/wizard/experimentwizardpage.h>

class ChirpConfigWidget;
class RfConfig;

namespace BC::Key::WizChirp {
static const QString key("WizardChirpConfigPage");
}

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
