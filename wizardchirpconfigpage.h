#ifndef WIZARDCHIRPCONFIGPAGE_H
#define WIZARDCHIRPCONFIGPAGE_H

#include <QWizardPage>

class ChirpConfigWidget;
class RfConfig;

class WizardChirpConfigPage : public QWizardPage
{
    Q_OBJECT
public:
    WizardChirpConfigPage(QWidget *parent = 0);
    ~WizardChirpConfigPage();

    // QWizardPage interface
    int nextId() const;
    bool isComplete() const;

    RfConfig getRfConfig() const;

private:
    ChirpConfigWidget *p_ccw;
};

#endif // WIZARDCHIRPCONFIGPAGE_H
