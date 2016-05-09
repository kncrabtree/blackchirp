#ifndef WIZARDCHIRPCONFIGPAGE_H
#define WIZARDCHIRPCONFIGPAGE_H

#include <QWizardPage>

class ChirpConfigWidget;
class ChirpConfig;

class WizardChirpConfigPage : public QWizardPage
{
    Q_OBJECT
public:
    WizardChirpConfigPage(QWidget *parent = 0);
    ~WizardChirpConfigPage();

    // QWizardPage interface
    int nextId() const;
    bool isComplete() const;

    ChirpConfig getChirpConfig() const;

private:
    ChirpConfigWidget *p_ccw;
};

#endif // WIZARDCHIRPCONFIGPAGE_H
