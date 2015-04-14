#ifndef WIZARDSTARTPAGE_H
#define WIZARDSTARTPAGE_H

#include <QWizardPage>
#include <QCheckBox>

class WizardStartPage : public QWizardPage
{
    Q_OBJECT
public:
    WizardStartPage(QWidget *parent = 0);
    ~WizardStartPage();

    // QWizardPage interface
    int nextId() const;
    bool isComplete() const;

    bool ftmwEnabled() const;
    bool lifEnabled() const;

private:
    QCheckBox *p_ftmw, *p_lif;
};

#endif // WIZARDSTARTPAGE_H
