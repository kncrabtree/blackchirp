#ifndef WIZARDSTARTPAGE_H
#define WIZARDSTARTPAGE_H

#include <QWizardPage>

class QCheckBox;
class QSpinBox;

class WizardStartPage : public QWizardPage
{
    Q_OBJECT
public:
    WizardStartPage(QWidget *parent = 0);
    ~WizardStartPage();

    // QWizardPage interface
    int nextId() const;
    bool isComplete() const;
    void initializePage();

    bool ftmwEnabled() const;
    bool lifEnabled() const;
    bool motorEnabled() const;
    int auxDataInterval() const;
    int snapshotInterval() const;

private:
    QCheckBox *p_ftmw;
#ifdef BC_LIF
    QCheckBox *p_lif;
#endif
#ifdef BC_MOTOR
    QCheckBox *p_motor;
#endif
    QSpinBox *p_auxDataIntervalBox, *p_snapshotBox;
};

#endif // WIZARDSTARTPAGE_H
