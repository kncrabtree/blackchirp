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
    int auxDataInterval() const;
    int snapshotInterval() const;

    void saveToSettings() const;

private:
    QCheckBox *p_ftmw, *p_lif;
    QSpinBox *p_auxDataIntervalBox, *p_snapshotBox;
};

#endif // WIZARDSTARTPAGE_H
