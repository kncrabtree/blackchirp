#ifndef WIZARDLIFCONFIGPAGE_H
#define WIZARDLIFCONFIGPAGE_H

#include "experimentwizardpage.h"

class QDoubleSpinBox;
class QCheckBox;
class LifControlWidget;
class QComboBox;

class WizardLifConfigPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardLifConfigPage(QWidget *parent = nullptr);
    ~WizardLifConfigPage();

    void setFromConfig(const LifConfig c);

    // QWizardPage interface
    void initializePage();
    bool validatePage();
    int nextId() const;

signals:
    void newTrace(const LifTrace c);
    void updateScope(const BlackChirp::LifScopeConfig);
    void scopeConfigChanged(const BlackChirp::LifScopeConfig);
    void lifColorChanged();

private:
    QDoubleSpinBox *p_delayStart, *p_delayStep, *p_delayEnd;
    QDoubleSpinBox *p_laserStart, *p_laserStep, *p_laserEnd;
    QCheckBox *p_delaySingle, *p_laserSingle;
    QComboBox *p_orderBox, *p_completeBox;
    LifControlWidget *p_lifControl;


};

#endif // WIZARDLIFCONFIGPAGE_H
