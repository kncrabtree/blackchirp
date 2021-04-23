#ifndef WIZARDLIFCONFIGPAGE_H
#define WIZARDLIFCONFIGPAGE_H

#include "experimentwizardpage.h"

class QDoubleSpinBox;
class QSpinBox;
class QCheckBox;
class LifControlWidget;
class LifLaserControlDoubleSpinBox;
class QComboBox;

class WizardLifConfigPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardLifConfigPage(QWidget *parent = nullptr);
    ~WizardLifConfigPage() override;

    void setFromConfig(LifConfig c);
    void setLaserPos(double pos);

    // QWizardPage interface
    void initializePage() override;
    bool validatePage() override;
    int nextId() const override;

signals:
    void newTrace(LifTrace c);
    void updateScope(BlackChirp::LifScopeConfig);
    void scopeConfigChanged(BlackChirp::LifScopeConfig);
    void laserPosUpdate(double);
    void lifColorChanged();

private:
    QDoubleSpinBox *p_delayStart, *p_delayStep;
    LifLaserControlDoubleSpinBox *p_laserStart, *p_laserStep;
    QSpinBox *p_delayNum, *p_laserNum;
    QCheckBox *p_delaySingle, *p_laserSingle;
    QComboBox *p_orderBox, *p_completeBox;
    LifControlWidget *p_lifControl;


};

#endif // WIZARDLIFCONFIGPAGE_H
