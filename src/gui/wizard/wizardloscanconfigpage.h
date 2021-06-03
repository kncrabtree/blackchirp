#ifndef WIZARDLOSCANCONFIGPAGE_H
#define WIZARDLOSCANCONFIGPAGE_H

#include "experimentwizardpage.h"

class QSpinBox;
class QDoubleSpinBox;
class QGroupBox;
class QCheckBox;

class WizardLoScanConfigPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardLoScanConfigPage(QWidget *parent = nullptr);

    // QWizardPage interface
public:
    virtual void initializePage();
    virtual bool validatePage();
    virtual bool isComplete() const;
    virtual int nextId() const;

public slots:
    void startChanged(BlackChirp::ClockType t, double val);
    void endChanged(BlackChirp::ClockType t, double val);
    void minorStepChanged(BlackChirp::ClockType t, int val);
    void minorStepSizeChanged(BlackChirp::ClockType t, double val);
    void majorStepChanged(BlackChirp::ClockType t, int val);
    void majorStepSizeChanged(BlackChirp::ClockType t, double val);
    void fixedChanged(bool fixed);
    void constantOffsetChanged(bool co);

protected:
    void saveToSettings() const;
    void loadFromSettings();

private:
    QSpinBox *p_upNumMinorBox, *p_downNumMinorBox, *p_upNumMajorBox, *p_downNumMajorBox, *p_shotsPerStepBox, *p_targetSweepsBox;
    QDoubleSpinBox *p_upStartBox, *p_downStartBox, *p_upEndBox, *p_downEndBox, *p_upMinorStepBox, *p_downMinorStepBox, *p_upMajorStepBox, *p_downMajorStepBox;
    QGroupBox *p_upBox, *p_downBox;
    QCheckBox *p_fixedDownLoBox, *p_constantDownOffsetBox;

    RfConfig d_rfConfig;

    double calculateMajorStepSize(BlackChirp::ClockType t);
};

#endif // WIZARDLOSCANCONFIGPAGE_H
