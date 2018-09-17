#ifndef WIZARDSTARTPAGE_H
#define WIZARDSTARTPAGE_H

#include "experimentwizardpage.h"

class QGroupBox;
class QSpinBox;
class QDateTimeEdit;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;

class WizardStartPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardStartPage(QWidget *parent = 0);
    ~WizardStartPage();

    // QWizardPage interface
    int nextId() const;
    bool isComplete() const;
    void initializePage();
    bool validatePage();

public slots:
    void configureUI();

private:
    QGroupBox *p_ftmw;
#ifdef BC_LIF
    QGroupBox *p_lif;
#endif
#ifdef BC_MOTOR
    QGroupBox *p_motor;
#endif

    QSpinBox *p_auxDataIntervalBox, *p_snapshotBox, *p_ftmwShotsBox;
    QComboBox *p_ftmwTypeBox;
    QDateTimeEdit *p_ftmwTargetTimeBox;
    QCheckBox *p_phaseCorrectionBox, *p_chirpScoringBox;
    QDoubleSpinBox *p_thresholdBox, *p_chirpOffsetBox;



};

#endif // WIZARDSTARTPAGE_H
