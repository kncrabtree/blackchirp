#ifndef EXPERIMENTWIZARD_H
#define EXPERIMENTWIZARD_H

#include <QWizard>

class WizardStartPage;
class WizardChirpConfigPage;
class WizardFtmwConfigPage;
class WizardSummaryPage;
class WizardPulseConfigPage;
class WizardLifConfigPage;
class WizardValidationPage;
class WizardMotorScanConfigPage;
class BatchManager;

#include "experiment.h"


class ExperimentWizard : public QWizard
{
    Q_OBJECT
public:
    ExperimentWizard(QWidget *parent = 0);
    ~ExperimentWizard();

    enum Page {
        StartPage,
        ChirpConfigPage,
        FtmwConfigPage,
#ifdef BC_LIF
        LifConfigPage,
#endif
#ifdef BC_MOTOR
        MotorScanConfigPage,
#endif
        PulseConfigPage,
        ValidationPage,
        SummaryPage
    };

    void setPulseConfig(const PulseGenConfig c);
    void setFlowConfig(const FlowConfig c);
    Experiment getExperiment() const;
    bool sleepWhenDone() const;


private:
    WizardStartPage *p_startPage;
    WizardChirpConfigPage *p_chirpConfigPage;
    WizardFtmwConfigPage *p_ftmwConfigPage;
    WizardSummaryPage *p_summaryPage;
    WizardPulseConfigPage *p_pulseConfigPage;
    WizardValidationPage *p_validationPage;

    FlowConfig d_flowConfig;

#ifdef BC_LIF
signals:
    void newTrace(const LifTrace);
    void updateScope(const BlackChirp::LifScopeConfig);
    void scopeConfigChanged(const BlackChirp::LifScopeConfig);
    void lifColorChanged();

private:
    WizardLifConfigPage *p_lifConfigPage;
#endif

#ifdef BC_MOTOR
private:
    WizardMotorScanConfigPage *p_motorScanConfigPage;
#endif

};

#endif // EXPERIMENTWIZARD_H
