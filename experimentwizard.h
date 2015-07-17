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
        LifConfigPage,
        PulseConfigPage,
        ValidationPage,
        SummaryPage
    };

    void setPulseConfig(const PulseGenConfig c);
    void setFlowConfig(const FlowConfig c);
    Experiment getExperiment() const;
    BatchManager *getBatchManager() const;

signals:
    void newTrace(const LifTrace);
    void updateScope(const BlackChirp::LifScopeConfig);
    void scopeConfigChanged(const BlackChirp::LifScopeConfig);
    void lifColorChanged();

private:
    WizardStartPage *p_startPage;
    WizardChirpConfigPage *p_chirpConfigPage;
    WizardFtmwConfigPage *p_ftmwConfigPage;
    WizardSummaryPage *p_summaryPage;
    WizardPulseConfigPage *p_pulseConfigPage;
    WizardLifConfigPage *p_lifConfigPage;
    WizardValidationPage *p_validationPage;

    FlowConfig d_flowConfig;

    void saveToSettings() const;
};

#endif // EXPERIMENTWIZARD_H
