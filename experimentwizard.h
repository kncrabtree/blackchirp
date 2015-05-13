#ifndef EXPERIMENTWIZARD_H
#define EXPERIMENTWIZARD_H

#include <QWizard>

class WizardStartPage;
class WizardChirpConfigPage;
class WizardFtmwConfigPage;
class WizardSummaryPage;
class WizardPulseConfigPage;
class WizardLifConfigPage;
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
        SummaryPage
    };

    void setPulseConfig(const PulseGenConfig c);
    void setFlowConfig(const FlowConfig c);
    Experiment getExperiment() const;
    BatchManager *getBatchManager() const;
    void saveToSettings();

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
    FlowConfig d_flowConfig;
};

#endif // EXPERIMENTWIZARD_H
