#ifndef EXPERIMENTWIZARD_H
#define EXPERIMENTWIZARD_H

#include <QWizard>

#include "wizardstartpage.h"
#include "wizardchirpconfigpage.h"
#include "wizardftmwconfigpage.h"
#include "wizardsummarypage.h"
#include "wizardpulseconfigpage.h"
#include "experiment.h"
#include "batchsingle.h"


class ExperimentWizard : public QWizard
{
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

private:
    WizardStartPage *p_startPage;
    WizardChirpConfigPage *p_chirpConfigPage;
    WizardFtmwConfigPage *p_ftmwConfigPage;
    WizardSummaryPage *p_summaryPage;
    WizardPulseConfigPage *p_pulseConfigPage;
    FlowConfig d_flowConfig;
};

#endif // EXPERIMENTWIZARD_H
