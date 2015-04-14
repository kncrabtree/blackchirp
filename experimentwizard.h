#ifndef EXPERIMENTWIZARD_H
#define EXPERIMENTWIZARD_H

#include <QWizard>
#include "wizardstartpage.h"
#include "wizardchirpconfigpage.h"
#include "wizardftmwconfigpage.h"
#include "wizardsummarypage.h"
#include "experiment.h"


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
        SummaryPage
    };

    Experiment getExperiment() const;
    void saveToSettings();

private:
    WizardStartPage *p_startPage;
    WizardChirpConfigPage *p_chirpConfigPage;
    WizardFtmwConfigPage *p_ftmwConfigPage;
    WizardSummaryPage *p_summaryPage;
};

#endif // EXPERIMENTWIZARD_H
