#ifndef EXPERIMENTWIZARDPAGE_H
#define EXPERIMENTWIZARDPAGE_H

#include <QWizardPage>
#include <src/gui/wizard/experimentwizard.h>
#include <src/data/experiment/experiment.h>

class ExperimentWizardPage : public QWizardPage
{
    Q_OBJECT
public:
    ExperimentWizardPage(QWidget *parent = nullptr);

protected:
    Experiment *getExperiment() const;
    int startingFtmwPage() const;
};

#endif // EXPERIMENTWIZARDPAGE_H
