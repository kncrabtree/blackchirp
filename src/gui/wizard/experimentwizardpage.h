#ifndef EXPERIMENTWIZARDPAGE_H
#define EXPERIMENTWIZARDPAGE_H

#include <QWizardPage>
#include "experimentwizard.h"
#include "experiment.h"

class ExperimentWizardPage : public QWizardPage
{
    Q_OBJECT
public:
    ExperimentWizardPage(QWidget *parent = nullptr);

signals:
    void experimentUpdate(const Experiment);

protected:
    Experiment getExperiment() const;
    int startingFtmwPage() const;
};

#endif // EXPERIMENTWIZARDPAGE_H
