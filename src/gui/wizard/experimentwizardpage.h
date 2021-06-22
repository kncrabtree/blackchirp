#ifndef EXPERIMENTWIZARDPAGE_H
#define EXPERIMENTWIZARDPAGE_H

#include <QWizardPage>
#include <src/gui/wizard/experimentwizard.h>
#include <src/data/experiment/experiment.h>
#include <src/data/storage/settingsstorage.h>

class ExperimentWizardPage : public QWizardPage, public SettingsStorage
{
    Q_OBJECT
public:
    ExperimentWizardPage(const QString key, QWidget *parent = nullptr);

protected:
    Experiment *getExperiment() const;
    int startingFtmwPage() const;
};

#endif // EXPERIMENTWIZARDPAGE_H
