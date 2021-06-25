#ifndef EXPERIMENTWIZARDPAGE_H
#define EXPERIMENTWIZARDPAGE_H

#include <QWizardPage>
#include <gui/wizard/experimentwizard.h>
#include <data/experiment/experiment.h>
#include <data/storage/settingsstorage.h>

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
