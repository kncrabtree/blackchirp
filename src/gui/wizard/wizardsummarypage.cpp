#include "wizardsummarypage.h"

#include <QVBoxLayout>

#include <gui/wizard/experimentwizard.h>
#include <gui/widget/experimentsummarywidget.h>


WizardSummaryPage::WizardSummaryPage(QWidget *parent) :
    ExperimentWizardPage(BC::Key::WizSummary::key,parent)
{
    setTitle(QString("Experiment Summary"));
    setSubTitle(QString("The settings shown below will be used for this experiment. If anything is incorrect, use the back button to make changes."));

    QVBoxLayout *vbl = new QVBoxLayout(this);
    p_sw = new ExperimentSummaryWidget(this);


    vbl->addWidget(p_sw);
    setLayout(vbl);

}

WizardSummaryPage::~WizardSummaryPage()
{
}



void WizardSummaryPage::initializePage()
{
    p_sw->setExperiment(getExperiment());
}

int WizardSummaryPage::nextId() const
{
    return -1;
}
