#include "wizardsummarypage.h"

#include <QVBoxLayout>
#include <QTreeView>
#include <QHeaderView>
#include <QCheckBox>

#include <gui/wizard/experimentwizard.h>
#include <data/model/exptsummarymodel.h>


WizardSummaryPage::WizardSummaryPage(QWidget *parent) :
    ExperimentWizardPage(BC::Key::WizSummary::key,parent)
{
    setTitle(QString("Experiment Summary"));
    setSubTitle(QString("The settings shown below will be used for this experiment. If anything is incorrect, use the back button to make changes."));

    QVBoxLayout *vbl = new QVBoxLayout(this);
    p_tv = new QTreeView(this);


    vbl->addWidget(p_tv);
    setLayout(vbl);

}

WizardSummaryPage::~WizardSummaryPage()
{
}



void WizardSummaryPage::initializePage()
{
    auto e = getExperiment().get();
    if(p_model)
        p_model->deleteLater();

    p_model = new ExptSummaryModel(e,this);
    p_tv->setModel(p_model);
    p_tv->header()->setSectionResizeMode(QHeaderView::Stretch);
}

int WizardSummaryPage::nextId() const
{
    return -1;
}
