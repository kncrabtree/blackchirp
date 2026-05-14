#include "experimentsummarypage.h"

#include <QVBoxLayout>

#include <gui/widget/experimentsummarywidget.h>

using namespace BC::Key::WizSummary;

ExperimentSummaryPage::ExperimentSummaryPage(Experiment *exp, QWidget *parent)
    : ExperimentConfigPage(key, title, exp, parent)
{
    auto vbl = new QVBoxLayout;
    p_summaryWidget = new ExperimentSummaryWidget(this);
    vbl->addWidget(p_summaryWidget);
    setLayout(vbl);
}

void ExperimentSummaryPage::initialize()
{
    p_summaryWidget->setExperiment(p_exp);
}

bool ExperimentSummaryPage::validate()
{
    return true;
}

void ExperimentSummaryPage::apply()
{
}
