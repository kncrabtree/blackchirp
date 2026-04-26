#include "experimentdrscanconfigpage.h"

#include <QVBoxLayout>

ExperimentDRScanConfigPage::ExperimentDRScanConfigPage(Experiment *exp, QWidget *parent)
    : ExperimentConfigPage(BC::Key::WizDR::key, BC::Key::WizDR::title, exp, parent)
{
    p_widget = new DRScanConfigWidget(exp, this);
    auto *l = new QVBoxLayout;
    l->setContentsMargins(0, 0, 0, 0);
    l->addWidget(p_widget);
    setLayout(l);
    connect(p_widget, &DRScanConfigWidget::error, this, &ExperimentDRScanConfigPage::error);
    connect(p_widget, &DRScanConfigWidget::warning, this, &ExperimentDRScanConfigPage::warning);
}

void ExperimentDRScanConfigPage::initialize() { p_widget->initialize(); }
bool ExperimentDRScanConfigPage::validate() { return p_widget->validate(); }
void ExperimentDRScanConfigPage::apply() { p_widget->apply(); }
