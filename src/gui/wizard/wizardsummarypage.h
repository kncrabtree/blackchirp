#ifndef WIZARDSUMMARYPAGE_H
#define WIZARDSUMMARYPAGE_H

#include <gui/wizard/experimentwizardpage.h>

class QTreeView;
class ExperimentSummaryWidget;

namespace BC::Key::WizSummary {
static const QString key("WizardSummaryPage");
}

class WizardSummaryPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardSummaryPage(QWidget *parent = 0);
    ~WizardSummaryPage();

    // QWizardPage interface
    void initializePage();
    int nextId() const;

private:
    ExperimentSummaryWidget *p_sw;
};

#endif // WIZARDSUMMARYPAGE_H
