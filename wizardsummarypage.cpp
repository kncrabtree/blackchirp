#include "wizardsummarypage.h"

#include <QVBoxLayout>
#include <QPlainTextEdit>

#include "experimentwizard.h"


WizardSummaryPage::WizardSummaryPage(QWidget *parent) :
    QWizardPage(parent)
{
    setTitle(QString("Experiment Summary"));
    setSubTitle(QString("The settings shown below will be used for this experiment. If anything is incorrect, use the back button to make changes."));

    QVBoxLayout *vbl = new QVBoxLayout(this);
    p_pte = new QPlainTextEdit(this);
    p_pte->setReadOnly(true);

    vbl->addWidget(p_pte);

}

WizardSummaryPage::~WizardSummaryPage()
{
}



void WizardSummaryPage::initializePage()
{
    p_pte->clear();

    ExperimentWizard *w = static_cast<ExperimentWizard*>(wizard());
    Experiment e = w->getExperiment();

    auto header = e.headerMap();
    auto it = header.constBegin();
    while(it != header.constEnd())
    {
        p_pte->appendPlainText(QString("%1\t%2\t%3\n").arg(it.key()).arg(it.value().first.toString()).arg(it.value().second));
        it++;
    }
    p_pte->moveCursor(QTextCursor::Start);
    p_pte->ensureCursorVisible();
}

int WizardSummaryPage::nextId() const
{
    return -1;
}
