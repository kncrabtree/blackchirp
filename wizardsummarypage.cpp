#include "wizardsummarypage.h"

#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>

#include "experimentwizard.h"


WizardSummaryPage::WizardSummaryPage(QWidget *parent) :
    QWizardPage(parent)
{
    setTitle(QString("Experiment Summary"));
    setSubTitle(QString("The settings shown below will be used for this experiment. If anything is incorrect, use the back button to make changes."));

    QVBoxLayout *vbl = new QVBoxLayout(this);
    p_tw = new QTableWidget(this);
    p_tw->setColumnCount(3);
    p_tw->setEditTriggers(QTableWidget::NoEditTriggers);
    p_tw->setSelectionBehavior(QAbstractItemView::SelectRows);

    p_tw->setHorizontalHeaderItem(0,new QTableWidgetItem(QString("Key")));
    p_tw->setHorizontalHeaderItem(1,new QTableWidgetItem(QString("Value")));
    p_tw->setHorizontalHeaderItem(2,new QTableWidgetItem(QString("Unit")));
    p_tw->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);


    vbl->addWidget(p_tw);
    setLayout(vbl);

}

WizardSummaryPage::~WizardSummaryPage()
{
}



void WizardSummaryPage::initializePage()
{
    p_tw->clearContents();

    ExperimentWizard *w = static_cast<ExperimentWizard*>(wizard());
    Experiment e = w->getExperiment();

    auto header = e.headerMap();
    auto it = header.constBegin();
    p_tw->setRowCount(header.size());
    int i = 0;
    while(it != header.constEnd())
    {
        p_tw->setItem(i,0,new QTableWidgetItem(it.key()));
        p_tw->setItem(i,1,new QTableWidgetItem(it.value().first.toString()));
        p_tw->setItem(i,2,new QTableWidgetItem(it.value().second));

        it++;
        i++;
    }
//    p_pte->moveCursor(QTextCursor::Start);
//    p_pte->ensureCursorVisible();
    p_tw->resizeColumnsToContents();
}

int WizardSummaryPage::nextId() const
{
    return -1;
}
