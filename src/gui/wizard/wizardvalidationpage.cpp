#include "wizardvalidationpage.h"

#include <QVBoxLayout>
#include <QLabel>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QCheckBox>
#include <QToolButton>

#include <gui/wizard/experimentwizard.h>
#include <data/model/validationmodel.h>

WizardValidationPage::WizardValidationPage(QWidget *parent) :
    ExperimentWizardPage(BC::Key::WizVal::key,parent)
{
    setTitle(QString("Validation Settings"));
    setSubTitle(QString("Set up conditions that will automatically abort the experiment."));

    QVBoxLayout *vl = new QVBoxLayout;

    p_validationView = new QTableView();
    ValidationModel *valmodel = new ValidationModel(p_validationView);
    p_validationView->setModel(valmodel);
    p_validationView->setItemDelegateForColumn(0,new CompleterLineEditDelegate);
    p_validationView->setItemDelegateForColumn(1,new ValidationDoubleSpinBoxDelegate);
    p_validationView->setItemDelegateForColumn(2,new ValidationDoubleSpinBoxDelegate);
    p_validationView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    p_validationView->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    p_validationView->setMinimumWidth(300);
    vl->addWidget(p_validationView);

    QHBoxLayout *tl = new QHBoxLayout;
    tl->addStretch(1);

    p_addButton = new QToolButton();
    p_addButton->setIcon(QIcon(QString(":/icons/add.png")));
    p_addButton->setText(QString(""));
    tl->addWidget(p_addButton,0);

    p_removeButton = new QToolButton();
    p_removeButton->setIcon(QIcon(QString(":/icons/remove.png")));
    p_removeButton->setText(QString(""));
    tl->addWidget(p_removeButton,0);
    tl->addStretch(1);
    vl->addLayout(tl);

    setLayout(vl);

    connect(p_addButton,&QToolButton::clicked,[=](){ valmodel->addNewItem(); });
    connect(p_removeButton,&QToolButton::clicked,[=](){
        QModelIndexList l = p_validationView->selectionModel()->selectedIndexes();
        if(l.isEmpty())
            return;

        QList<int> rowList;
        for(int i=0; i<l.size(); i++)
        {
            if(!rowList.contains(l.at(i).row()))
                rowList.append(l.at(i).row());
        }

        std::stable_sort(rowList.begin(),rowList.end());

        for(int i=rowList.size()-1; i>=0; i--)
            valmodel->removeRows(rowList.at(i),1,QModelIndex());
    });
}



int WizardValidationPage::nextId() const
{
    return ExperimentWizard::SummaryPage;
}

QMap<QString, BlackChirp::ValidationItem> WizardValidationPage::getValidation() const
{
    auto l = static_cast<ValidationModel*>(p_validationView->model())->getList();
    QMap<QString,BlackChirp::ValidationItem> out;
    for(int i=0; i<l.size(); i++)
        out.insert(l.at(i).key,l.at(i));
    return out;
}


void WizardValidationPage::initializePage()
{
    auto e = getExperiment();

    dynamic_cast<ValidationModel*>(p_validationView->model())->setFromMap(e->validationItems());

    p_validationView->resizeColumnsToContents();

}

bool WizardValidationPage::validatePage()
{
    auto e = getExperiment();

    auto l = static_cast<ValidationModel*>(p_validationView->model())->getList();
    e->setValidationItems(QMap<QString, BlackChirp::ValidationItem>());
    for(int i=0; i<l.size(); i++)
        e->addValidationItem(l.at(i));

    
    return true;
}
