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
    setSubTitle(QString(
R"000(Set up conditions that will automatically abort the experiment if a measured value falls outside a certain range. Each validation condition is identified by 2 keys: one referring to the object that generates the data (e.g., FlowController), and the other to the specific value (e.g., flow1).)000"));

    QVBoxLayout *vl = new QVBoxLayout;

    p_validationView = new QTableView();
    ValidationModel *valmodel = new ValidationModel(p_validationView);
    p_validationView->setModel(valmodel);
    p_validationView->setItemDelegate(new ValidationDelegate(valmodel));
    p_validationView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    p_validationView->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
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

void WizardValidationPage::setValidationKeys(std::map<QString, QStringList> m)
{
    d_validationKeys = m;
    static_cast<ValidationModel*>(p_validationView->model())->d_validationKeys = m;
}



int WizardValidationPage::nextId() const
{
    return ExperimentWizard::SummaryPage;
}


void WizardValidationPage::initializePage()
{
    auto model = p_validationView->model();
    for(int i=model->rowCount()-1; i>=0; --i)
    {
        auto ok = model->data(model->index(i,0)).toString();
        auto vk = model->data(model->index(i,1)).toString();

        auto it = d_validationKeys.find(ok);
        if(it == d_validationKeys.end())
        {
            model->removeRow(i);
            continue;
        }

        if(!it->second.contains(vk))
            model->removeRow(i);
    }

}

bool WizardValidationPage::validatePage()
{
    auto e = getExperiment();

    auto model = p_validationView->model();
    ExperimentValidator::ValidationMap m;
    for(int i=0; i<model->rowCount(); ++i)
    {
        auto k1 = model->data(model->index(i,0),Qt::DisplayRole).toString();
        auto k2 = model->data(model->index(i,1),Qt::DisplayRole).toString();
        auto min = model->data(model->index(i,2),Qt::DisplayRole).toDouble();
        auto max = model->data(model->index(i,3),Qt::DisplayRole).toDouble();

        if(k1.isEmpty() || k2.isEmpty())
            continue;

        if(max < min)
            qSwap(min,max);

        auto it = m.find(k1);
        if(it == m.end())
            m.insert({k1,{{k2,{min,max}}}});
        else
            it->second.insert({k2,{min,max}});
    }

    e->setValidationMap(m);
    
    return true;
}
