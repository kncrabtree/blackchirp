#include "experimentvalidatorconfigpage.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTableView>
#include <QToolButton>
#include <QHeaderView>
#include <data/model/validationmodel.h>

using namespace BC::Key::WizardVal;

ExperimentValidatorConfigPage::ExperimentValidatorConfigPage(Experiment *exp, const std::map<QString, QStringList> &valKeys, QWidget *parent) :
    ExperimentConfigPage(key,title,exp,parent)
{
    QVBoxLayout *vl = new QVBoxLayout;

    p_validationView = new QTableView();
    ValidationModel *valModel = new ValidationModel(p_validationView);
    valModel->d_validationKeys = valKeys;
    p_validationView->setModel(valModel);
    p_validationView->setToolTip(QString(
                                     R"000(Set up conditions that will automatically abort the experiment if a measured value falls outside a certain range.
Each validation condition is identified by 2 keys: one referring to the object that generates the data (e.g., FlowController),
and the other to the specific value (e.g., flow1).)000"));
    p_validationView->setItemDelegate(new ValidationDelegate(valModel));
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

    connect(p_addButton,&QToolButton::clicked,[=](){ valModel->addNewItem(); });
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
            valModel->removeRows(rowList.at(i),1,QModelIndex());
    });

    for(int i=valModel->rowCount()-1; i>=0; --i)
    {
        auto ok = valModel->data(valModel->index(i,0),Qt::DisplayRole).toString();
        auto vk = valModel->data(valModel->index(i,1),Qt::DisplayRole).toString();

        auto it = valKeys.find(ok);
        if(it == valKeys.end())
        {
            valModel->removeRow(i);
            continue;
        }

        if(!it->second.contains(vk))
            valModel->removeRow(i);
    }
}


void ExperimentValidatorConfigPage::initialize()
{
}

bool ExperimentValidatorConfigPage::validate()
{
    auto model = p_validationView->model();
    ExperimentValidator::ValidationMap m;
    for(int i=0; i<model->rowCount(); ++i)
    {
        auto k1 = model->data(model->index(i,0),Qt::DisplayRole).toString();
        auto k2 = model->data(model->index(i,1),Qt::DisplayRole).toString();
        auto min = model->data(model->index(i,2),Qt::DisplayRole).toDouble();
        auto max = model->data(model->index(i,3),Qt::DisplayRole).toDouble();

        if(k1.isEmpty() || k2.isEmpty())
        {
            emit error(QString("Validation setting %1 is missing a key").arg(i+1));
            return false;
        }

        if(max < min)
        {
            emit warning(QString("Min and max for validation setting %1 will be swapped.").arg(i+1));
            qSwap(min,max);
        }

        auto it = m.find(k1);
        if(it == m.end())
            m.insert({k1,{{k2,{min,max}}}});
        else
            it->second.insert({k2,{min,max}});
    }

    return true;
}

void ExperimentValidatorConfigPage::apply()
{
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

    p_exp->setValidationMap(m);
}
