#include "experimentsetupdialog.h"

#include <QTreeWidget>
#include <QStackedWidget>
#include <QPushButton>
#include <QTextEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>

#include <gui/widget/experimentsummarywidget.h>

#include "experimenttypepage.h"
#include "experimentrfconfigpage.h"

ExperimentSetupDialog::ExperimentSetupDialog(Experiment *exp, const std::map<QString, QString> &hw, const QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks, const std::map<QString, QStringList> &valKeys, QWidget *parent)
    : QDialog{parent}
{
    setWindowTitle("Experiment Setup");

    auto hbl = new QHBoxLayout;

    p_navTree = new QTreeWidget(this);
    p_navTree->setColumnCount(1);

    hbl->addWidget(p_navTree);

    p_configWidget = new QStackedWidget(this);
    hbl->addWidget(p_configWidget,1);

    auto vbl = new QVBoxLayout;

    p_summaryWidget = new ExperimentSummaryWidget(this);
    vbl->addWidget(p_summaryWidget,3);

    p_statusTextEdit = new QTextEdit(this);
    p_statusTextEdit->setReadOnly(true);
    vbl->addWidget(p_statusTextEdit,1);

    p_validateButton = new QPushButton(QString("Validate"),this);
    vbl->addWidget(p_validateButton,0);

    hbl->addLayout(vbl,0);

    setLayout(hbl);

    p_exp = exp;

    //first, create setup page and all FTMW pages.
    //then LIF (if enabled)
    //then optional HW pages
    //finally, validation page

    auto sp = new ExperimentTypePage(p_exp);
    auto k = BC::Key::WizStart::key;
    auto i = p_configWidget->addWidget(sp);
    d_pages.insert({k,{i,k,sp}});
    auto expTypeItem = new QTreeWidgetItem({sp->d_title});
    expTypeItem->setData(0,Qt::UserRole,k);
    p_navTree->addTopLevelItem(expTypeItem);

    auto rfp = new ExperimentRfConfigPage(p_exp,clocks);
    k = BC::Key::WizRf::key;
    rfp->setEnabled(sp->ftmwEnabled());
    i = p_configWidget->addWidget(rfp);
    d_pages.insert({k,{i,k,rfp}});
    auto rfItem = new QTreeWidgetItem(expTypeItem,{rfp->d_title});
    rfItem->setDisabled(!sp->ftmwEnabled());
    rfItem->setData(0,Qt::UserRole,k);


    connect(sp,&ExperimentTypePage::typeChanged,[=](){
       bool ften = sp->ftmwEnabled();

       rfp->setEnabled(ften);
       rfItem->setDisabled(!ften);

    });

    for(auto &[kk,id] : d_pages)
    {
        connect(id.page,&ExperimentConfigPage::warning,this,&ExperimentSetupDialog::warning);
        connect(id.page,&ExperimentConfigPage::error,this,&ExperimentSetupDialog::error);
    }

    validateAndApply(p_navTree->invisibleRootItem());
    p_summaryWidget->setExperiment(exp);

    connect(p_navTree,&QTreeWidget::currentItemChanged,this,&ExperimentSetupDialog::pageChanged);
    p_navTree->expandAll();
}

void ExperimentSetupDialog::pageChanged(QTreeWidgetItem *newItem, QTreeWidgetItem *prevItem)
{
    if(!prevItem)
        return;
    auto pageKey = prevItem->data(0,Qt::UserRole).toString();
    auto it = d_pages.find(pageKey);

    if(it != d_pages.end())
    {
        if(it->second.page->isEnabled())
        {
            if(it->second.page->validate())
            {
                it->second.page->apply();
                prevItem->setBackground(0,QBrush());
                p_summaryWidget->setExperiment(p_exp);
            }
            else
            {
                p_navTree->blockSignals(true);
                prevItem->setSelected(true);
                prevItem->setBackground(0,QBrush(Qt::red));
                if(newItem)
                    newItem->setSelected(false);
                p_navTree->setCurrentItem(prevItem);
                p_navTree->blockSignals(false);
                return;
            }
        }
    }

    if(newItem)
    {
        auto newPageKey = newItem->data(0,Qt::UserRole).toString();
        auto newit = d_pages.find(newPageKey);
        if(newit != d_pages.end())
            p_configWidget->setCurrentIndex(newit->second.index);
    }

    validateAll();
}

void ExperimentSetupDialog::validateAll()
{
    p_statusTextEdit->clear();
    validate(p_navTree->invisibleRootItem());
}

bool ExperimentSetupDialog::validate(QTreeWidgetItem *item)
{
    auto pageKey = item->data(0,Qt::UserRole).toString();
    auto it = d_pages.find(pageKey);
    bool out = true;
    if(it != d_pages.end() && it->second.page->isEnabled())
    {
        out = it->second.page->validate();
        if(out)
            item->setBackground(0,QBrush());
        else
            item->setBackground(0,QBrush(Qt::red));
    }
    for(int i=0; i<item->childCount(); i++)
        out = out && validate(item->child(i));

    return out;
}

bool ExperimentSetupDialog::validateAndApply(QTreeWidgetItem *item)
{
    auto pageKey = item->data(0,Qt::UserRole).toString();
    auto it = d_pages.find(pageKey);
    bool out = true;
    if(it != d_pages.end() && it->second.page->isEnabled())
    {
        out = it->second.page->validate();
        if(out)
            it->second.page->apply();
    }
    for(int i=0; i<item->childCount(); i++)
        out = out && validateAndApply(item->child(i));

    return out;
}

void ExperimentSetupDialog::warning(const QString text)
{
    p_statusTextEdit->append(text);
}

void ExperimentSetupDialog::error(const QString text)
{
    p_statusTextEdit->append(QString("<span style=\"font-weight:bold;color:red\">%1</span>").arg(text));

}
