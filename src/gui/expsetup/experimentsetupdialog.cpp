#include "experimentsetupdialog.h"

#include <QTreeWidget>
#include <QStackedWidget>
#include <QPushButton>
#include <QTextEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDialogButtonBox>

#include <gui/widget/experimentsummarywidget.h>

#include "experimenttypepage.h"
#include "experimentrfconfigpage.h"
#include "experimentloscanconfigpage.h"
#include "experimentdrscanconfigpage.h"

ExperimentSetupDialog::ExperimentSetupDialog(Experiment *exp, const std::map<QString, QString> &hw, const QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks, const std::map<QString, QStringList> &valKeys, QWidget *parent)
    : QDialog{parent}
{
    setWindowTitle("Experiment Setup");

    auto mainLayout = new QVBoxLayout;

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
    connect(p_validateButton,&QPushButton::clicked,this,&ExperimentSetupDialog::validateAll);
    vbl->addWidget(p_validateButton,0);

    hbl->addLayout(vbl,0);

    mainLayout->addLayout(hbl,1);
    auto bb = new QDialogButtonBox(QDialogButtonBox::Cancel,this);
    p_startButton = new QPushButton("Start Experiment");
    p_startButton->setAutoDefault(true);
    bb->addButton(p_startButton,QDialogButtonBox::AcceptRole);
    mainLayout->addWidget(bb);

    connect(bb,&QDialogButtonBox::rejected,this,&ExperimentSetupDialog::reject);
    connect(bb,&QDialogButtonBox::accepted,this,&ExperimentSetupDialog::accept);

    setLayout(mainLayout);

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
    sp->initialize();

    auto rfp = new ExperimentRfConfigPage(p_exp,clocks);
    k = BC::Key::WizRf::key;
    i = p_configWidget->addWidget(rfp);
    d_pages.insert({k,{i,k,rfp}});
    auto rfItem = new QTreeWidgetItem(expTypeItem,{rfp->d_title});
    rfp->setEnabled(sp->ftmwEnabled());
    rfItem->setDisabled(!sp->ftmwEnabled());
    rfItem->setData(0,Qt::UserRole,k);

    auto lop = new ExperimentLOScanConfigPage(p_exp);
    k = BC::Key::WizLoScan::key;
    i = p_configWidget->addWidget(lop);
    d_pages.insert({k,{i,k,lop}});
    auto loItem = new QTreeWidgetItem(rfItem,{lop->d_title});
    lop->setEnabled(sp->ftmwEnabled() && (sp->getFtmwType() == FtmwConfig::LO_Scan));
    loItem->setDisabled(!sp->ftmwEnabled() || (sp->getFtmwType() != FtmwConfig::LO_Scan));
    loItem->setData(0,Qt::UserRole,k);

    auto drop = new ExperimentDRScanConfigPage(p_exp);
    k = BC::Key::WizDR::key;
    i = p_configWidget->addWidget(drop);
    d_pages.insert({k,{i,k,drop}});
    auto dropItem = new QTreeWidgetItem(rfItem,{drop->d_title});
    drop->setEnabled(sp->ftmwEnabled() && (sp->getFtmwType() == FtmwConfig::DR_Scan));
    dropItem->setDisabled(!sp->ftmwEnabled() || (sp->getFtmwType() != FtmwConfig::DR_Scan));
    dropItem->setData(0,Qt::UserRole,k);

    connect(sp,&ExperimentTypePage::typeChanged,[=](){
        sp->apply();
        bool ften = sp->ftmwEnabled();

        rfp->setEnabled(ften);
        rfItem->setDisabled(!ften);

        lop->setEnabled(ften && (sp->getFtmwType() == FtmwConfig::LO_Scan));
        loItem->setDisabled(!ften || (sp->getFtmwType() != FtmwConfig::LO_Scan));

        drop->setEnabled(ften && (sp->getFtmwType() == FtmwConfig::DR_Scan));
        dropItem->setDisabled(!ften || (sp->getFtmwType() != FtmwConfig::DR_Scan));

        validateAll();

    });

    for(auto &[kk,id] : d_pages)
    {
        connect(id.page,&ExperimentConfigPage::warning,this,&ExperimentSetupDialog::warning);
        connect(id.page,&ExperimentConfigPage::error,this,&ExperimentSetupDialog::error);
    }

    p_navTree->expandAll();
    validate(p_navTree->invisibleRootItem(),true);
    p_summaryWidget->setExperiment(exp);

    connect(p_navTree,&QTreeWidget::currentItemChanged,this,&ExperimentSetupDialog::pageChanged);
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
                prevItem->setBackground(0,QBrush(Qt::red));
        }
    }

    if(newItem)
    {
        auto newPageKey = newItem->data(0,Qt::UserRole).toString();
        auto newit = d_pages.find(newPageKey);
        if(newit != d_pages.end())
        {
            newit->second.page->initialize();
            p_configWidget->setCurrentIndex(newit->second.index);
        }
    }

    validateAll();
}

bool ExperimentSetupDialog::validateAll(bool apply)
{
    //first, apply settings on current page if it validates
    auto page = dynamic_cast<ExperimentConfigPage*>(p_configWidget->currentWidget());
    if(page && page->validate())
        page->apply();

    p_statusTextEdit->clear();
    bool out = validate(p_navTree->invisibleRootItem(),apply);
    p_startButton->setEnabled(out);
    return out;
}

bool ExperimentSetupDialog::validate(QTreeWidgetItem *item, bool apply)
{
    auto pageKey = item->data(0,Qt::UserRole).toString();
    auto it = d_pages.find(pageKey);
    bool out = true;
    if(it != d_pages.end() && it->second.page->isEnabled())
    {
        out = it->second.page->validate();
        if(out)
        {
            if(apply)
                it->second.page->apply();
            item->setBackground(0,QBrush());
        }
        else
            item->setBackground(0,QBrush(Qt::red));
    }
    for(int i=0; i<item->childCount(); i++)
        out = out && validate(item->child(i),apply);

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


void ExperimentSetupDialog::reject()
{
    for(auto &[k,p] : d_pages)
        p.page->discardChanges();

    QDialog::reject();
}

void ExperimentSetupDialog::accept()
{
    if(validateAll(true))
        QDialog::accept();
}
