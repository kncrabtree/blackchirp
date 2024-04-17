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
#include "experimentchirpconfigpage.h"
#include "experimentftmwdigitizerconfigpage.h"

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
    auto ften = sp->ftmwEnabled();
    auto type = sp->getFtmwType();

    auto ftmwItem = new QTreeWidgetItem(expTypeItem,{"FTMW"});
    auto en = ften;
    ftmwItem->setDisabled(!en);
    ftmwItem->setData(0,Qt::UserRole,QString(""));

    auto rfp = new ExperimentRfConfigPage(p_exp,clocks);
    en = ften;
    k = BC::Key::WizRf::key;
    i = p_configWidget->addWidget(rfp);
    d_pages.insert({k,{i,k,rfp,en}});
    auto rfItem = new QTreeWidgetItem(ftmwItem,{rfp->d_title});
    rfp->setEnabled(en);
    rfItem->setDisabled(!en);
    rfItem->setData(0,Qt::UserRole,k);

    auto lop = new ExperimentLOScanConfigPage(p_exp);
    en = ften && (type == FtmwConfig::LO_Scan);
    k = BC::Key::WizLoScan::key;
    i = p_configWidget->addWidget(lop);
    d_pages.insert({k,{i,k,lop,en}});
    auto loItem = new QTreeWidgetItem(rfItem,{lop->d_title});
    lop->setEnabled(en);
    loItem->setDisabled(!en);
    loItem->setData(0,Qt::UserRole,k);

    auto drop = new ExperimentDRScanConfigPage(p_exp);
    en = ften && (type == FtmwConfig::DR_Scan);
    k = BC::Key::WizDR::key;
    i = p_configWidget->addWidget(drop);
    d_pages.insert({k,{i,k,drop,en}});
    auto dropItem = new QTreeWidgetItem(rfItem,{drop->d_title});
    drop->setEnabled(en);
    dropItem->setDisabled(en);
    dropItem->setData(0,Qt::UserRole,k);

    auto chp = new ExperimentChirpConfigPage(p_exp);
    en = ften;
    k = BC::Key::WizChirp::key;
    i = p_configWidget->addWidget(chp);
    d_pages.insert({k,{i,k,chp,en}});
    auto chpItem = new QTreeWidgetItem(rfItem,{chp->d_title});
    chp->setEnabled(en);
    chpItem->setDisabled(!en);
    chpItem->setData(0,Qt::UserRole,k);

    auto ftdp = new ExperimentFtmwDigitizerConfigPage(p_exp);
    en = ften;
    k = BC::Key::WizFtDig::key;
    i = p_configWidget->addWidget(ftdp);
    d_pages.insert({k,{i,k,ftdp,en}});
    auto ftdItem = new QTreeWidgetItem(rfItem,{ftdp->d_title});
    ftdp->setEnabled(en);
    ftdItem->setDisabled(!en);
    ftdItem->setData(0,Qt::UserRole,k);


    connect(sp,&ExperimentTypePage::typeChanged,[=](){
        sp->apply();
        bool f = sp->ftmwEnabled();
        auto t = sp->getFtmwType();

        d_pages[rfp->d_key].enabled = f;
        d_pages[lop->d_key].enabled = f && (t == FtmwConfig::LO_Scan);
        d_pages[drop->d_key].enabled = f && (t == FtmwConfig::DR_Scan);
        d_pages[chp->d_key].enabled = f;
        d_pages[ftdp->d_key].enabled = f;

        for( auto &[kk,pp] : d_pages)
            pp.page->setEnabled(pp.enabled);

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
        it->second.page->initialize();
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
    if(out)
    {
        enableChildren(item,true);
        for(int i=0; i<item->childCount(); i++)
            out = out && validate(item->child(i),apply);
    }
    else
        enableChildren(item,false);

    return out;
}

void ExperimentSetupDialog::enableChildren(QTreeWidgetItem *item, bool enable)
{
    for(int i=0; i<item->childCount(); i++)
    {
        auto c = item->child(i);
        enableChildren(c,enable);
        auto pageKey = c->data(0,Qt::UserRole).toString();
        auto it = d_pages.find(pageKey);
        if(it == d_pages.end())
        {
            c->setDisabled(!enable);
        }
        else
        {
            c->setDisabled(!enable || !it->second.enabled);
            it->second.page->setEnabled(enable && it->second.enabled);
        }
    }
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
