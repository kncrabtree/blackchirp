#include "experimentsetupdialog.h"
#include <gui/style/themecolors.h>

#include <QTreeWidget>
#include <QPushButton>
#include <QTextEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDialogButtonBox>

#include <gui/style/themecolors.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <hardware/optional/flowcontroller/flowcontroller.h>
#include <hardware/optional/tempcontroller/temperaturecontroller.h>
#include <hardware/optional/pressurecontroller/pressurecontroller.h>
#include <hardware/optional/ioboard/ioboard.h>

#include "experimenttypepage.h"
#include "experimentftmwconfigpage.h"
#include <gui/widget/rfconfigwidget.h>
#include "experimentpulsegenconfigpage.h"
#include "experimentflowconfigpage.h"
#include "experimenttemperaturecontrollerconfigpage.h"
#include "experimentpressurecontrollerconfigpage.h"
#include "experimentioboardconfigpage.h"
#include "experimentvalidatorconfigpage.h"
#include "experimentsummarypage.h"

#include <gui/lif/gui/experimentlifconfigpage.h>
#include <data/storage/applicationconfigmanager.h>

ExperimentSetupDialog::ExperimentSetupDialog(Experiment *exp, const QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks, const std::map<QString, QStringList, std::less<>> &valKeys, QWidget *parent)
    : QDialog{parent}
{
    setWindowTitle("Experiment Setup");
    
    // Set BlackChirp branding
    setWindowIcon(ThemeColors::createThemedIcon(":/icons/bc_logo_trans.svg", ThemeColors::IconPrimary, this));

    auto mainLayout = new QVBoxLayout;

    auto hbl = new QHBoxLayout;

    auto leftVbl = new QVBoxLayout;

    p_navTree = new QTreeWidget(this);
    p_navTree->setColumnCount(1);
    p_navTree->setMinimumWidth(200);
    p_navTree->setHeaderHidden(true);
    leftVbl->addWidget(p_navTree,3);

    p_statusTextEdit = new QTextEdit(this);
    p_statusTextEdit->setReadOnly(true);
    leftVbl->addWidget(p_statusTextEdit,1);

    p_validateButton = new QPushButton(QString("Validate"),this);
    connect(p_validateButton,&QPushButton::clicked,this,&ExperimentSetupDialog::validateAll);
    leftVbl->addWidget(p_validateButton,0);

    hbl->addLayout(leftVbl,0);

    p_configWidget = new QStackedWidget(this);
    hbl->addWidget(p_configWidget,1);

    mainLayout->addLayout(hbl,1);
    auto bb = new QDialogButtonBox(QDialogButtonBox::Cancel,this);
    p_startButton = new QPushButton("Start Experiment");
    p_startButton->setAutoDefault(true);
    bb->addButton(p_startButton,QDialogButtonBox::AcceptRole);
    mainLayout->addWidget(bb);

    connect(bb,&QDialogButtonBox::rejected,this,&ExperimentSetupDialog::reject);
    connect(bb,&QDialogButtonBox::accepted,this,&ExperimentSetupDialog::accept);

    setLayout(mainLayout);
    mainLayout->setSizeConstraint(QLayout::SetNoConstraint);
    setMinimumSize(1100, 850);

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

    auto en = ften;
    auto ftmwp = new ExperimentFtmwConfigPage(p_exp, clocks);
    connect(ftmwp,&ExperimentFtmwConfigPage::presetChanged,[this](){validateAll();});
    en = ften;
    k = BC::Key::WizFtmw::key;
    i = p_configWidget->addWidget(ftmwp);
    d_pages.insert({k,{i,k,ftmwp,en}});
    auto ftmwConfigItem = new QTreeWidgetItem(expTypeItem,{ftmwp->d_title});
    ftmwp->setEnabled(en);
    ftmwConfigItem->setDisabled(!en);
    ftmwConfigItem->setData(0,Qt::UserRole,k);

    connect(ftmwp->rfConfigWidget(), &RfConfigWidget::clockHwChanged,
            this, &ExperimentSetupDialog::onClockHwChanged);

    if(ApplicationConfigManager::instance().isLifEnabled()) {
        en = sp->lifEnabled();

        auto [lifp,lifpItem] = addConfigPage<ExperimentLifConfigPage>(BC::Key::WizLif::key,expTypeItem,en);
        Q_UNUSED(lifp)
        Q_UNUSED(lifpItem)
    }

    addOptHwPages<ExperimentPulseGenConfigPage>(QString(PulseGenerator::staticMetaObject.className()),expTypeItem);
    addOptHwPages<ExperimentFlowConfigPage>(QString(FlowController::staticMetaObject.className()),expTypeItem);
    addOptHwPages<ExperimentTemperatureControllerConfigPage>(QString(TemperatureController::staticMetaObject.className()),expTypeItem);
    addOptHwPages<ExperimentPressureControllerConfigPage>(QString(PressureController::staticMetaObject.className()),expTypeItem);
    addOptHwPages<ExperimentIOBoardConfigPage>(QString(IOBoard::staticMetaObject.className()),expTypeItem);

    auto valp = new ExperimentValidatorConfigPage(p_exp,valKeys);
    k = BC::Key::WizardVal::key;
    i = p_configWidget->addWidget(valp);
    d_pages.insert({k,{i,k,valp,true}});
    auto valItem = new QTreeWidgetItem(expTypeItem,{valp->d_title});
    valItem->setData(0,Qt::UserRole,k);

    auto sumP = new ExperimentSummaryPage(p_exp);
    k = BC::Key::WizSummary::key;
    i = p_configWidget->addWidget(sumP);
    d_pages.insert({k,{i,k,sumP,true}});
    auto sumItem = new QTreeWidgetItem({sumP->d_title});
    sumItem->setData(0,Qt::UserRole,k);
    p_navTree->addTopLevelItem(sumItem);


    connect(sp,&ExperimentTypePage::typeChanged,[=,this](){
        sp->apply();
        bool f = sp->ftmwEnabled();
        // auto t = sp->getFtmwType();

        d_pages[ftmwp->d_key].enabled = f;
        // d_pages[lop->d_key].enabled = f && (t == FtmwConfig::LO_Scan);
        // d_pages[drop->d_key].enabled = f && (t == FtmwConfig::DR_Scan);

        if(ApplicationConfigManager::instance().isLifEnabled()) {
            // Find LIF page in pages map and update its enabled state
            for(auto &[key, pageData] : d_pages) {
                if(key == BC::Key::WizLif::key) {
                    pageData.enabled = sp->lifEnabled();
                    break;
                }
            }
        }

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

    connect(p_navTree,&QTreeWidget::currentItemChanged,this,&ExperimentSetupDialog::pageChanged);
}

RfConfigWidget *ExperimentSetupDialog::rfConfigWidget()
{
    auto it = d_pages.find(BC::Key::WizFtmw::key);
    if(it == d_pages.end())
        return nullptr;

    auto p = dynamic_cast<ExperimentFtmwConfigPage*>(it->second.page);
    return p ? p->rfConfigWidget() : nullptr;
}

LifControlWidget *ExperimentSetupDialog::lifControlWidget()
{
    if(!ApplicationConfigManager::instance().isLifEnabled()) {
        return nullptr;
    }
    
    auto it = d_pages.find(BC::Key::WizLif::key);
    if(it == d_pages.end()) {
        return nullptr;
    }
    
    auto p = dynamic_cast<ExperimentLifConfigPage*>(it->second.page);
    return p == nullptr ? nullptr : p->lifControlWidget();
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
            }
            else
                prevItem->setBackground(0,QBrush(ThemeColors::getThemeAwareColor(ThemeColors::StatusError,this)));
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
            item->setBackground(0,QBrush(ThemeColors::getThemeAwareColor(ThemeColors::StatusError,this)));
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
    p_statusTextEdit->append(QString("<span style=\"font-weight:bold;color:%1\">%2</span>").arg(ThemeColors::getThemeAwareColor(ThemeColors::StatusWarning,this).name()).arg(text));
}

void ExperimentSetupDialog::error(const QString text)
{
    p_statusTextEdit->append(QString("<span style=\"font-weight:bold;color:%1\">%2</span>").arg(ThemeColors::getThemeAwareColor(ThemeColors::StatusError,this).name()).arg(text));

}


void ExperimentSetupDialog::reject()
{
    for(auto &[k,p] : d_pages)
    {
        p.page->discardChanges();

        // Also discard changes on child widgets that have their own
        // SettingsStorage (e.g., RfConfigWidget, ChirpConfigWidget,
        // ChirpTableModel, ClockTableModel) so settings don't leak
        // from a canceled wizard dialog.
        auto children = p.page->findChildren<QObject*>();
        for(auto c : children)
        {
            auto ss = dynamic_cast<SettingsStorage*>(c);
            if(ss)
                ss->discardChanges();
        }
    }

    QDialog::reject();
}

void ExperimentSetupDialog::onClockHwChanged()
{
    auto it = d_pages.find(BC::Key::WizStart::key);
    if(it != d_pages.end())
    {
        auto sp = dynamic_cast<ExperimentTypePage*>(it->second.page);
        if(sp)
            sp->initialize();
    }
    validateAll();
}

void ExperimentSetupDialog::accept()
{
    auto it = d_pages.find(BC::Key::WizFtmw::key);
    if (it != d_pages.end() && it->second.page->isEnabled())
    {
        if (auto *ftmwPage = dynamic_cast<ExperimentFtmwConfigPage*>(it->second.page))
            ftmwPage->commitFtmwPreset();
    }

    if(validateAll(true))
        QDialog::accept();
}
