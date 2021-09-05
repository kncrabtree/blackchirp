#include "hwdialog.h"

#include <QTreeView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QGroupBox>
#include <QDialogButtonBox>
#include <QPushButton>

#include <data/model/hwsettingsmodel.h>
#include <hardware/core/hardwareobject.h>

HWDialog::HWDialog(QString key, QStringList forbiddenKeys, QWidget *controlWidget, QWidget *parent) : QDialog(parent)
{
    setAttribute(Qt::WA_DeleteOnClose);
    setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);

    auto vbl = new QVBoxLayout;
    
    SettingsStorage s(key,SettingsStorage::Hardware);
    auto name = s.get(BC::Key::HW::name,key);
    setWindowTitle(QString("%1 Settings").arg(name));

    if(controlWidget)
    {
        auto cBox = new QGroupBox(QString("%1 Control").arg(name));
        auto cvbl = new QVBoxLayout;
        
        auto cLabel = new QLabel(QString("Changes made in this section will be applied immediately."));
        cLabel->setWordWrap(true);
        cLabel->setAlignment(Qt::AlignCenter);
        cvbl->addWidget(cLabel,0);

        cvbl->addWidget(controlWidget);
        
        cBox->setLayout(cvbl);
        
        vbl->addWidget(cBox,0);
    }
    
    auto sBox = new QGroupBox(QString("%1 Settings").arg(name));
    auto svbl = new QVBoxLayout;
    
    auto sLabel = new QLabel("Changes made in this section will only be applied when this dialog is closed with the Ok button. Editing these settings incorrectly may result in unexpected behavior. Consider backing up your config file before making changes.");
    sLabel->setWordWrap(true);
    sLabel->setAlignment(Qt::AlignCenter);
    svbl->addWidget(sLabel,0);
    
    p_view = new QTreeView(this);
    p_view->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::Preferred);
    p_model = new HWSettingsModel(key,forbiddenKeys,this);
    p_view->setModel(p_model);
    p_view->resizeColumnToContents(0);
    svbl->addWidget(p_view,1);

    auto hbl = new QHBoxLayout;

    auto ibButton = new QPushButton("Insert Before");
    ibButton->setEnabled(false);
    ibButton->setToolTip("Only possible for array values.");
    connect(ibButton,&QPushButton::clicked,this,&HWDialog::insertBefore);

    auto iaButton = new QPushButton("Insert After");
    iaButton->setEnabled(false);
    iaButton->setToolTip("Only possible for array values.");
    connect(iaButton,&QPushButton::clicked,this,&HWDialog::insertAfter);

    auto rButton = new QPushButton("Remove");
    rButton->setEnabled(false);
    rButton->setToolTip("Only possible for array values.");
    connect(rButton,&QPushButton::clicked,this,&HWDialog::remove);

    connect(p_view,&QTreeView::clicked,[=](const QModelIndex &idx){
        auto item = p_model->getItem(idx);
        if(item && item->canAddChildren())
        {
            ibButton->setEnabled(true);
            iaButton->setEnabled(true);
            rButton->setEnabled(true);
        }
        else
        {
            ibButton->setEnabled(false);
            iaButton->setEnabled(false);
            rButton->setEnabled(false);
        }
    });

    hbl->addWidget(ibButton,1);
    hbl->addWidget(iaButton,1);
    hbl->addWidget(rButton,1);
    svbl->addLayout(hbl,0);
    
    sBox->setLayout(svbl);
    vbl->addWidget(sBox);
    
    auto bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Close);
    connect(bb->button(QDialogButtonBox::Ok),&QPushButton::clicked,this,&HWDialog::accept);
    connect(bb->button(QDialogButtonBox::Close),&QPushButton::clicked,this,&HWDialog::reject);
    
    vbl->addWidget(bb);
    setLayout(vbl);
}

void HWDialog::insertBefore()
{
    auto idx = p_view->currentIndex();
    auto item = p_model->getItem(idx);

    if(!item->canAddChildren())
        return;

    p_model->insertRows(idx.row(),1,p_model->parent(idx));
}

void HWDialog::insertAfter()
{
    auto idx = p_view->currentIndex();
    auto item = p_model->getItem(idx);

    if(!item->canAddChildren())
        return;

    p_model->insertRows(idx.row()+1,1,p_model->parent(idx));
}

void HWDialog::remove()
{
    auto idx = p_view->currentIndex();
    auto item = p_model->getItem(idx);

    if(!item->canAddChildren())
        return;

    p_model->removeRows(idx.row(),1,p_model->parent(idx));
}


void HWDialog::accept()
{
    p_model->saveChanges();

    QDialog::accept();
}

void HWDialog::reject()
{
    p_model->discardChanges(true);

    QDialog::reject();
}


QSize HWDialog::sizeHint() const
{
    return {500,800};
}
