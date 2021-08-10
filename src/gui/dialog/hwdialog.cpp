#include "hwdialog.h"

#include <QTreeView>
#include <QVBoxLayout>
#include <QLabel>
#include <QGroupBox>
#include <QDialogButtonBox>
#include <QPushButton>

#include <data/model/hwsettingsmodel.h>
#include <hardware/core/hardwareobject.h>

HWDialog::HWDialog(QString key, QWidget *controlWidget, QWidget *parent) : QDialog(parent)
{
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
    
    auto sView = new QTreeView(this);
    p_model = new HWSettingsModel(key,this);
    sView->setModel(p_model);
    svbl->addWidget(sView,1);
    
    sBox->setLayout(svbl);
    vbl->addWidget(sBox);
    
    auto bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Close);
    connect(bb->button(QDialogButtonBox::Ok),&QAbstractButton::clicked,this,&HWDialog::accept);
    connect(bb->button(QDialogButtonBox::Cancel),&QAbstractButton::clicked,this,&HWDialog::reject);
    
    vbl->addWidget(bb);
    setLayout(vbl);
}


void HWDialog::accept()
{
    //todo

    QDialog::accept();
}

void HWDialog::reject()
{
    //todo
    p_model->discardChanges(true);

    QDialog::reject();

}


QSize HWDialog::sizeHint() const
{
    return {500,500};
}
