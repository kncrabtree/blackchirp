#include "hwdialog.h"
#include <data/settings/hardwarekeys.h>

#include <QTreeView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QGroupBox>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QComboBox>
#include <QMetaEnum>

#include <data/model/hwsettingsmodel.h>
#include <hardware/core/hardwareobject.h>
#include <hardware/core/communication/communicationprotocol.h>

HWDialog::HWDialog(QString key, QStringList forbiddenKeys, QWidget *controlWidget, QWidget *parent) : QDialog(parent), d_hwKey(key)
{
    setAttribute(Qt::WA_DeleteOnClose);
    setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);

    auto vbl = new QVBoxLayout;
    
    SettingsStorage s(key,SettingsStorage::Hardware);
    auto model = s.get(BC::Key::HW::model, key);
    setWindowTitle(QString("%1 Settings").arg(key));

    if(controlWidget)
    {
        auto cBox = new QGroupBox(QString("%1 Control").arg(key));
        auto cvbl = new QVBoxLayout;
        
        auto cLabel = new QLabel(QString("Changes made in this section will be applied immediately."));
        cLabel->setWordWrap(true);
        cLabel->setAlignment(Qt::AlignCenter);
        cvbl->addWidget(cLabel,0);

        cvbl->addWidget(controlWidget);
        
        cBox->setLayout(cvbl);
        
        vbl->addWidget(cBox,0);
    }
    
    auto sBox = new QGroupBox(QString("%1 Settings").arg(key));
    auto svbl = new QVBoxLayout;
    
    auto sLabel = new QLabel("Changes made in this section will only be applied when this dialog is closed with the Ok button. Editing these settings incorrectly may result in unexpected behavior. Consider backing up your config file before making changes.");
    sLabel->setWordWrap(true);
    sLabel->setAlignment(Qt::AlignCenter);
    svbl->addWidget(sLabel,0);

    //Label showing hardware model
    auto nl = new QHBoxLayout;
    auto nLbl = new QLabel("Model");
    nl->addWidget(nLbl,0);
    auto modelLbl = new QLabel(model,this);
    nl->addWidget(modelLbl,1);
    svbl->addLayout(nl);


    // Protocol selection if multiple protocols are supported
    auto supportedProtocolsVar = s.get(BC::Key::HW::supportedProtocols, QVariantList());
    auto supportedProtocols = supportedProtocolsVar.toList();
    
    if(supportedProtocols.size() > 1) {
        auto pl = new QHBoxLayout;
        p_protocolLabel = new QLabel("Communication Protocol");
        pl->addWidget(p_protocolLabel,0);
        p_protocolCombo = new QComboBox(this);
        
        // Get the QMetaEnum for CommunicationProtocol::CommType
        auto commTypeEnum = QMetaEnum::fromType<CommunicationProtocol::CommType>();
        
        // Populate combo box with supported protocols
        auto currentProtocol = s.get(BC::Key::HW::commType, static_cast<int>(CommunicationProtocol::Virtual));
        int currentIndex = 0;
        
        for(int i = 0; i < supportedProtocols.size(); ++i) {
            auto protocolInt = supportedProtocols[i].toInt();
            QString protocolName = commTypeEnum.valueToKey(protocolInt);
            p_protocolCombo->addItem(protocolName, protocolInt);
            
            if(protocolInt == currentProtocol) {
                currentIndex = i;
            }
        }
        
        p_protocolCombo->setCurrentIndex(currentIndex);
        pl->addWidget(p_protocolCombo,1);
        svbl->addLayout(pl);
    } else {
        p_protocolCombo = nullptr;
        p_protocolLabel = nullptr;
    }
    
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

    connect(p_view,&QTreeView::clicked,this,[=,this](const QModelIndex &idx){
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
    vbl->addWidget(sBox,1);
    
    auto bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Close);
    connect(bb->button(QDialogButtonBox::Ok),&QPushButton::clicked,this,&HWDialog::accept);
    connect(bb->button(QDialogButtonBox::Close),&QPushButton::clicked,this,&HWDialog::reject);
    
    vbl->addWidget(bb,0);
    setLayout(vbl);
}

int HWDialog::getSelectedProtocol() const
{
    if(p_protocolCombo) {
        return p_protocolCombo->currentData().toInt();
    }
    return -1; // No protocol selection available
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
    auto selectedProtocol = getSelectedProtocol();
    p_model->saveChanges(selectedProtocol);

    QDialog::accept();
}

void HWDialog::reject()
{
    p_model->discardChanges(true);

    QDialog::reject();
}


QSize HWDialog::sizeHint() const
{
    return {500,850};
}
