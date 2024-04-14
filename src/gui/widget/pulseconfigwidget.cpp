#include "pulseconfigwidget.h"

#include <QMetaEnum>
#include <QPushButton>
#include <QToolButton>
#include <QComboBox>
#include <QLineEdit>
#include <QDialog>
#include <QFormLayout>
#include <QDialogButtonBox>
#include <QMessageBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QStandardItemModel>
#include <QStandardItem>

#include <gui/plot/pulseplot.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <hardware/optional/chirpsource/awg.h>

PulseConfigWidget::PulseConfigWidget(QString key, QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::PulseWidget::key), d_key{key}
{

    auto ki = BC::Key::parseKey(key);
    pu_config = std::make_unique<PulseGenConfig>(ki.second);

    auto vc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    SettingsStorage s(d_key,Hardware);
    int numChannels = s.get<int>(BC::Key::PGen::numChannels);
    if(!containsArray(BC::Key::PulseWidget::channels))
        setArray(BC::Key::PulseWidget::channels,{});

    auto hbl = new QHBoxLayout;
    setLayout(hbl);

    auto vbl = new QVBoxLayout;
    hbl->addLayout(vbl,0);

    p_pulsePlot = new PulsePlot(key,this);
    hbl->addWidget(p_pulsePlot,1);

    auto mainGb = new QGroupBox("System Settings",this);
    auto gl = new QGridLayout;
    gl->addWidget(new QLabel("Pulsing Enabled"),0,0);
    gl->itemAtPosition(0,0)->setAlignment(Qt::AlignRight);
    p_sysOnOffButton = new QPushButton("Off",this);
    p_sysOnOffButton->setCheckable(true);
    p_sysOnOffButton->setChecked(false);
    connect(p_sysOnOffButton,&QPushButton::toggled,[=](bool en){
        emit changeSetting(d_key,-1,PulseGenConfig::PGenEnabledSetting,en);
        if(en)
            p_sysOnOffButton->setText(QString("On"));
        else
            p_sysOnOffButton->setText(QString("Off"));

        for(auto &ch : d_widgetList)
        {
            ch.modeBox->setDisabled(en || ch.locked);
            ch.syncBox->setDisabled(en || ch.locked);
        }
        p_sysModeBox->setDisabled(en);
    });
    gl->addWidget(p_sysOnOffButton,0,1);

    gl->addWidget(new QLabel("Pulse Mode"),0,2);
    gl->itemAtPosition(0,2)->setAlignment(Qt::AlignRight);
    p_sysModeBox = new EnumComboBox<PulseGenConfig::PGenMode>(this);
    connect(p_sysModeBox,qOverload<int>(&QComboBox::currentIndexChanged),[=](int i){
        emit changeSetting(d_key,-1,PulseGenConfig::PGenModeSetting,p_sysModeBox->value(i));
        p_repRateBox->setEnabled(p_sysModeBox->value(i) == PulseGenConfig::Continuous);
    });
    gl->addWidget(p_sysModeBox,0,3);

    gl->addWidget(new QLabel("Rep Rate"),1,2);
    gl->itemAtPosition(1,2)->setAlignment(Qt::AlignRight);
    p_repRateBox = new QDoubleSpinBox(this);
    p_repRateBox->setSuffix(QString(" Hz"));
    connect(p_repRateBox,qOverload<double>(&QDoubleSpinBox::valueChanged),[this](double val){
        emit changeSetting(d_key,-1,PulseGenConfig::RepRateSetting,val);
    });
    gl->addWidget(p_repRateBox,1,3);

    mainGb->setLayout(gl);


    vbl->addWidget(mainGb);

    auto chgb = new QGroupBox("Channel Configuration",this);
    auto pulseConfigBoxLayout = new QGridLayout;
    pulseConfigBoxLayout->addWidget(new QLabel("Ch"),0,0);
    pulseConfigBoxLayout->addWidget(new QLabel("Sync"),0,1);
    pulseConfigBoxLayout->addWidget(new QLabel("Delay"),0,2);
    pulseConfigBoxLayout->addWidget(new QLabel("Width"),0,3);
    pulseConfigBoxLayout->addWidget(new QLabel("Mode"),0,4);
    pulseConfigBoxLayout->addWidget(new QLabel("Enabled"),0,5);
    pulseConfigBoxLayout->addWidget(new QLabel("Cfg"),0,6);

    for(int i=0; i<7; i++)
        pulseConfigBoxLayout->itemAtPosition(0,i)->setAlignment(Qt::AlignCenter);

    chgb->setLayout(pulseConfigBoxLayout);

    vbl->addWidget(chgb);
    vbl->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::MinimumExpanding));

    QWidget *lastFocusWidget = nullptr;
    for(int i=0; i<numChannels; i++)
    {
        pu_config->addChannel();

        while(static_cast<std::size_t>(i) >= getArraySize(BC::Key::PulseWidget::channels))
            appendArrayMap(BC::Key::PulseWidget::channels,{});

        //values will be set later
        ChWidgets ch;
        int col = 0;

        ch.label = new QLabel(this);
        ch.label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        pulseConfigBoxLayout->addWidget(ch.label,i+1,col);
        col++;

        ch.syncBox = new QComboBox(this);
        for(int j=0; j<= numChannels; j++)
        {
            if(j == 0)
                ch.syncBox->addItem("T0",i);
            else
                ch.syncBox->addItem(QString("Ch")+QString::number(j),j);
        }
        auto md = dynamic_cast<QStandardItemModel*>(ch.syncBox->model());
        if(md != nullptr)
        {
            auto item = dynamic_cast<QStandardItem*>(md->item(i+1));
            if(item != nullptr)
                item->setEnabled(false);
        }
        connect(ch.syncBox,qOverload<int>(&QComboBox::currentIndexChanged),[=](int j){
            if(pu_config->testCircularSync(i,j))
            {
                QMessageBox::warning(this,"Circular Sync","Cannot set sync channel because of a circular reference (i.e., A triggers B, but B triggers A).",QMessageBox::Ok,QMessageBox::Ok);
                ch.syncBox->blockSignals(true);
                ch.syncBox->setCurrentIndex(pu_config->setting(i,PulseGenConfig::SyncSetting).toInt());
                ch.syncBox->blockSignals(false);
            }
            else
                emit changeSetting(d_key,i,PulseGenConfig::SyncSetting,j);
        });
        ch.syncBox->setEnabled(false);
        pulseConfigBoxLayout->addWidget(ch.syncBox,i+1,col);
        col++;

        ch.delayBox = new QDoubleSpinBox(this);
        ch.delayBox->setKeyboardTracking(false);
        ch.delayBox->setDecimals(3);
        ch.delayBox->setSuffix(QString::fromUtf16(u" µs"));
        pulseConfigBoxLayout->addWidget(ch.delayBox,i+1,col,1,1);
        connect(ch.delayBox,vc,[=](double val){
            emit changeSetting(d_key,i,PulseGenConfig::DelaySetting,val);
        } );
        col++;

        ch.widthBox = new QDoubleSpinBox(this);
        ch.widthBox->setKeyboardTracking(false);
        ch.widthBox->setDecimals(3);
        ch.widthBox->setSuffix(QString::fromUtf16(u" µs"));
        ch.widthBox->setSingleStep(getArrayValue<double>(BC::Key::PulseWidget::channels,i,
                                                         BC::Key::PulseWidget::widthStep,1.0));
        pulseConfigBoxLayout->addWidget(ch.widthBox,i+1,col,1,1);
        connect(ch.widthBox,vc,this,[=](double val){
            emit changeSetting(d_key,i,PulseGenConfig::WidthSetting,val);
        });
        col++;

        ch.modeBox = new EnumComboBox<PulseGenConfig::ChannelMode>(this);
        connect(ch.modeBox,qOverload<int>(&QComboBox::currentIndexChanged),[=](int j){
            emit changeSetting(d_key,i,PulseGenConfig::ModeSetting,ch.modeBox->value(j));
        });
        ch.modeBox->setEnabled(false);
        pulseConfigBoxLayout->addWidget(ch.modeBox,i+1,col);
        col++;

        ch.onButton = new QPushButton(this);
        ch.onButton->setCheckable(true);
        ch.onButton->setChecked(false);
        ch.onButton->setText(QString("Off"));
        pulseConfigBoxLayout->addWidget(ch.onButton,i+1,col,1,1);
        connect(ch.onButton,&QPushButton::toggled,this,[=](bool en){ emit changeSetting(d_key,i,PulseGenConfig::EnabledSetting,en); } );
        connect(ch.onButton,&QPushButton::toggled,this,[=](bool en){
            if(en)
                ch.onButton->setText(QString("On"));
            else
                ch.onButton->setText(QString("Off")); } );
        col++;

        ch.cfgButton = new QToolButton(this);
        ch.cfgButton->setIcon(QIcon(":/icons/configure.png"));
        ch.cfgButton->setIconSize(QSize(12,12));
        pulseConfigBoxLayout->addWidget(ch.cfgButton,i+1,col,1,1);
        connect(ch.cfgButton,&QToolButton::clicked,[=](){ launchChannelConfig(i); } );
        col++;
        lastFocusWidget = ch.cfgButton;

        ch.nameEdit = new QLineEdit(ch.label->text(),this);
        ch.nameEdit->setMaxLength(8);
        connect(ch.nameEdit,&QLineEdit::editingFinished,[=](){
            emit changeSetting(d_key,i,PulseGenConfig::NameSetting,ch.nameEdit->text());
        });
        ch.nameEdit->hide();

        ch.levelBox = new EnumComboBox<PulseGenConfig::ActiveLevel>(this);
        connect(ch.levelBox,qOverload<int>(&QComboBox::currentIndexChanged),[=](){
            emit changeSetting(d_key,i,PulseGenConfig::LevelSetting,ch.levelBox->currentValue());
        });
        ch.levelBox->hide();

        ch.delayStepBox = new QDoubleSpinBox(this);
        ch.delayStepBox->setDecimals(3);
        ch.delayStepBox->setRange(0.001,1000.0);
        ch.delayStepBox->setSuffix(QString::fromUtf16(u" µs"));
        connect(ch.delayStepBox,qOverload<double>(&QDoubleSpinBox::valueChanged),ch.delayBox,&QDoubleSpinBox::setSingleStep);
        ch.delayStepBox->hide();

        ch.widthStepBox = new QDoubleSpinBox(this);
        ch.widthStepBox->setDecimals(3);
        ch.widthStepBox->setRange(0.001,1000.0);
        ch.widthStepBox->setSuffix(QString::fromUtf16(u" µs"));
        connect(ch.widthStepBox,qOverload<double>(&QDoubleSpinBox::valueChanged),ch.widthBox,&QDoubleSpinBox::setSingleStep);
        ch.widthStepBox->hide();

        ch.roleBox = new EnumComboBox<PulseGenConfig::Role>(this);
        QMetaEnum rt = QMetaEnum::fromType<PulseGenConfig::Role>();
        ch.roleBox->hide();
        connect(ch.roleBox,qOverload<int>(&QComboBox::currentIndexChanged),[=](int index) {
            auto r = ch.roleBox->value(index);
            auto name = QString("Ch%1").arg(i+1);
            if(r == PulseGenConfig::None)
                ch.nameEdit->setEnabled(true);
            else
            {
                name = rt.valueToKey(r);
                ch.nameEdit->setEnabled(false);
            }
            ch.nameEdit->setText(name);
            emit changeSetting(d_key,i,PulseGenConfig::NameSetting,name);
            emit changeSetting(d_key,i,PulseGenConfig::RoleSetting,r);
        });

        ch.dutyOnBox = new QSpinBox(this);
        ch.dutyOnBox->setMinimum(1);
        connect(ch.dutyOnBox,qOverload<int>(&QSpinBox::valueChanged),[=](int d){
           emit changeSetting(d_key,i,PulseGenConfig::DutyOnSetting,d);
        });
        ch.dutyOnBox->hide();

        ch.dutyOffBox = new QSpinBox(this);
        ch.dutyOffBox->setMinimum(1);
        connect(ch.dutyOffBox,qOverload<int>(&QSpinBox::valueChanged),[=](int d){
           emit changeSetting(d_key,i,PulseGenConfig::DutyOffSetting,d);
        });
        ch.dutyOffBox->hide();

        d_widgetList.append(ch);


    }

    if(lastFocusWidget != nullptr)
        setTabOrder(lastFocusWidget,p_repRateBox);

    connect(p_repRateBox,vc,[this](double val){
        emit changeSetting(d_key,-1,PulseGenConfig::RepRateSetting,val);
    });

    updateFromSettings();

    setFocusPolicy(Qt::TabFocus);
    setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::Minimum);
}

PulseConfigWidget::~PulseConfigWidget()
{
}

PulseGenConfig PulseConfigWidget::getConfig() const
{
    return *pu_config;
}

void PulseConfigWidget::configureForWizard()
{    
    d_wizardMode = true;

    connect(this,&PulseConfigWidget::changeSetting,this,&PulseConfigWidget::newSetting);

    for(auto &ch : d_widgetList)
        ch.locked = false;
}

#ifdef BC_LIF
void PulseConfigWidget::configureLif(const LifConfig &c)
{
    if(d_widgetList.isEmpty())
        return;

    auto lifCh= pu_config->channelForRole(PulseGenConfig::LIF);
    if(lifCh < 0)
    {
        QMessageBox::warning(this,QString("Cannot configure LIF pulse"),QString("No channel has been configured for the \"LIF\" role. Blackchirp will be unable to set the LIF Delay.\n\nPlease select a channel for the LIF role, then refresh this page (go back one page and then come back to this one) in order to proceed."),QMessageBox::Ok,QMessageBox::Ok);
        return;
    }

    auto delay = c.delayRange().first;

    pu_config->setCh(PulseGenConfig::LIF,PulseGenConfig::DelaySetting,delay);
    pu_config->setCh(PulseGenConfig::LIF,PulseGenConfig::EnabledSetting,true);
    setFromConfig(d_config);

    lockChannel(lifCh);
}
#endif

void PulseConfigWidget::configureFtmw(const FtmwConfig &c)
{
    SettingsStorage s(BC::Key::AWG::key,Hardware);
    bool awgHasProt = s.get<bool>(BC::Key::AWG::prot,false);

    SettingsStorage s2(BC::Key::PGen::key,Hardware);
    bool pGenCanSync = s2.get(BC::Key::PGen::canSyncToChannel,false);

    auto protChannel = pu_config->channelForRole(PulseGenConfig::Prot);
    auto awgChannel = pu_config->channelForRole(PulseGenConfig::AWG);
    auto ampChannel = pu_config->channelForRole(PulseGenConfig::Amp);

    if(!awgHasProt && protChannel < 0)
    {
        QMessageBox::warning(this,QString("Cannot configure protection pulse"),QString("No channel has been configured for the \"Prot\" role, and your AWG does not produce its own protection signal.\n\nBlackchirp cannot guarantee that your receiver amp will be protected!\n\nIf you wish for Blackchirp to generate a protection pulse, close the wizard and configure the pulse generator under Hardware > Pulse Generator."),QMessageBox::Ok,QMessageBox::Ok);
        d_wizardOk = false;
    }


    if(!awgHasProt && awgChannel < 0)
    {
        QMessageBox::warning(this,QString("Cannot configure protection pulse"),QString("No channel has been configured for the \"AWG\" role, and your AWG does not produce its own protection signal.\n\nBlackchirp cannot guarantee that your receiver amp will be protected because it does not know when your AWG is triggered!\n\nIf you wish for Blackchirp to generate a protection pulse, close the wizard and configure the pulse generator under Hardware > Pulse Generator."),QMessageBox::Ok,QMessageBox::Ok);
        d_wizardOk = false;
    }


    if(!awgHasProt && c.d_rfConfig.d_chirpConfig.numChirps() > 1)
        QMessageBox::warning(this,QString("Warning: multiple chirps"),QString("You have requested multiple chirps, and your AWG cannot generate its own protection signal.\nBlackchirp does not know how to configure your delay generator to generate a protection signal with each chirp.\n\nProceed at your own risk."),QMessageBox::Ok,QMessageBox::Ok);

//    if(pGenCanSync)
//    {
//        if(ampChannel >= 0 && awgChannel >=0)
//        {
//            if(pu_config->d_channels.at(ampChannel).syncCh != protChannel + 1 ||
//                    pu_config->d_channels.at(awgChannel).syncCh != ampChannel + 1)
//                QMessageBox::warning(this,"Configuration notice","Blackchirp will change the sync sources for the Amp and AWG channels according to the sequence Prot -> Amp -> AWG.",QMessageBox::Ok,QMessageBox::Ok);
//        }
//        else if(awgChannel >= 0)
//        {
//            if(pu_config->d_channels.at(awgChannel).syncCh != protChannel + 1)
//                QMessageBox::warning(this,"Configuration notice","Blackchirp will change the sync source for the AWG channel to the Prot channel.",QMessageBox::Ok,QMessageBox::Ok);
//        }
//    }

    if(d_widgetList.isEmpty())
        return;

//    auto cc = c.d_rfConfig.d_chirpConfig;
//    pu_config->setCh(PulseGenConfig::AWG,PulseGenConfig::EnabledSetting,true);
//    pu_config->setCh(PulseGenConfig::Prot,PulseGenConfig::EnabledSetting,true);
//    pu_config->setCh(PulseGenConfig::Amp,PulseGenConfig::EnabledSetting,true);

//    double protStart = pu_config->setting(PulseGenConfig::Prot,PulseGenConfig::DelaySetting).toDouble();
//    double ampStart = cc.preChirpProtectionDelay();
//    if(!pGenCanSync)
//        ampStart += protStart;
//    else
//        pu_config->setCh(PulseGenConfig::Amp,PulseGenConfig::SyncSetting,protChannel+1);
//    double awgStart = cc.preChirpGateDelay();
//    if(!pGenCanSync)
//        awgStart += ampStart;
//    else
//    {
//        if(ampChannel < 0)
//            pu_config->setCh(PulseGenConfig::AWG,PulseGenConfig::SyncSetting,protChannel+1);
//        else
//            pu_config->setCh(PulseGenConfig::AWG,PulseGenConfig::SyncSetting,ampChannel+1);
//    }

//    double protWidth = cc.totalProtectionWidth();
//    double gateWidth = cc.totalGateWidth();

//    pu_config->setCh(PulseGenConfig::Prot,PulseGenConfig::WidthSetting,protWidth);
//    pu_config->setCh(PulseGenConfig::Amp,PulseGenConfig::DelaySetting,ampStart);
//    pu_config->setCh(PulseGenConfig::Amp,PulseGenConfig::WidthSetting,gateWidth);
//    pu_config->setCh(PulseGenConfig::AWG,PulseGenConfig::DelaySetting,awgStart);


    setFromConfig(d_key, d_config);


//    lockChannel(ampChannel);
//    lockChannel(protChannel);
//    lockChannel(awgChannel);

//    if(pGenCanSync && protChannel >= 0)
//        d_widgetList.at(protChannel).delayBox->setEnabled(true);



}

void PulseConfigWidget::launchChannelConfig(int ch)
{
    if(ch < 0 || ch >= d_widgetList.size())
        return;

    updateRoles();

    QDialog d(this);
    d.setWindowTitle(QString("Configure Pulse Channel %1").arg(ch+1));

    QFormLayout *fl = new QFormLayout();
    QVBoxLayout *vbl = new QVBoxLayout();
    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Close);
    connect(bb->button(QDialogButtonBox::Close),&QPushButton::clicked,&d,&QDialog::accept);

    ChWidgets chw = d_widgetList.at(ch);

    auto lbl = new QLabel(QString("Channel Name"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    fl->addRow(lbl,chw.nameEdit);

    lbl = new QLabel(QString("Role"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    fl->addRow(lbl,chw.roleBox);

    lbl = new QLabel(QString("Active Level"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    fl->addRow(lbl,chw.levelBox);

    lbl = new QLabel(QString("Duty Cycle On Pulses"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    fl->addRow(lbl,chw.dutyOnBox);

    lbl = new QLabel(QString("Duty Cycle Off Pulses"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    fl->addRow(lbl,chw.dutyOffBox);

    lbl = new QLabel(QString("Delay Step Size"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    fl->addRow(lbl,chw.delayStepBox);

    lbl = new QLabel(QString("Width Step Size"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    fl->addRow(lbl,chw.widthStepBox);



    chw.nameEdit->show();
    chw.roleBox->show();
    chw.levelBox->show();
    chw.dutyOnBox->show();
    chw.dutyOffBox->show();
    chw.delayStepBox->show();
    chw.widthStepBox->show();

    vbl->addLayout(fl,1);
    vbl->addWidget(bb);

    d.setLayout(vbl);
    d.exec();


    setArrayValue(BC::Key::PulseWidget::channels,ch,BC::Key::PulseWidget::delayStep,
                          chw.delayStepBox->value(),false);

    setArrayValue(BC::Key::PulseWidget::channels,ch,BC::Key::PulseWidget::widthStep,
                  chw.widthStepBox->value(),false);

    if(!d_wizardMode)
    {
        setArrayValue(BC::Key::PulseWidget::channels,ch,
                      BC::Key::PulseWidget::role,chw.roleBox->currentData(),false);

        setArrayValue(BC::Key::PulseWidget::channels,ch,
                      BC::Key::PulseWidget::name,chw.nameEdit->text(),false);
    }



    chw.nameEdit->setParent(this);
    chw.nameEdit->hide();
    chw.roleBox->setParent(this);
    chw.roleBox->hide();
    chw.levelBox->setParent(this);
    chw.levelBox->hide();
    chw.dutyOnBox->setParent(this);
    chw.dutyOnBox->hide();
    chw.dutyOffBox->setParent(this);
    chw.dutyOffBox->hide();
    chw.delayStepBox->setParent(this);
    chw.delayStepBox->hide();
    chw.widthStepBox->setParent(this);
    chw.widthStepBox->hide();

}

void PulseConfigWidget::newSetting(QString key, int index, PulseGenConfig::Setting s, QVariant val)
{
    if(index > d_widgetList.size() || key != d_key)
        return;

    blockSignals(true);
    pu_config->setCh(index,s,val);

    switch(s) {
    case PulseGenConfig::DelaySetting:
        d_widgetList.at(index).delayBox->setValue(val.toDouble());
        break;
    case PulseGenConfig::WidthSetting:
        d_widgetList.at(index).widthBox->setValue(val.toDouble());
        break;
    case PulseGenConfig::LevelSetting:
        d_widgetList.at(index).levelBox->setCurrentValue(val.value<PulseGenConfig::ActiveLevel>());
        break;
    case PulseGenConfig::EnabledSetting:
        d_widgetList.at(index).onButton->setChecked(val.toBool());
        break;
    case PulseGenConfig::NameSetting:
        d_widgetList.at(index).label->setText(val.toString());
        for(int i=0; i<d_widgetList.size(); i++)
            d_widgetList.at(i).syncBox->setItemText(index+1,val.toString());
        break;
    case PulseGenConfig::RoleSetting:
        updateRoles();
        break;
    case PulseGenConfig::ModeSetting:
        d_widgetList.at(index).modeBox->setCurrentValue(val.value<PulseGenConfig::ChannelMode>());
        break;
    case PulseGenConfig::SyncSetting:
        d_widgetList.at(index).syncBox->setCurrentIndex(val.toInt());
        break;
    case PulseGenConfig::DutyOnSetting:
        d_widgetList.at(index).dutyOnBox->setValue(val.toInt());
        break;
    case PulseGenConfig::DutyOffSetting:
        d_widgetList.at(index).dutyOffBox->setValue(val.toInt());
        break;
    case PulseGenConfig::RepRateSetting:
        p_repRateBox->setValue(val.toDouble());
        pu_config->d_repRate = val.toDouble();
        p_pulsePlot->newConfig(d_config);
        break;
    case PulseGenConfig::PGenModeSetting:
        p_sysModeBox->setCurrentValue(val.value<PulseGenConfig::PGenMode>());
        pu_config->d_mode = val.value<PulseGenConfig::PGenMode>();
        p_pulsePlot->newConfig(d_config);
        break;
    case PulseGenConfig::PGenEnabledSetting:
        p_sysOnOffButton->setChecked(val.toBool());
        pu_config->d_pulseEnabled = val.toBool();
        break;
    }


    blockSignals(false);

    p_pulsePlot->newConfig(d_config);
}

void PulseConfigWidget::setFromConfig(QString key, const PulseGenConfig &c)
{
    if(key != d_key)
        return;

    blockSignals(true);
    pu_config = std::make_unique<PulseGenConfig>(c);
    for(int i=0; i<c.size(); i++)
    {
        auto ch = c.d_channels.at(i);
        d_widgetList.at(i).delayBox->setValue(ch.delay);
        d_widgetList.at(i).widthBox->setValue(ch.width);
        d_widgetList.at(i).levelBox->setCurrentValue(ch.level);
        d_widgetList.at(i).onButton->setChecked(ch.enabled);
        d_widgetList.at(i).modeBox->setCurrentValue(ch.mode);
        for(int j=0; j<c.size(); j++)
            d_widgetList.at(j).syncBox->setItemText(i+1,ch.channelName);
        d_widgetList.at(i).syncBox->blockSignals(true);
        d_widgetList.at(i).syncBox->setCurrentIndex(ch.syncCh);
        d_widgetList.at(i).syncBox->blockSignals(false);
        d_widgetList.at(i).dutyOnBox->setValue(ch.dutyOn);
        d_widgetList.at(i).dutyOffBox->setValue(ch.dutyOff);

        d_widgetList.at(i).modeBox->setDisabled(c.d_pulseEnabled || d_widgetList.at(i).locked);
        d_widgetList.at(i).syncBox->setDisabled(c.d_pulseEnabled || d_widgetList.at(i).locked);
    }
    p_repRateBox->setValue(c.d_repRate);
    p_repRateBox->setEnabled(c.d_mode == PulseGenConfig::Continuous);
    p_sysModeBox->setCurrentValue(c.d_mode);
    p_sysModeBox->setDisabled(c.d_pulseEnabled);
    p_sysOnOffButton->setChecked(c.d_pulseEnabled);
    blockSignals(false);

    p_pulsePlot->newConfig(*pu_config);
}

void PulseConfigWidget::updateFromSettings()
{
    SettingsStorage s(BC::Key::PGen::key,Hardware);
    for(int i=0; i<d_widgetList.size(); i++)
    {
        auto chw = d_widgetList.at(i);

        chw.delayStepBox->setValue(getArrayValue(BC::Key::PulseWidget::channels,i,
                                                 BC::Key::PulseWidget::delayStep,1.0));
        chw.widthStepBox->setValue(getArrayValue(BC::Key::PulseWidget::channels,i,
                                                 BC::Key::PulseWidget::widthStep,1.0));


        chw.delayBox->blockSignals(true);
        chw.delayBox->setRange(s.get<double>(BC::Key::PGen::minDelay,0.0),
                               s.get<double>(BC::Key::PGen::maxDelay,1e5));
        chw.delayBox->blockSignals(false);


        chw.widthBox->blockSignals(true);
        chw.widthBox->setRange(s.get<double>(BC::Key::PGen::minWidth,0.010),
                               s.get<double>(BC::Key::PGen::maxWidth,1e5));
        chw.widthBox->blockSignals(false);


        auto r = getArrayValue<PulseGenConfig::Role>(BC::Key::PulseWidget::channels,i,
                                                     BC::Key::PulseWidget::role,PulseGenConfig::None);

        chw.roleBox->setCurrentValue(r);

        auto n = getArrayValue<QString>(BC::Key::PulseWidget::channels,i,
                                        BC::Key::PulseWidget::name,QString("Ch")+QString::number(i+1));


        if(chw.label != nullptr)
            chw.label->setText(n);

        if(chw.nameEdit != nullptr)
            chw.nameEdit->setText(n);

        chw.delayBox->setSingleStep(getArrayValue<double>(BC::Key::PulseWidget::channels,i,
                                                          BC::Key::PulseWidget::delayStep,1.0));

        chw.widthBox->setSingleStep(getArrayValue<double>(BC::Key::PulseWidget::channels,i,
                                                          BC::Key::PulseWidget::widthStep,1.0));

    }

    p_repRateBox->setRange(s.get(BC::Key::PGen::minRepRate,0.01),s.get(BC::Key::PGen::maxRepRate,1e5));

}

void PulseConfigWidget::unlockAll()
{
    for(int i=0; i<d_widgetList.size(); i++)
        lockChannel(i,false);
}

void PulseConfigWidget::lockChannel(int i, bool locked)
{
    if(i < 0 || i >= d_widgetList.size())
        return;

    auto &ch = d_widgetList[i];
    ch.widthBox->setDisabled(locked);
    ch.delayBox->setDisabled(locked);
    ch.onButton->setDisabled(locked);
    ch.roleBox->setDisabled(locked);
    ch.locked = locked;
}

void PulseConfigWidget::updateRoles()
{
    auto active = pu_config->activeRoles();

    for(int i=0; i<d_widgetList.size(); i++)
    {
        auto rb = d_widgetList.at(i).roleBox;
        for(int j=0; j<rb->count(); j++)
        {
            auto item = rb->itemAt(j);
            if(item)
            {
                bool en = !active.contains(rb->value(j));
                item->setEnabled(en);
            }
        }
    }
}

QSize PulseConfigWidget::sizeHint() const
{
    return {800,600};
}
