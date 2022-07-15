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

#include <gui/plot/pulseplot.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <hardware/optional/chirpsource/awg.h>

PulseConfigWidget::PulseConfigWidget(QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::PulseWidget::key)
{

    auto vc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    SettingsStorage s(BC::Key::PGen::key,Hardware);
    int numChannels = s.get<int>(BC::Key::PGen::numChannels);
    if(!containsArray(BC::Key::PulseWidget::channels))
        setArray(BC::Key::PulseWidget::channels,{});

    auto hbl = new QHBoxLayout;
    setLayout(hbl);

    auto vbl = new QVBoxLayout;
    hbl->addLayout(vbl,0);

    p_pulsePlot = new PulsePlot(this);
    hbl->addWidget(p_pulsePlot,1);

    auto mainGb = new QGroupBox("System Settings",this);
    auto gl = new QGridLayout;
    gl->addWidget(new QLabel("Pulsing Enabled"),0,0);
    gl->itemAtPosition(0,0)->setAlignment(Qt::AlignRight);
    p_sysOnOffButton = new QPushButton("Off",this);
    p_sysOnOffButton->setCheckable(true);
    p_sysOnOffButton->setChecked(false);
    connect(p_sysOnOffButton,&QPushButton::toggled,this,&PulseConfigWidget::changeSysPulsing);
    connect(p_sysOnOffButton,&QPushButton::toggled,[=](bool en){
        if(en)
            p_sysOnOffButton->setText(QString("On"));
        else
            p_sysOnOffButton->setText(QString("Off"));

        for(auto &ch : d_widgetList)
        {
            ch.modeBox->setDisabled(en || ch.locked);
            ch.syncBox->setDisabled(en || ch.locked);
        }
    });
    gl->addWidget(p_sysOnOffButton,0,1);

    gl->addWidget(new QLabel("Pulse Mode"),0,2);
    gl->itemAtPosition(0,2)->setAlignment(Qt::AlignRight);
    p_sysModeBox = new EnumComboBox<PulseGenConfig::PGenMode>(this);
    connect(p_sysModeBox,qOverload<int>(&QComboBox::currentIndexChanged),[=](int i){
        emit changeSysMode(p_sysModeBox->value(i));
        p_repRateBox->setEnabled(p_sysModeBox->value(i) == PulseGenConfig::Continuous);
    });
    gl->addWidget(p_sysModeBox,0,3);

    gl->addWidget(new QLabel("Rep Rate"),1,2);
    gl->itemAtPosition(1,2)->setAlignment(Qt::AlignRight);
    p_repRateBox = new QDoubleSpinBox(this);
    p_repRateBox->setSuffix(QString(" Hz"));
    connect(p_repRateBox,qOverload<double>(&QDoubleSpinBox::valueChanged),this,&PulseConfigWidget::changeRepRate);
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
        d_config.addChannel();

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
        connect(ch.syncBox,qOverload<int>(&QComboBox::currentIndexChanged),[=](int j){
            emit changeSetting(i,PulseGenConfig::SyncSetting,j);
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
            emit changeSetting(i,PulseGenConfig::DelaySetting,val);
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
            emit changeSetting(i,PulseGenConfig::WidthSetting,val);
        });
        col++;

        ch.modeBox = new EnumComboBox<PulseGenConfig::ChannelMode>(this);
        connect(ch.modeBox,qOverload<int>(&QComboBox::currentIndexChanged),[=](int j){
            emit changeSetting(i,PulseGenConfig::ModeSetting,ch.modeBox->value(j));
        });
        ch.modeBox->setEnabled(false);
        pulseConfigBoxLayout->addWidget(ch.modeBox,i+1,col);
        col++;

        ch.onButton = new QPushButton(this);
        ch.onButton->setCheckable(true);
        ch.onButton->setChecked(false);
        ch.onButton->setText(QString("Off"));
        pulseConfigBoxLayout->addWidget(ch.onButton,i+1,col,1,1);
        connect(ch.onButton,&QPushButton::toggled,this,[=](bool en){ emit changeSetting(i,PulseGenConfig::EnabledSetting,en); } );
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
        ch.nameEdit->hide();

        ch.levelBox = new EnumComboBox<PulseGenConfig::ActiveLevel>(this);
        ch.levelBox->hide();

        ch.delayStepBox = new QDoubleSpinBox(this);
        ch.delayStepBox->setDecimals(3);
        ch.delayStepBox->setRange(0.001,1000.0);
        ch.delayStepBox->setSuffix(QString::fromUtf16(u" µs"));
        ch.delayStepBox->hide();

        ch.widthStepBox = new QDoubleSpinBox(this);
        ch.widthStepBox->setDecimals(3);
        ch.widthStepBox->setRange(0.001,1000.0);
        ch.widthStepBox->setSuffix(QString::fromUtf16(u" µs"));
        ch.widthStepBox->hide();

        ch.roleBox = new EnumComboBox<PulseGenConfig::Role>(this);
        QMetaEnum rt = QMetaEnum::fromType<PulseGenConfig::Role>();
        ch.roleBox->hide();
        connect(ch.roleBox,qOverload<int>(&QComboBox::currentIndexChanged),[ch,i,rt](int index) {
            auto r = ch.roleBox->value(index);
            if(r == PulseGenConfig::None)
            {
                ch.nameEdit->setText(QString("Ch%1").arg(i+1));
                ch.nameEdit->setEnabled(true);
            }
            else
            {
                ch.nameEdit->setText(rt.valueToKey(r));
                ch.nameEdit->setEnabled(false);
            }
        });

        ch.dutyOnBox = new QSpinBox(this);
        ch.dutyOnBox->setMinimum(1);
        ch.dutyOnBox->hide();

        ch.dutyOffBox = new QSpinBox(this);
        ch.dutyOffBox->setMinimum(1);
        ch.dutyOffBox->hide();

        d_widgetList.append(ch);
    }

    if(lastFocusWidget != nullptr)
        setTabOrder(lastFocusWidget,p_repRateBox);

    connect(p_repRateBox,vc,this,&PulseConfigWidget::setRepRate);

    updateFromSettings();

    setFocusPolicy(Qt::TabFocus);
    setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::Minimum);
}

PulseConfigWidget::~PulseConfigWidget()
{
}

PulseGenConfig PulseConfigWidget::getConfig() const
{
    return d_config;
}

void PulseConfigWidget::configureForWizard()
{
    connect(this,&PulseConfigWidget::changeSetting,this,&PulseConfigWidget::newSetting);
    for(auto &ch : d_widgetList)
        ch.locked = false;
}

#ifdef BC_LIF
void PulseConfigWidget::configureLif(const LifConfig &c)
{
    if(d_widgetList.isEmpty())
        return;

    auto channels = d_config.channelsForRole(PulseGenConfig::LIF);
    if(channels.isEmpty())
    {
        QMessageBox::warning(this,QString("Cannot configure LIF pulse"),QString("No channel has been configured for the \"LIF\" role.\n\nPlease select a channel for the LIF role, then refresh this page (go back one page and then come back to this one) in order to proceed."),QMessageBox::Ok,QMessageBox::Ok);
        return;
    }

    auto delay = c.delayRange().first;

    d_config.setCh(PulseGenConfig::LIF,PulseGenConfig::DelaySetting,delay);
    d_config.setCh(PulseGenConfig::LIF,PulseGenConfig::EnabledSetting,true);
    setFromConfig(d_config);

    for(int i=0; i<channels.size(); i++)
        lockChannel(i);
}
#endif

void PulseConfigWidget::configureFtmw(const FtmwConfig &c)
{
    SettingsStorage s(BC::Key::AWG::key,Hardware);
    bool awgHasProt = s.get<bool>(BC::Key::AWG::prot,false);
    bool awgHasAmpEnable = s.get<bool>(BC::Key::AWG::amp,false);

    auto protChannels = d_config.channelsForRole(PulseGenConfig::Prot);
    auto awgChannels = d_config.channelsForRole(PulseGenConfig::AWG);
    auto ampChannels = d_config.channelsForRole(PulseGenConfig::Amp);

    if(!awgHasProt && protChannels.isEmpty())
        QMessageBox::warning(this,QString("Cannot configure protection pulse"),QString("No channel has been configured for the \"Prot\" role, and your AWG does not produce its own protection signal.\n\nBlackchirp cannot guarantee that your receiver amp will be protected!\n\nIf you wish for Blackchirp to generate a protection pulse, select a channel for the Prot role and refresh this page (go back one page and then come back to this one)."),QMessageBox::Ok,QMessageBox::Ok);

    if(!awgHasProt && awgChannels.isEmpty())
        QMessageBox::warning(this,QString("Cannot configure protection pulse"),QString("No channel has been configured for the \"AWG\" role, and your AWG does not produce its own protection signal.\n\nBlackchirp cannot guarantee that your receiver amp will be protected because it does not know when your AWG is triggered!\n\nIf you wish for Blackchirp to generate a protection pulse, select a channel for the AWG role and refresh this page (go back one page and then come back to this one)."),QMessageBox::Ok,QMessageBox::Ok);

    if(!awgHasProt && c.d_rfConfig.d_chirpConfig.numChirps() > 1)
        QMessageBox::warning(this,QString("Warning: multiple chirps"),QString("You have requested multiple chirps, and your AWG cannot generate its own protection signal.\nBlackchirp does not know how to configure your delay generator in a burst mode to generate a protection signal with each chirp.\n\nProceed at your own risk."),QMessageBox::Ok,QMessageBox::Ok);

    if(d_widgetList.isEmpty())
        return;

    auto cc = c.d_rfConfig.d_chirpConfig;
    d_config.setCh(PulseGenConfig::AWG,PulseGenConfig::EnabledSetting,true);
    auto l = d_config.setting(PulseGenConfig::AWG,PulseGenConfig::DelaySetting);

    if(l.size() > 1)
    {
        d_config.setCh(PulseGenConfig::AWG,PulseGenConfig::DelaySetting,l.constFirst());
        d_config.setCh(PulseGenConfig::AWG,PulseGenConfig::WidthSetting,d_config.setting(PulseGenConfig::Amp,PulseGenConfig::WidthSetting).constFirst().toDouble());
    }

    ///TODO:: account for sync settings here

    if(!l.isEmpty())
    {
        double awgStart = l.constFirst().toDouble();
        if(!awgHasProt)
        {
            double protStart = awgStart - cc.preChirpProtectionDelay() - cc.preChirpGateDelay();
            if(protStart < 0.0)
            {
                awgStart -= protStart;
                d_config.setCh(PulseGenConfig::AWG,PulseGenConfig::DelaySetting,awgStart);
                protStart = 0.0;
            }

            double protWidth = cc.totalProtectionWidth();

            d_config.setCh(PulseGenConfig::Prot,PulseGenConfig::DelaySetting,protStart);
            d_config.setCh(PulseGenConfig::Prot,PulseGenConfig::WidthSetting,protWidth);
            d_config.setCh(PulseGenConfig::Prot,PulseGenConfig::EnabledSetting,true);
        }

        bool checkProt = false;
        if(!awgHasAmpEnable)
        {
            double gateStart = awgStart - cc.preChirpGateDelay();
            if(gateStart < 0.0)
            {
                awgStart -= gateStart;
                d_config.setCh(PulseGenConfig::AWG,PulseGenConfig::DelaySetting,awgStart);
                gateStart = 0.0;
                checkProt = true;
            }

            double gateWidth = cc.totalGateWidth();

            d_config.setCh(PulseGenConfig::Amp,PulseGenConfig::DelaySetting,gateStart);
            d_config.setCh(PulseGenConfig::Amp,PulseGenConfig::WidthSetting,gateWidth);
            d_config.setCh(PulseGenConfig::Amp,PulseGenConfig::EnabledSetting,true);
        }

        if(!awgHasProt && checkProt)
        {
            double protStart = awgStart - cc.preChirpProtectionDelay() - cc.preChirpGateDelay();
            double protWidth = cc.totalProtectionWidth();

            d_config.setCh(PulseGenConfig::Prot,PulseGenConfig::DelaySetting,protStart);
            d_config.setCh(PulseGenConfig::Prot,PulseGenConfig::WidthSetting,protWidth);
        }
    }

    setFromConfig(d_config);

    for(int i=0; i<awgChannels.size(); i++)
        lockChannel(i);

    for(int i=0; i<protChannels.size(); i++)
        lockChannel(i);

    for(int i=0; i<ampChannels.size(); i++)
        lockChannel(i);


}

void PulseConfigWidget::launchChannelConfig(int ch)
{
    if(ch < 0 || ch >= d_widgetList.size())
        return;

    QDialog d(this);
    d.setWindowTitle(QString("Configure Pulse Channel %1").arg(ch+1));

    QFormLayout *fl = new QFormLayout();
    QVBoxLayout *vbl = new QVBoxLayout();
    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel);
    connect(bb->button(QDialogButtonBox::Ok),&QPushButton::clicked,&d,&QDialog::accept);
    connect(bb->button(QDialogButtonBox::Cancel),&QPushButton::clicked,&d,&QDialog::reject);

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
    if(d.exec() == QDialog::Accepted)
    {

        chw.delayBox->setSingleStep(chw.delayStepBox->value());
        setArrayValue(BC::Key::PulseWidget::channels,ch,BC::Key::PulseWidget::delayStep,
                      chw.delayStepBox->value(),false);

        chw.widthBox->setSingleStep(chw.widthStepBox->value());
        setArrayValue(BC::Key::PulseWidget::channels,ch,BC::Key::PulseWidget::widthStep,
                      chw.widthStepBox->value(),false);

        d_config.setCh(ch,PulseGenConfig::LevelSetting,chw.levelBox->currentValue());
        emit changeSetting(ch,PulseGenConfig::LevelSetting,chw.levelBox->currentValue());

        setArrayValue(BC::Key::PulseWidget::channels,ch,
                      BC::Key::PulseWidget::role,chw.roleBox->currentData(),false);
        d_config.setCh(ch,PulseGenConfig::RoleSetting,chw.roleBox->currentData());
        emit changeSetting(ch,PulseGenConfig::RoleSetting,d_config.at(ch).role);

        chw.label->setText(chw.nameEdit->text());
        //this is the last setting to save; write the array
        setArrayValue(BC::Key::PulseWidget::channels,ch,
                      BC::Key::PulseWidget::name,chw.nameEdit->text(),true);
        d_config.setCh(ch,PulseGenConfig::NameSetting,chw.nameEdit->text());
        emit changeSetting(ch,PulseGenConfig::NameSetting,chw.nameEdit->text());


        p_pulsePlot->newConfig(d_config);
        updateFromSettings();
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

void PulseConfigWidget::newSetting(int index, PulseGenConfig::Setting s, QVariant val)
{
    if(index < 0 || index > d_widgetList.size())
        return;

    blockSignals(true);

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
    case PulseGenConfig::RoleSetting:
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
    }

    d_config.setCh(index,s,val);
    blockSignals(false);

    p_pulsePlot->newSetting(index,s,val);
}

void PulseConfigWidget::setFromConfig(const PulseGenConfig &c)
{
    blockSignals(true);
    d_config = c;
    for(int i=0; i<c.size(); i++)
    {
        auto ch = c.d_channels.at(i);
        d_widgetList.at(i).delayBox->setValue(ch.delay);
        d_widgetList.at(i).widthBox->setValue(ch.width);
        d_widgetList.at(i).levelBox->setCurrentValue(ch.level);
        d_widgetList.at(i).onButton->setChecked(ch.enabled);
        d_widgetList.at(i).modeBox->setCurrentValue(ch.mode);
        d_widgetList.at(i).syncBox->setCurrentIndex(ch.syncCh);
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



    p_pulsePlot->newConfig(d_config);
}

void PulseConfigWidget::newRepRate(double r)
{
    p_repRateBox->blockSignals(true);
    p_repRateBox->setValue(r);
    p_repRateBox->blockSignals(false);
    p_pulsePlot->newRepRate(r);
    d_config.d_repRate = r;
}

void PulseConfigWidget::updateFromSettings()
{
    SettingsStorage s(BC::Key::PGen::key,Hardware);
    for(int i=0; i<d_widgetList.size(); i++)
    {
        auto chw = d_widgetList.at(i);

        if(chw.delayBox != nullptr)
        {
            chw.delayBox->blockSignals(true);
            chw.delayBox->setRange(s.get<double>(BC::Key::PGen::minDelay,0.0),
                          s.get<double>(BC::Key::PGen::maxDelay,1e5));
            chw.delayBox->blockSignals(false);
        }

        if(chw.widthBox != nullptr)
        {
            chw.widthBox->blockSignals(true);
            chw.widthBox->setRange(s.get<double>(BC::Key::PGen::minWidth,0.010),
                          s.get<double>(BC::Key::PGen::maxWidth,1e5));
            chw.widthBox->blockSignals(false);
        }

        auto r = getArrayValue<PulseGenConfig::Role>(BC::Key::PulseWidget::channels,i,
                               BC::Key::PulseWidget::role,PulseGenConfig::None);
        if(chw.roleBox != nullptr)
            chw.roleBox->setCurrentIndex(chw.roleBox->findData(QVariant::fromValue(r)));

        auto n = getArrayValue<QString>(BC::Key::PulseWidget::channels,i,
                                        BC::Key::PulseWidget::name,QString("Ch")+QString::number(i+1));

        d_config.setCh(i,PulseGenConfig::RoleSetting,r);
        d_config.setCh(i,PulseGenConfig::NameSetting,n);

        if(chw.label != nullptr)
            chw.label->setText(n);

        if(chw.nameEdit != nullptr)
            chw.nameEdit->setText(n);

        p_pulsePlot->newSetting(i,PulseGenConfig::NameSetting,n);

        chw.delayBox->setSingleStep(getArrayValue<double>(BC::Key::PulseWidget::channels,i,
                                                               BC::Key::PulseWidget::delayStep,1.0));

        chw.widthBox->setSingleStep(getArrayValue<double>(BC::Key::PulseWidget::channels,i,
                                                               BC::Key::PulseWidget::widthStep,1.0));

    }

    p_repRateBox->setRange(s.get(BC::Key::PGen::minRepRate,0.01),s.get(BC::Key::PGen::maxRepRate,1e5));

}

void PulseConfigWidget::setRepRate(const double r)
{
    p_pulsePlot->newRepRate(r);
    d_config.d_repRate = r;
    emit changeRepRate(r);
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

QSize PulseConfigWidget::sizeHint() const
{
    return {800,600};
}
