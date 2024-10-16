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

PulseConfigWidget::PulseConfigWidget(const PulseGenConfig &cfg, QWidget *parent) :
    QWidget(parent),
    SettingsStorage(BC::Key::widgetKey(BC::Key::PulseWidget::key,cfg.headerKey(),cfg.hwSubKey())),
    d_key{cfg.headerKey()}
{

    ps_config = std::make_shared<PulseGenConfig>(cfg);
    auto vc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);

    int numChannels = cfg.d_channels.size();
    if(!containsArray(BC::Key::PulseWidget::channels))
        setArray(BC::Key::PulseWidget::channels,{});

    auto hbl = new QHBoxLayout;
    setLayout(hbl);

    auto vbl = new QVBoxLayout;
    hbl->addLayout(vbl,0);

    auto mainGb = new QGroupBox("System Settings",this);
    auto gl = new QGridLayout;
    gl->addWidget(new QLabel("Pulsing Enabled"),0,0);
    gl->itemAtPosition(0,0)->setAlignment(Qt::AlignRight);
    p_sysOnOffButton = new QPushButton("Off",this);
    p_sysOnOffButton->setCheckable(true);
    p_sysOnOffButton->setChecked(false);
    connect(p_sysOnOffButton,&QPushButton::toggled,this,[this](bool en){
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
    connect(p_sysModeBox,qOverload<int>(&QComboBox::currentIndexChanged),this,[this](int i){
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
    for(int i=0; i<cfg.d_channels.size(); i++)
    {
        while(static_cast<std::size_t>(i) >= getArraySize(BC::Key::PulseWidget::channels))
            appendArrayMap(BC::Key::PulseWidget::channels,{});

        //values will be set later
        ChWidgets ch;
        int col = 0;

        ch.label = new QLabel(cfg.setting(i,PulseGenConfig::NameSetting).toString(),this);
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
        connect(ch.syncBox,qOverload<int>(&QComboBox::currentIndexChanged),this,[this,i,ch](int j){
            if(ps_config->testCircularSync(i,j))
            {
                QMessageBox::warning(this,"Circular Sync","Cannot set sync channel because of a circular reference (i.e., A triggers B, but B triggers A).",QMessageBox::Ok,QMessageBox::Ok);
                ch.syncBox->blockSignals(true);
                ch.syncBox->setCurrentIndex(ps_config->setting(i,PulseGenConfig::SyncSetting).toInt());
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
        connect(ch.delayBox,vc,this,[this,i](double val){
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
        connect(ch.widthBox,vc,this,[this,i](double val){
            emit changeSetting(d_key,i,PulseGenConfig::WidthSetting,val);
        });
        col++;

        ch.modeBox = new EnumComboBox<PulseGenConfig::ChannelMode>(this);
        connect(ch.modeBox,qOverload<int>(&QComboBox::currentIndexChanged),this,[this,i,ch](int j){
            emit changeSetting(d_key,i,PulseGenConfig::ModeSetting,ch.modeBox->value(j));
        });
        ch.modeBox->setEnabled(false);
        pulseConfigBoxLayout->addWidget(ch.modeBox,i+1,col);
        col++;

        ch.onButton = new QPushButton(this);
        ch.onButton->setCheckable(true);
        SettingsStorage s(cfg.headerKey(),SettingsStorage::Hardware);
        if(s.get(BC::Key::PGen::canDisableChannels,true))
        {
            ch.onButton->setChecked(false);
            ch.onButton->setText(QString("Off"));
        }
        else
        {
            ch.onButton->setChecked(true);
            ch.onButton->setText("On");
            ch.onButton->setEnabled(false);
        }
        pulseConfigBoxLayout->addWidget(ch.onButton,i+1,col,1,1);
        connect(ch.onButton,&QPushButton::toggled,this,[this,i](bool en){ emit changeSetting(d_key,i,PulseGenConfig::EnabledSetting,en); } );
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
        connect(ch.cfgButton,&QToolButton::clicked,this,[this,i](){ launchChannelConfig(i); } );
        col++;
        lastFocusWidget = ch.cfgButton;

        ch.nameEdit = new QLineEdit(ch.label->text(),this);
        ch.nameEdit->setMaxLength(8);
        connect(ch.nameEdit,&QLineEdit::editingFinished,this,[this,i,ch](){
            emit changeSetting(d_key,i,PulseGenConfig::NameSetting,ch.nameEdit->text());
        });
        ch.nameEdit->hide();

        ch.levelBox = new EnumComboBox<PulseGenConfig::ActiveLevel>(this);
        connect(ch.levelBox,qOverload<int>(&QComboBox::currentIndexChanged),this,[this,i,ch](){
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
        connect(ch.roleBox,qOverload<int>(&QComboBox::currentIndexChanged),this,[this,i,ch,rt](int index) {
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
        connect(ch.dutyOnBox,qOverload<int>(&QSpinBox::valueChanged),this,[this,i](int d){
           emit changeSetting(d_key,i,PulseGenConfig::DutyOnSetting,d);
        });
        ch.dutyOnBox->hide();

        ch.dutyOffBox = new QSpinBox(this);
        ch.dutyOffBox->setMinimum(1);
        connect(ch.dutyOffBox,qOverload<int>(&QSpinBox::valueChanged),this,[this,i](int d){
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

    //ps_config is initialized here
    p_pulsePlot = nullptr;
    setFromConfig(d_key,cfg);
    p_pulsePlot = new PulsePlot(ps_config,this);
    hbl->addWidget(p_pulsePlot,1);
}

PulseConfigWidget::~PulseConfigWidget()
{
}

const PulseGenConfig &PulseConfigWidget::getConfig() const
{
    return *ps_config;
}

void PulseConfigWidget::configureForWizard()
{    
    d_wizardMode = true;

    connect(this,&PulseConfigWidget::changeSetting,this,&PulseConfigWidget::newSetting);

    for(auto &ch : d_widgetList)
        ch.locked = false;
}

void PulseConfigWidget::launchChannelConfig(int ch)
{
    if(ch < 0 || ch >= d_widgetList.size())
        return;

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
    ps_config->setCh(index,s,val);

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
        d_widgetList.at(index).roleBox->setCurrentValue(val.value<PulseGenConfig::Role>());
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
        ps_config->d_repRate = val.toDouble();
        break;
    case PulseGenConfig::PGenModeSetting:
        p_sysModeBox->setCurrentValue(val.value<PulseGenConfig::PGenMode>());
        ps_config->d_mode = val.value<PulseGenConfig::PGenMode>();
        break;
    case PulseGenConfig::PGenEnabledSetting:
        p_sysOnOffButton->setChecked(val.toBool());
        ps_config->d_pulseEnabled = val.toBool();
        break;
    }
    blockSignals(false);

    p_pulsePlot->updatePulsePlot();
}

void PulseConfigWidget::setFromConfig(QString key, const PulseGenConfig &c)
{
    if(key != d_key)
        return;

    blockSignals(true);
    ps_config = std::make_shared<PulseGenConfig>(c);
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

    if(p_pulsePlot)
        p_pulsePlot->newConfig(ps_config);
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


        auto r = ps_config->setting(i,PulseGenConfig::RoleSetting).value<PulseGenConfig::Role>();

        chw.roleBox->setCurrentValue(r);

        auto n = ps_config->setting(i,PulseGenConfig::NameSetting).toString();

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

QSize PulseConfigWidget::sizeHint() const
{
    return {800,600};
}
