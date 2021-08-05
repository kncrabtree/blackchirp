#include "pulseconfigwidget.h"
#include "ui_pulseconfigwidget.h"

#include <QMetaEnum>
#include <QPushButton>
#include <QToolButton>
#include <QComboBox>
#include <QLineEdit>
#include <QDialog>
#include <QFormLayout>
#include <QDialogButtonBox>
#include <QMessageBox>

#include <hardware/core/pulsegenerator/pulsegenerator.h>
#include <hardware/core/chirpsource/awg.h>

PulseConfigWidget::PulseConfigWidget(QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::PulseWidget::key),
    ui(new Ui::PulseConfigWidget)
{
    ui->setupUi(this);    

    auto vc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    SettingsStorage s(BC::Key::PGen::key,Hardware);
    int numChannels = s.get<int>(BC::Key::PGen::numChannels);
    if(!containsArray(BC::Key::PulseWidget::channels))
        setArray(BC::Key::PulseWidget::channels,{});

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
        ui->pulseConfigBoxLayout->addWidget(ch.label,i+1,col,1,1);
        col++;

        ch.delayBox = new QDoubleSpinBox(this);
        ch.delayBox->setKeyboardTracking(false);
        ch.delayBox->setDecimals(3);
        ch.delayBox->setSuffix(QString::fromUtf16(u" µs"));
        ch.delayBox->setSingleStep(s.getArrayValue<double>(BC::Key::PulseWidget::channels,i,
                                                           BC::Key::PulseWidget::delayStep,1.0));
        ui->pulseConfigBoxLayout->addWidget(ch.delayBox,i+1,col,1,1);
        connect(ch.delayBox,vc,[=](double val){
            emit changeSetting(i,PulseGenConfig::DelaySetting,val);
        } );
        col++;

        ch.widthBox = new QDoubleSpinBox(this);
        ch.widthBox->setKeyboardTracking(false);
        ch.widthBox->setDecimals(3);
        ch.widthBox->setSuffix(QString::fromUtf16(u" µs"));
        ch.widthBox->setSingleStep(get<double>(BC::Key::PulseWidget::widthStep,1.0));
        ui->pulseConfigBoxLayout->addWidget(ch.widthBox,i+1,col,1,1);
        connect(ch.widthBox,vc,this,[=](double val){ emit changeSetting(i,PulseGenConfig::WidthSetting,val); } );
        col++;

        ch.onButton = new QPushButton(this);
        ch.onButton->setCheckable(true);
        ch.onButton->setChecked(false);
        ch.onButton->setText(QString("Off"));
        ui->pulseConfigBoxLayout->addWidget(ch.onButton,i+1,col,1,1);
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
        ui->pulseConfigBoxLayout->addWidget(ch.cfgButton,i+1,col,1,1);
        connect(ch.cfgButton,&QToolButton::clicked,[=](){ launchChannelConfig(i); } );
        col++;
        lastFocusWidget = ch.cfgButton;

        ch.nameEdit = new QLineEdit(ch.label->text(),this);
        ch.nameEdit->setMaxLength(8);
        ch.nameEdit->hide();

        ch.levelButton = new QPushButton(this);
        ch.levelButton->setCheckable(true);
        ch.levelButton->setChecked(true);
        ch.levelButton->setText(QString("Active High"));

        connect(ch.levelButton,&QPushButton::toggled,[=](bool en){
            if(en)
                ch.levelButton->setText(QString("Active High"));
            else
                ch.levelButton->setText(QString("Active Low"));
        });
        ch.levelButton->hide();

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

        ch.roleBox = new QComboBox(this);
        QMetaEnum rt = QMetaEnum::fromType<PulseGenConfig::Role>();
        for(int i=0; i<rt.keyCount(); ++i)
            ch.roleBox->addItem(rt.key(i),rt.value(i));
        ch.roleBox->hide();
        connect(ch.roleBox,static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),[ch,i,rt](int index) {
            auto r = ch.roleBox->itemData(index).value<PulseGenConfig::Role>();
            if(r == PulseGenConfig::None)
            {
                ch.nameEdit->setText(QString("Ch%1").arg(i));
                ch.nameEdit->setEnabled(true);
            }
            else
            {
                ch.nameEdit->setText(QString(rt.key(r)));
                ch.nameEdit->setEnabled(false);
            }
        });

        d_widgetList.append(ch);
    }

    ui->pulseConfigBoxLayout->setColumnStretch(0,0);
    ui->pulseConfigBoxLayout->setColumnStretch(1,1);
    ui->pulseConfigBoxLayout->setColumnStretch(2,1);
    ui->pulseConfigBoxLayout->setColumnStretch(3,0);
    ui->pulseConfigBoxLayout->setColumnStretch(4,0);

    if(lastFocusWidget != nullptr)
        setTabOrder(lastFocusWidget,ui->repRateBox);

    connect(ui->repRateBox,vc,this,&PulseConfigWidget::setRepRate);

    updateFromSettings();

    setFocusPolicy(Qt::TabFocus);
}

PulseConfigWidget::~PulseConfigWidget()
{
    delete ui;
}

PulseGenConfig PulseConfigWidget::getConfig() const
{
    return d_config;
}

void PulseConfigWidget::configureForWizard()
{
    connect(this,&PulseConfigWidget::changeSetting,this,&PulseConfigWidget::newSetting);
}

#ifdef BC_LIF
void PulseConfigWidget::configureLif(const LifConfig c)
{
    if(d_widgetList.isEmpty() || !c.isEnabled())
        return;

    auto channels = d_config.channelsForRole(PulseGenConfig::LifRole);
    if(channels.isEmpty())
    {
        QMessageBox::warning(this,QString("Cannot configure LIF pulse"),QString("No channel has been configured for the \"LIF\" role.\n\nPlease select a channel for the LIF role, then refresh this page (go back one page and then come back to this one) in order to proceed."),QMessageBox::Ok,QMessageBox::Ok);
        return;
    }

    auto delay = c.delayRange().first;

    d_config.set(PulseGenConfig::LifRole,PulseGenConfig::DelaySetting,delay);
    d_config.set(PulseGenConfig::LifRole,PulseGenConfig::EnabledSetting,true);
    setFromConfig(d_config);

    for(int i=0; i<channels.size(); i++)
    {
        auto ch = channels.at(i);
        d_widgetList.at(ch).delayBox->setEnabled(false);
        d_widgetList.at(ch).onButton->setEnabled(false);
        d_widgetList.at(ch).cfgButton->setEnabled(false);
    }
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
    d_config.set(PulseGenConfig::AWG,PulseGenConfig::EnabledSetting,true);
    auto l = d_config.setting(PulseGenConfig::AWG,PulseGenConfig::DelaySetting);

    if(l.size() > 1)
    {
        d_config.set(PulseGenConfig::AWG,PulseGenConfig::DelaySetting,l.constFirst());
        d_config.set(PulseGenConfig::AWG,PulseGenConfig::WidthSetting,d_config.setting(PulseGenConfig::Amp,PulseGenConfig::WidthSetting).constFirst().toDouble());
    }

    if(!l.isEmpty())
    {
        double awgStart = l.constFirst().toDouble();
        if(!awgHasProt)
        {
            double protStart = awgStart - cc.preChirpProtectionDelay() - cc.preChirpGateDelay();
            if(protStart < 0.0)
            {
                awgStart -= protStart;
                d_config.set(PulseGenConfig::AWG,PulseGenConfig::DelaySetting,awgStart);
                protStart = 0.0;
            }

            double protWidth = cc.totalProtectionWidth();

            d_config.set(PulseGenConfig::Prot,PulseGenConfig::DelaySetting,protStart);
            d_config.set(PulseGenConfig::Prot,PulseGenConfig::WidthSetting,protWidth);
            d_config.set(PulseGenConfig::Prot,PulseGenConfig::EnabledSetting,true);
        }

        bool checkProt = false;
        if(!awgHasAmpEnable)
        {
            double gateStart = awgStart - cc.preChirpGateDelay();
            if(gateStart < 0.0)
            {
                awgStart -= gateStart;
                d_config.set(PulseGenConfig::AWG,PulseGenConfig::DelaySetting,awgStart);
                gateStart = 0.0;
                checkProt = true;
            }

            double gateWidth = cc.totalGateWidth();

            d_config.set(PulseGenConfig::Amp,PulseGenConfig::DelaySetting,gateStart);
            d_config.set(PulseGenConfig::Amp,PulseGenConfig::WidthSetting,gateWidth);
            d_config.set(PulseGenConfig::Amp,PulseGenConfig::EnabledSetting,true);
        }

        if(!awgHasProt && checkProt)
        {
            double protStart = awgStart - cc.preChirpProtectionDelay() - cc.preChirpGateDelay();
            double protWidth = cc.totalProtectionWidth();

            d_config.set(PulseGenConfig::Prot,PulseGenConfig::DelaySetting,protStart);
            d_config.set(PulseGenConfig::Prot,PulseGenConfig::WidthSetting,protWidth);
        }
    }

    setFromConfig(d_config);

    for(int i=0; i<awgChannels.size(); i++)
    {
        auto ch = awgChannels.at(i);
        d_widgetList.at(ch).onButton->setEnabled(false);
        d_widgetList.at(ch).delayBox->setEnabled(false);
        d_widgetList.at(ch).cfgButton->setEnabled(false);
    }

    for(int i=0; i<protChannels.size(); i++)
    {
        auto ch = protChannels.at(i);
        d_widgetList.at(ch).onButton->setEnabled(false);
        d_widgetList.at(ch).delayBox->setEnabled(false);
        d_widgetList.at(ch).widthBox->setEnabled(false);
        d_widgetList.at(ch).cfgButton->setEnabled(false);
    }

    for(int i=0; i<ampChannels.size(); i++)
    {
        auto ch = ampChannels.at(i);
        d_widgetList.at(ch).onButton->setEnabled(false);
        d_widgetList.at(ch).delayBox->setEnabled(false);
        d_widgetList.at(ch).widthBox->setEnabled(false);
        d_widgetList.at(ch).cfgButton->setEnabled(false);
    }


}

void PulseConfigWidget::launchChannelConfig(int ch)
{
    if(ch < 0 || ch >= d_widgetList.size())
        return;

    QDialog d(this);
    d.setWindowTitle(QString("Configure Pulse Channel %1").arg(ch));

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

    lbl = new QLabel(QString("Active Level"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    fl->addRow(lbl,chw.levelButton);

    lbl = new QLabel(QString("Delay Step Size"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    fl->addRow(lbl,chw.delayStepBox);

    lbl = new QLabel(QString("Width Step Size"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    fl->addRow(lbl,chw.widthStepBox);

    lbl = new QLabel(QString("Role"));
    lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
    fl->addRow(lbl,chw.roleBox);



    chw.nameEdit->show();
    chw.levelButton->show();
    chw.delayStepBox->show();
    chw.widthStepBox->show();
    chw.roleBox->show();

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

        if(chw.levelButton->isChecked())
        {
            d_config.set(ch,PulseGenConfig::LevelSetting,QVariant::fromValue(PulseGenConfig::ActiveHigh));
            emit changeSetting(ch,PulseGenConfig::LevelSetting,QVariant::fromValue(PulseGenConfig::ActiveHigh));
        }
        else
        {
            d_config.set(ch,PulseGenConfig::LevelSetting,QVariant::fromValue(PulseGenConfig::ActiveLow));
            emit changeSetting(ch,PulseGenConfig::LevelSetting,QVariant::fromValue(PulseGenConfig::ActiveLow));
        }

        setArrayValue(BC::Key::PulseWidget::channels,ch,
                      BC::Key::PulseWidget::role,chw.roleBox->currentData(),false);
        d_config.set(ch,PulseGenConfig::RoleSetting,chw.roleBox->currentData());
        emit changeSetting(ch,PulseGenConfig::RoleSetting,d_config.at(ch).role);

        chw.label->setText(chw.nameEdit->text());
        //this is the last setting to save; write the array
        setArrayValue(BC::Key::PulseWidget::channels,ch,
                      BC::Key::PulseWidget::name,chw.nameEdit->text(),true);
        d_config.set(ch,PulseGenConfig::NameSetting,chw.nameEdit->text());
        emit changeSetting(ch,PulseGenConfig::NameSetting,chw.nameEdit->text());


        ui->pulsePlot->newConfig(d_config);
    }

    chw.nameEdit->setParent(this);
    chw.nameEdit->hide();
    chw.levelButton->setParent(this);
    chw.levelButton->hide();
    chw.delayStepBox->setParent(this);
    chw.delayStepBox->hide();
    chw.widthStepBox->setParent(this);
    chw.widthStepBox->hide();
    chw.roleBox->setParent(this);
    chw.roleBox->hide();

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
        d_widgetList.at(index).levelButton->setChecked(val == QVariant(PulseGenConfig::ActiveHigh));
        break;
    case PulseGenConfig::EnabledSetting:
        d_widgetList.at(index).onButton->setChecked(val.toBool());
        break;
    case PulseGenConfig::NameSetting:
    case PulseGenConfig::RoleSetting:
    default:
        break;
    }

    d_config.set(index,s,val);
    blockSignals(false);

    ui->pulsePlot->newSetting(index,s,val);
}

void PulseConfigWidget::setFromConfig(const PulseGenConfig &c)
{
    blockSignals(true);
    for(int i=0; i<c.size(); i++)
    {
        d_config.set(i,PulseGenConfig::DelaySetting,c.at(i).delay);
        d_config.set(i,PulseGenConfig::WidthSetting,c.at(i).width);
        d_config.set(i,PulseGenConfig::LevelSetting,c.at(i).level);
        d_config.set(i,PulseGenConfig::EnabledSetting,c.at(i).enabled);


        d_widgetList.at(i).delayBox->setValue(c.at(i).delay);
        d_widgetList.at(i).widthBox->setValue(c.at(i).width);
        d_widgetList.at(i).levelButton->setChecked(c.at(i).level == PulseGenConfig::ActiveHigh);
        d_widgetList.at(i).onButton->setChecked(c.at(i).enabled);
    }
    d_config.setRepRate(c.repRate());
    ui->repRateBox->setValue(c.repRate());
    blockSignals(false);

    ui->pulsePlot->newConfig(c);
}

void PulseConfigWidget::newRepRate(double r)
{
    ui->repRateBox->blockSignals(true);
    ui->repRateBox->setValue(r);
    ui->repRateBox->blockSignals(false);
    ui->pulsePlot->newRepRate(r);
    d_config.setRepRate(r);
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

        d_config.set(i,PulseGenConfig::RoleSetting,r);
        d_config.set(i,PulseGenConfig::NameSetting,n);

        if(chw.label != nullptr)
            chw.label->setText(n);

        if(chw.nameEdit != nullptr)
            chw.nameEdit->setText(n);

        ui->pulsePlot->newSetting(i,PulseGenConfig::NameSetting,n);

        if(chw.delayStepBox != nullptr)
            chw.delayStepBox->setValue(getArrayValue<double>(BC::Key::PulseWidget::channels,i,
                                                             BC::Key::PulseWidget::delayStep,1.0));

        if(chw.widthStepBox != nullptr)
            chw.widthStepBox->setValue(getArrayValue<double>(BC::Key::PulseWidget::channels,i,
                                                             BC::Key::PulseWidget::widthStep,1.0));

    }

    ui->repRateBox->setRange(s.get(BC::Key::PGen::minRepRate,0.01),s.get(BC::Key::PGen::maxRepRate,1e5));

}

void PulseConfigWidget::setRepRate(const double r)
{
    ui->pulsePlot->newRepRate(r);
    d_config.setRepRate(r);
    emit changeRepRate(r);
}
