#include "pulseconfigwidget.h"
#include "ui_pulseconfigwidget.h"

#include <QSettings>
#include <QPushButton>
#include <QToolButton>
#include <QComboBox>
#include <QLineEdit>
#include <QDialog>
#include <QFormLayout>
#include <QDialogButtonBox>
#include <QMessageBox>

PulseConfigWidget::PulseConfigWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PulseConfigWidget)
{
    ui->setupUi(this);    

    auto vc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    QSettings s(QSettings::SystemScope, QApplication::organizationName(), QApplication::applicationName());
    QString subKey = s.value(QString("pGen/subKey"),QString("virtual")).toString();
    s.beginGroup(QString("pGen"));
    s.beginGroup(subKey);
    s.beginReadArray(QString("channels"));
    QWidget *lastFocusWidget = nullptr;
    auto roles = BlackChirp::allPulseRoles();
    for(int i=0; i<BC_PGEN_NUMCHANNELS; i++)
    {
        s.setArrayIndex(i);
        ChWidgets ch;
        int col = 0;

        ch.label = new QLabel(s.value(QString("name"),QString("Ch%1").arg(i)).toString(),this);
        ch.label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        ui->pulseConfigBoxLayout->addWidget(ch.label,i+1,col,1,1);
        col++;

        ch.delayBox = new QDoubleSpinBox(this);
        ch.delayBox->setKeyboardTracking(false);
        ch.delayBox->setRange(0.0,100000.0);
        ch.delayBox->setDecimals(3);
        ch.delayBox->setSuffix(QString::fromUtf16(u" µs"));
        ch.delayBox->setValue(s.value(QString("defaultDelay"),0.0).toDouble());
        ch.delayBox->setSingleStep(s.value(QString("delayStep"),1.0).toDouble());
        ui->pulseConfigBoxLayout->addWidget(ch.delayBox,i+1,col,1,1);
        connect(ch.delayBox,vc,this,[=](double val){ emit changeSetting(i,BlackChirp::PulseDelaySetting,val); } );
        col++;

        ch.widthBox = new QDoubleSpinBox(this);
        ch.widthBox->setKeyboardTracking(false);
        ch.widthBox->setRange(0.010,100000.0);
        ch.widthBox->setDecimals(3);
        ch.widthBox->setSuffix(QString::fromUtf16(u" µs"));
        ch.widthBox->setValue(s.value(QString("defaultWidth"),0.050).toDouble());
        ch.widthBox->setSingleStep(s.value(QString("widthStep"),1.0).toDouble());
        ui->pulseConfigBoxLayout->addWidget(ch.widthBox,i+1,col,1,1);
        connect(ch.widthBox,vc,this,[=](double val){ emit changeSetting(i,BlackChirp::PulseWidthSetting,val); } );
        col++;

        ch.onButton = new QPushButton(this);
        ch.onButton->setCheckable(true);
        ch.onButton->setChecked(s.value(QString("defaultEnabled"),false).toBool());
        if(ch.onButton->isChecked())
            ch.onButton->setText(QString("On"));
        else
            ch.onButton->setText(QString("Off"));
        ui->pulseConfigBoxLayout->addWidget(ch.onButton,i+1,col,1,1);
        connect(ch.onButton,&QPushButton::toggled,this,[=](bool en){ emit changeSetting(i,BlackChirp::PulseEnabledSetting,en); } );
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
        if(static_cast<BlackChirp::PulseActiveLevel>(s.value(QString("level"),BlackChirp::PulseLevelActiveHigh).toInt()) == BlackChirp::PulseLevelActiveHigh)
        {
            ch.levelButton->setChecked(true);
            ch.levelButton->setText(QString("Active High"));
        }
        else
        {
            ch.levelButton->setChecked(false);
            ch.levelButton->setText(QString("Active Low"));
        }
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
        ch.delayStepBox->setValue(s.value(QString("delayStep"),1.0).toDouble());
        ch.delayStepBox->hide();

        ch.widthStepBox = new QDoubleSpinBox(this);
        ch.widthStepBox->setDecimals(3);
        ch.widthStepBox->setRange(0.001,1000.0);
        ch.widthStepBox->setSuffix(QString::fromUtf16(u" µs"));
        ch.widthStepBox->setValue(s.value(QString("widthStep"),1.0).toDouble());
        ch.widthStepBox->hide();

        ch.roleBox = new QComboBox(this);
        for(int i=0; i<roles.size(); i++)
            ch.roleBox->addItem(BlackChirp::getPulseName(roles.at(i)),roles.at(i));
        auto role = static_cast<BlackChirp::PulseRole>(s.value(QString("role"),BlackChirp::NoPulseRole).toInt());
        ch.roleBox->setCurrentIndex(roles.indexOf(role));
        ch.roleBox->hide();
        connect(ch.roleBox,&QComboBox::currentTextChanged,this,[=](QString t) {
            if(t.contains("None"))
            {
                ch.nameEdit->setText(QString("Ch%1").arg(i));
                ch.nameEdit->setEnabled(true);
            }
            else
            {
                ch.nameEdit->setText(t);
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

    updateHardwareLimits();

    s.endArray();
    s.endGroup();
    s.endGroup();

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

    auto channels = d_config.channelsForRole(BlackChirp::LaserPulseRole);
    if(channels.isEmpty())
    {
        QMessageBox::warning(this,QString("Cannot configure LIF pulse"),QString("No channel has been configured for the \"LIF\" role.\n\nPlease select a channel for the LIF role, then refresh this page (go back one page and then come back to this one) in order to proceed."),QMessageBox::Ok,QMessageBox::Ok);
        return;
    }

    auto delay = c.delayRange().first;

    d_config.set(BlackChirp::LifPulseRole,BlackChirp::PulseDelaySetting,delay);
    d_config.set(BlackChirp::LifPulseRole,BlackChirp::PulseEnabledSetting,true);
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

void PulseConfigWidget::configureFtmw(const FtmwConfig c)
{
    if(!c.isEnabled())
        return;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("awg"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    bool awgHasProt = s.value(QString("hasProtectionPulse"),false).toBool();
    bool awgHasAmpEnable = s.value(QString("hasAmpEnablePulse"),false).toBool();
    s.endGroup();
    s.endGroup();

    auto protChannels = d_config.channelsForRole(BlackChirp::ProtPulseRole);
    auto awgChannels = d_config.channelsForRole(BlackChirp::AwgPulseRole);
    auto ampChannels = d_config.channelsForRole(BlackChirp::AmpPulseRole);

    if(!awgHasProt && protChannels.isEmpty())
        QMessageBox::warning(this,QString("Cannot configure protection pulse"),QString("No channel has been configured for the \"Prot\" role, and your AWG does not produce its own protection signal.\n\nBlackchirp cannot guarantee that your receiver amp will be protected!\n\nIf you wish for Blackchirp to generate a protection pulse, select a channel for the Prot role and refresh this page (go back one page and then come back to this one)."),QMessageBox::Ok,QMessageBox::Ok);

    if(!awgHasProt && awgChannels.isEmpty())
        QMessageBox::warning(this,QString("Cannot configure protection pulse"),QString("No channel has been configured for the \"AWG\" role, and your AWG does not produce its own protection signal.\n\nBlackchirp cannot guarantee that your receiver amp will be protected because it does not know when your AWG is triggered!\n\nIf you wish for Blackchirp to generate a protection pulse, select a channel for the AWG role and refresh this page (go back one page and then come back to this one)."),QMessageBox::Ok,QMessageBox::Ok);

    if(!awgHasProt && c.chirpConfig(0).numChirps() > 1)
        QMessageBox::warning(this,QString("Warning: multiple chirps"),QString("You have requested multiple chirps, and your AWG cannot generate its own protection signal.\nBlackchirp does not know how to configure your delay generator in a burst mode to generate a protection signal with each chirp.\n\nProceed at your own risk."),QMessageBox::Ok,QMessageBox::Ok);

    if(d_widgetList.isEmpty())
        return;

    auto cc = c.chirpConfig();
    d_config.set(BlackChirp::AwgPulseRole,BlackChirp::PulseEnabledSetting,true);
    auto l = d_config.setting(BlackChirp::AwgPulseRole,BlackChirp::PulseDelaySetting);

    if(l.size() > 1)
    {
        d_config.set(BlackChirp::AwgPulseRole,BlackChirp::PulseDelaySetting,l.constFirst());
        d_config.set(BlackChirp::AwgPulseRole,BlackChirp::PulseWidthSetting,d_config.setting(BlackChirp::AmpPulseRole,BlackChirp::PulseWidthSetting).constFirst().toDouble());
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
                d_config.set(BlackChirp::AwgPulseRole,BlackChirp::PulseDelaySetting,awgStart);
                protStart = 0.0;
            }

            double protWidth = cc.totalProtectionWidth();

            d_config.set(BlackChirp::ProtPulseRole,BlackChirp::PulseDelaySetting,protStart);
            d_config.set(BlackChirp::ProtPulseRole,BlackChirp::PulseWidthSetting,protWidth);
            d_config.set(BlackChirp::ProtPulseRole,BlackChirp::PulseEnabledSetting,true);
        }

        bool checkProt = false;
        if(!awgHasAmpEnable)
        {
            double gateStart = awgStart - cc.preChirpGateDelay();
            if(gateStart < 0.0)
            {
                awgStart -= gateStart;
                d_config.set(BlackChirp::AwgPulseRole,BlackChirp::PulseDelaySetting,awgStart);
                gateStart = 0.0;
                checkProt = true;
            }

            double gateWidth = cc.totalGateWidth();

            d_config.set(BlackChirp::AmpPulseRole,BlackChirp::PulseDelaySetting,gateStart);
            d_config.set(BlackChirp::AmpPulseRole,BlackChirp::PulseWidthSetting,gateWidth);
            d_config.set(BlackChirp::AmpPulseRole,BlackChirp::PulseEnabledSetting,true);
        }

        if(!awgHasProt && checkProt)
        {
            double protStart = awgStart - cc.preChirpProtectionDelay() - cc.preChirpGateDelay();
            double protWidth = cc.totalProtectionWidth();

            d_config.set(BlackChirp::ProtPulseRole,BlackChirp::PulseDelaySetting,protStart);
            d_config.set(BlackChirp::ProtPulseRole,BlackChirp::PulseWidthSetting,protWidth);
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
        QSettings s(QSettings::SystemScope, QApplication::organizationName(), QApplication::applicationName());
        QString subKey = s.value(QString("pGen/subKey"),QString("virtual")).toString();
        s.beginGroup(QString("pGen"));;
        s.beginGroup(subKey);
        s.beginWriteArray(QString("channels"));
        s.setArrayIndex(ch);

        chw.delayBox->setSingleStep(chw.delayStepBox->value());
        s.setValue(QString("delayStep"),chw.delayStepBox->value());

        chw.widthBox->setSingleStep(chw.widthStepBox->value());
        s.setValue(QString("widthStep"),chw.widthStepBox->value());

        if(chw.levelButton->isChecked())
        {
            s.setValue(QString("level"),BlackChirp::PulseLevelActiveHigh);
            d_config.set(ch,BlackChirp::PulseLevelSetting,QVariant::fromValue(BlackChirp::PulseLevelActiveHigh));
            emit changeSetting(ch,BlackChirp::PulseLevelSetting,QVariant::fromValue(BlackChirp::PulseLevelActiveHigh));
        }
        else
        {
            s.setValue(QString("level"),BlackChirp::PulseLevelActiveLow);
            d_config.set(ch,BlackChirp::PulseLevelSetting,QVariant::fromValue(BlackChirp::PulseLevelActiveLow));
            emit changeSetting(ch,BlackChirp::PulseLevelSetting,QVariant::fromValue(BlackChirp::PulseLevelActiveLow));
        }

        s.setValue(QString("role"),chw.roleBox->currentData());
        d_config.set(ch,BlackChirp::PulseRoleSetting,static_cast<BlackChirp::PulseRole>(chw.roleBox->currentData().toInt()));
        emit changeSetting(ch,BlackChirp::PulseRoleSetting,d_config.at(ch).role);

        chw.label->setText(chw.nameEdit->text());
        s.setValue(QString("name"),chw.nameEdit->text());
        d_config.set(ch,BlackChirp::PulseNameSetting,chw.nameEdit->text());
        emit changeSetting(ch,BlackChirp::PulseNameSetting,chw.nameEdit->text());


        s.endArray();
        s.endGroup();
        s.endGroup();
        s.sync();

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

void PulseConfigWidget::newSetting(int index, BlackChirp::PulseSetting s, QVariant val)
{
    if(index < 0 || index > d_widgetList.size())
        return;

    blockSignals(true);

    switch(s) {
    case BlackChirp::PulseNameSetting:
        d_widgetList.at(index).label->setText(val.toString());
        d_widgetList.at(index).nameEdit->setText(val.toString());
        break;
    case BlackChirp::PulseDelaySetting:
        d_widgetList.at(index).delayBox->setValue(val.toDouble());
        break;
    case BlackChirp::PulseWidthSetting:
        d_widgetList.at(index).widthBox->setValue(val.toDouble());
        break;
    case BlackChirp::PulseLevelSetting:
        d_widgetList.at(index).levelButton->setChecked(val == QVariant(BlackChirp::PulseLevelActiveHigh));
        break;
    case BlackChirp::PulseEnabledSetting:
        d_widgetList.at(index).onButton->setChecked(val.toBool());
        break;
    case BlackChirp::PulseRoleSetting:
        d_widgetList.at(index).roleBox->setCurrentIndex(d_widgetList.at(index).roleBox->findData(val));
        break;
    }

    d_config.set(index,s,val);
    blockSignals(false);

    ui->pulsePlot->newSetting(index,s,val);
}

void PulseConfigWidget::setFromConfig(const PulseGenConfig c)
{
    d_config = c;
    blockSignals(true);
    for(int i=0; i<c.size(); i++)
    {
        d_widgetList.at(i).label->setText(c.at(i).channelName);
        d_widgetList.at(i).nameEdit->setText(c.at(i).channelName);
        d_widgetList.at(i).delayBox->setValue(c.at(i).delay);
        d_widgetList.at(i).widthBox->setValue(c.at(i).width);
        d_widgetList.at(i).levelButton->setChecked(c.at(i).level == BlackChirp::PulseLevelActiveHigh);
        d_widgetList.at(i).onButton->setChecked(c.at(i).enabled);
        d_widgetList.at(i).roleBox->setCurrentIndex(d_widgetList.at(i).roleBox->findData(c.at(i).role));
    }
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

void PulseConfigWidget::updateHardwareLimits()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("pGen"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    double minWidth = s.value(QString("minWidth"),0.01).toDouble();
    double maxWidth = s.value(QString("maxWidth"),1e5).toDouble();
    double minDelay = s.value(QString("minDelay"),0.0).toDouble();
    double maxDelay = s.value(QString("maxDelay"),1e5).toDouble();
    s.endGroup();
    s.endGroup();

    for(int i=0; i<d_widgetList.size(); i++)
    {
        QDoubleSpinBox *wid = d_widgetList.at(i).widthBox;
        QDoubleSpinBox *del = d_widgetList.at(i).delayBox;

        if(del != nullptr)
        {
            del->blockSignals(true);
            del->setRange(minDelay,maxDelay);
            del->blockSignals(false);
        }

        if(wid != nullptr)
        {
            wid->blockSignals(true);
            wid->setRange(minWidth,maxWidth);
            wid->blockSignals(false);
        }
    }

}

void PulseConfigWidget::setRepRate(const double r)
{
    ui->pulsePlot->newRepRate(r);
    d_config.setRepRate(r);
    emit changeRepRate(r);
}
