#include "pulseconfigwidget.h"
#include "ui_pulseconfigwidget.h"

#include <QSettings>
#include <QPushButton>
#include <QToolButton>
#include <QLineEdit>
#include <QDialog>
#include <QFormLayout>
#include <QDialogButtonBox>

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
    for(int i=0; i<BC_PGEN_NUMCHANNELS; i++)
    {
        s.setArrayIndex(i);
        ChWidgets ch;
        int col = 0;

        ch.label = new QLabel(s.value(QString("name"),QString("Ch%1").arg(i)).toString(),this);
        ch.label->setAlignment(Qt::AlignRight);
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
        connect(ch.delayBox,vc,this,[=](double val){ emit changeSetting(i,BlackChirp::PulseDelay,val); } );
        col++;

        ch.widthBox = new QDoubleSpinBox(this);
        ch.widthBox->setKeyboardTracking(false);
        ch.widthBox->setRange(0.010,100000.0);
        ch.widthBox->setDecimals(3);
        ch.widthBox->setSuffix(QString::fromUtf16(u" µs"));
        ch.widthBox->setValue(s.value(QString("defaultWidth"),0.050).toDouble());
        ch.widthBox->setSingleStep(s.value(QString("widthStep"),1.0).toDouble());
        ui->pulseConfigBoxLayout->addWidget(ch.widthBox,i+1,col,1,1);
        connect(ch.widthBox,vc,this,[=](double val){ emit changeSetting(i,BlackChirp::PulseWidth,val); } );
        col++;

        ch.onButton = new QPushButton(this);
        ch.onButton->setCheckable(true);
        ch.onButton->setChecked(s.value(QString("defaultEnabled"),false).toBool());
        if(ch.onButton->isChecked())
            ch.onButton->setText(QString("On"));
        else
            ch.onButton->setText(QString("Off"));
        ui->pulseConfigBoxLayout->addWidget(ch.onButton,i+1,col,1,1);
        connect(ch.onButton,&QPushButton::toggled,this,[=](bool en){ emit changeSetting(i,BlackChirp::PulseEnabled,en); } );
        connect(ch.onButton,&QPushButton::toggled,this,[=](bool en){
            if(en)
                ch.onButton->setText(QString("On"));
            else
                ch.onButton->setText(QString("Off")); } );
        col++;

        ch.cfgButton = new QToolButton(this);
        ch.cfgButton->setIcon(QIcon(":/icons/configure.png"));
        ui->pulseConfigBoxLayout->addWidget(ch.cfgButton,i+1,col,1,1);
        connect(ch.cfgButton,&QToolButton::clicked,[=](){ launchChannelConfig(i); } );
        col++;

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

        d_widgetList.append(ch);
    }

    ui->pulseConfigBoxLayout->setColumnStretch(0,0);
    ui->pulseConfigBoxLayout->setColumnStretch(1,1);
    ui->pulseConfigBoxLayout->setColumnStretch(2,1);
    ui->pulseConfigBoxLayout->setColumnStretch(3,0);
    ui->pulseConfigBoxLayout->setColumnStretch(4,0);

    connect(ui->repRateBox,vc,this,&PulseConfigWidget::changeRepRate);

    updateHardwareLimits();

    s.endArray();
    s.endGroup();
    s.endGroup();
}

PulseConfigWidget::~PulseConfigWidget()
{
    delete ui;
}

PulseGenConfig PulseConfigWidget::getConfig()
{
    return ui->pulsePlot->config();
}

void PulseConfigWidget::makeInternalConnections()
{
    connect(this,&PulseConfigWidget::changeSetting,ui->pulsePlot,&PulsePlot::newSetting);
}

void PulseConfigWidget::configureLif(double startingDelay)
{
    if(d_widgetList.isEmpty())
        return;

    d_widgetList.at(BC_PGEN_LIFCHANNEL).delayBox->setValue(startingDelay);
    d_widgetList.at(BC_PGEN_LIFCHANNEL).delayBox->setEnabled(false);
    d_widgetList.at(BC_PGEN_LIFCHANNEL).onButton->setChecked(true);
    d_widgetList.at(BC_PGEN_LIFCHANNEL).onButton->setEnabled(false);
    d_widgetList.at(BC_PGEN_LIFCHANNEL).label->setText(QString("LIF"));
    d_widgetList.at(BC_PGEN_LIFCHANNEL).nameEdit->setText(QString("LIF"));
    d_widgetList.at(BC_PGEN_LIFCHANNEL).nameEdit->setEnabled(false);
    ui->pulsePlot->newSetting(BC_PGEN_LIFCHANNEL,BlackChirp::PulseName,QString("LIF"));

}

void PulseConfigWidget::configureChirp()
{
    if(d_widgetList.isEmpty())
        return;

    d_widgetList.at(BC_PGEN_AWGCHANNEL).onButton->setChecked(true);
    d_widgetList.at(BC_PGEN_AWGCHANNEL).onButton->setEnabled(false);
    d_widgetList.at(BC_PGEN_AWGCHANNEL).label->setText(QString("AWG"));
    d_widgetList.at(BC_PGEN_AWGCHANNEL).nameEdit->setText(QString("AWG"));
    d_widgetList.at(BC_PGEN_AWGCHANNEL).nameEdit->setEnabled(false);
    ui->pulsePlot->newSetting(BC_PGEN_AWGCHANNEL,BlackChirp::PulseName,QString("AWG"));
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

    fl->addRow(QString("Channel Name"),chw.nameEdit);
    fl->addRow(QString("Active Level"),chw.levelButton);
    fl->addRow(QString("Delay Step Size"),chw.delayStepBox);
    fl->addRow(QString("Width Step Size"),chw.widthStepBox);

    chw.nameEdit->show();
    chw.levelButton->show();
    chw.delayStepBox->show();
    chw.widthStepBox->show();

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

        chw.label->setText(chw.nameEdit->text());
        s.setValue(QString("name"),chw.nameEdit->text());
        emit changeSetting(ch,BlackChirp::PulseName,chw.nameEdit->text());

        chw.delayBox->setSingleStep(chw.delayStepBox->value());
        s.setValue(QString("delayStep"),chw.delayStepBox->value());

        chw.widthBox->setSingleStep(chw.widthStepBox->value());
        s.setValue(QString("widthStep"),chw.widthStepBox->value());

        if(chw.levelButton->isChecked())
        {
            s.setValue(QString("level"),BlackChirp::PulseLevelActiveHigh);
            emit changeSetting(ch,BlackChirp::PulseLevel,QVariant::fromValue(BlackChirp::PulseLevelActiveHigh));
        }
        else
        {
            s.setValue(QString("level"),BlackChirp::PulseLevelActiveLow);
            emit changeSetting(ch,BlackChirp::PulseLevel,QVariant::fromValue(BlackChirp::PulseLevelActiveLow));
        }

        s.endArray();
        s.endGroup();
        s.endGroup();
        s.sync();
    }

    chw.nameEdit->setParent(this);
    chw.nameEdit->hide();
    chw.levelButton->setParent(this);
    chw.levelButton->hide();
    chw.delayStepBox->setParent(this);
    chw.delayStepBox->hide();
    chw.widthStepBox->setParent(this);
    chw.widthStepBox->hide();

}

void PulseConfigWidget::newSetting(int index, BlackChirp::PulseSetting s, QVariant val)
{
    if(index < 0 || index > d_widgetList.size())
        return;

    blockSignals(true);

    switch(s) {
    case BlackChirp::PulseName:
        d_widgetList.at(index).label->setText(val.toString());
        d_widgetList.at(index).nameEdit->setText(val.toString());
        break;
    case BlackChirp::PulseDelay:
        d_widgetList.at(index).delayBox->setValue(val.toDouble());
        break;
    case BlackChirp::PulseWidth:
        d_widgetList.at(index).widthBox->setValue(val.toDouble());
        break;
    case BlackChirp::PulseLevel:
        d_widgetList.at(index).levelButton->setChecked(val == QVariant(BlackChirp::PulseLevelActiveHigh));
        break;
    case BlackChirp::PulseEnabled:
        d_widgetList.at(index).onButton->setChecked(val.toBool());
        break;
    }

    blockSignals(false);

    ui->pulsePlot->newSetting(index,s,val);
}

void PulseConfigWidget::newConfig(const PulseGenConfig c)
{
    blockSignals(true);
    for(int i=0; i<c.size(); i++)
    {
        d_widgetList.at(i).label->setText(c.at(i).channelName);
        d_widgetList.at(i).nameEdit->setText(c.at(i).channelName);
        d_widgetList.at(i).delayBox->setValue(c.at(i).delay);
        d_widgetList.at(i).widthBox->setValue(c.at(i).width);
        d_widgetList.at(i).levelButton->setChecked(c.at(i).level == BlackChirp::PulseLevelActiveHigh);
        d_widgetList.at(i).onButton->setChecked(c.at(i).enabled);
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
