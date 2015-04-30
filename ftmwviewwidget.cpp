#include "ftmwviewwidget.h"
#include "ui_ftmwviewwidget.h"

#include "ftmwconfig.h"

FtmwViewWidget::FtmwViewWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FtmwViewWidget)
{
    ui->setupUi(this);

    connect(ui->frameBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::showFrame);

    ui->rollingAverageLabel->hide();
    ui->rollingAverageSpinbox->hide();
    ui->rollingAverageResetButton->hide();
}

FtmwViewWidget::~FtmwViewWidget()
{
    delete ui;
}

void FtmwViewWidget::initializeForExperiment(const FtmwConfig config)
{
    ui->shotsLabel->setText(QString("Shots: 0"));

    if(config.isEnabled() && config.type() == BlackChirp::FtmwPeakUp)
    {
        ui->rollingAverageLabel->show();
        ui->rollingAverageSpinbox->show();
        ui->rollingAverageResetButton->show();

        blockSignals(true);
        ui->rollingAverageSpinbox->setValue(config.targetShots());
        blockSignals(false);

        connect(ui->rollingAverageSpinbox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::rollingAverageShotsChanged,Qt::UniqueConnection);
        connect(ui->rollingAverageResetButton,&QPushButton::clicked,this,&FtmwViewWidget::rollingAverageReset,Qt::UniqueConnection);
    }
    else
    {
        ui->rollingAverageLabel->hide();
        ui->rollingAverageSpinbox->hide();
        ui->rollingAverageResetButton->hide();

        disconnect(ui->rollingAverageSpinbox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::rollingAverageShotsChanged);
        disconnect(ui->rollingAverageResetButton,&QPushButton::clicked,this,&FtmwViewWidget::rollingAverageReset);
    }
}

void FtmwViewWidget::newFidList(QList<Fid> fl)
{
    d_fidList = fl;
    if(ui->frameBox->maximum() != d_fidList.size())
    {
        if(ui->frameBox->value() > d_fidList.size())
        {
            blockSignals(true);
            ui->frameBox->setValue(d_fidList.size());
            blockSignals(false);
        }
        ui->frameBox->setMaximum(d_fidList.size());
    }
    showFrame(ui->frameBox->value());
}

void FtmwViewWidget::updateShotsLabel(qint64 shots)
{
    ui->shotsLabel->setText(QString("Shots: %1").arg(shots));
}

void FtmwViewWidget::showFrame(int num)
{
    ui->ftPlot->newFid(d_fidList.at(num-1));
    ui->fidPlot->receiveData(d_fidList.at(num-1));
}

void FtmwViewWidget::fidTest(Fid f)
{
    ui->ftPlot->newFid(f);
}
