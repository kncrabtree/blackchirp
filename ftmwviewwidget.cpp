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

    connect(ui->fidPlot,&FidPlot::ftStartChanged,ui->ftPlot,&FtPlot::ftStartChanged);
    connect(ui->fidPlot,&FidPlot::ftEndChanged,ui->ftPlot,&FtPlot::ftEndChanged);
}

FtmwViewWidget::~FtmwViewWidget()
{
    delete ui;
}

void FtmwViewWidget::prepareForExperiment(const FtmwConfig config)
{
    ui->shotsLabel->setText(QString("Shots: 0"));

    ui->fidPlot->prepareForExperiment(config);
    ui->ftPlot->prepareForExperiment(config);

    ui->frameBox->blockSignals(true);
    if(config.isEnabled())
    {
        int frames = config.scopeConfig().numFrames;
        if(config.scopeConfig().summaryFrame)
            frames = 1;
        if(ui->frameBox->value() > frames)
            ui->frameBox->setValue(1);

        ui->frameBox->setRange(1,frames);

        if(config.type() == BlackChirp::FtmwPeakUp)
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
    else
    {
        ui->frameBox->setRange(0,0);
        ui->frameBox->setValue(0);

        ui->rollingAverageLabel->hide();
        ui->rollingAverageSpinbox->hide();
        ui->rollingAverageResetButton->hide();

        disconnect(ui->rollingAverageSpinbox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::rollingAverageShotsChanged);
        disconnect(ui->rollingAverageResetButton,&QPushButton::clicked,this,&FtmwViewWidget::rollingAverageReset);

    }

    ui->frameBox->blockSignals(false);


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
