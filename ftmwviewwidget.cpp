#include "ftmwviewwidget.h"
#include "ui_ftmwviewwidget.h"

#include <QThread>

#include "ftworker.h"

FtmwViewWidget::FtmwViewWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FtmwViewWidget), d_replotWhenDone(false), d_processing(false), d_pzf(0)
{
    ui->setupUi(this);

    connect(ui->frameBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::showFrame);

    ui->rollingAverageLabel->hide();
    ui->rollingAverageSpinbox->hide();
    ui->rollingAverageResetButton->hide();

    connect(ui->fidPlot,&FidPlot::ftStartChanged,this,&FtmwViewWidget::ftStartChanged);
    connect(ui->fidPlot,&FidPlot::ftEndChanged,this,&FtmwViewWidget::ftEndChanged);
    ui->exptLabel->setVisible(false);

    p_ftw = new FtWorker();
    //make signal/slot connections
    connect(p_ftw,&FtWorker::ftDone,this,&FtmwViewWidget::ftDone);
    p_ftThread = new QThread(this);
    connect(p_ftThread,&QThread::finished,p_ftw,&FtWorker::deleteLater);
    p_ftw->moveToThread(p_ftThread);
    p_ftThread->start();

    connect(ui->ftPlot,&FtPlot::pzfChanged,this,&FtmwViewWidget::pzfChanged);
}

FtmwViewWidget::~FtmwViewWidget()
{
    p_ftThread->quit();
    p_ftThread->wait();

    delete ui;
}

void FtmwViewWidget::prepareForExperiment(const Experiment e)
{
    FtmwConfig config = e.ftmwConfig();
    ui->exptLabel->setText(QString("Experiment %1").arg(e.number()));
    if(!ui->exptLabel->isVisible())
        ui->exptLabel->setVisible(true);

    ui->shotsLabel->setText(QString("Shots: 0"));

    ui->fidPlot->prepareForExperiment(config);
    ui->ftPlot->prepareForExperiment(e);

    ui->frameBox->blockSignals(true);
    d_fidList.clear();
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
    //process FID
    if(!d_processing)
        updateFtPlot();
    else
        d_replotWhenDone = true;

    ui->fidPlot->receiveData(d_fidList.at(num-1));
}

void FtmwViewWidget::ftStartChanged(double s)
{
    QMetaObject::invokeMethod(p_ftw,"setStart",Q_ARG(double,s));
    if(d_fidList.isEmpty())
        return;

    if(!d_processing)
        updateFtPlot();
    else
        d_replotWhenDone = true;
}

void FtmwViewWidget::ftEndChanged(double e)
{
    QMetaObject::invokeMethod(p_ftw,"setEnd",Q_ARG(double,e));
    if(d_fidList.isEmpty())
        return;

    if(!d_processing)
        updateFtPlot();
    else
        d_replotWhenDone = true;
}

void FtmwViewWidget::pzfChanged(int zpf)
{
    d_pzf = zpf;
    QMetaObject::invokeMethod(p_ftw,"setPzf",Q_ARG(int,zpf));
    if(d_fidList.isEmpty())
        return;

    if(!d_processing)
        updateFtPlot();
    else
        d_replotWhenDone = true;
}

void FtmwViewWidget::updateFtPlot()
{
    QMetaObject::invokeMethod(p_ftw,"doFT",Q_ARG(const Fid,d_fidList.at(ui->frameBox->value()-1)));
    d_processing = true;
    d_replotWhenDone = false;
}

void FtmwViewWidget::ftDone(QVector<QPointF> ft, double max)
{
    d_processing = false;
    ui->ftPlot->newFt(ft,max);

    if(d_replotWhenDone)
        updateFtPlot();
}

