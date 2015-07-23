#include "ftmwviewwidget.h"
#include "ui_ftmwviewwidget.h"

#include <QThread>

#include "ftworker.h"

FtmwViewWidget::FtmwViewWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FtmwViewWidget), d_mode(BlackChirp::FtmwViewLive), d_replotWhenDone(false), d_processing(false), d_pzf(0)
{
    ui->setupUi(this);

    connect(ui->frameBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::showFrame);

    ui->peakUpControlBox->hide();

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
    connect(ui->controlButton,&QToolButton::toggled,this,&FtmwViewWidget::togglePanel);
    togglePanel(false);
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

    ui->liveUpdateButton->blockSignals(true);
    ui->liveUpdateButton->setChecked(true);
    d_mode = BlackChirp::FtmwViewLive;
    ui->liveUpdateButton->blockSignals(false);

    ui->snapDiffButton->setEnabled(false);
    ui->snapshotCheckbox->setEnabled(false);

    d_currentFidList.clear();
    if(config.isEnabled())
    {
        ui->controlButton->setEnabled(true);

        ui->frameBox->blockSignals(true);
        ui->refFrameSpinBox->blockSignals(true);
        int frames = config.scopeConfig().numFrames;
        if(config.scopeConfig().summaryFrame)
            frames = 1;
        if(ui->frameBox->value() > frames)
            ui->frameBox->setValue(1);
        if(ui->refFrameSpinBox->value() > frames)
            ui->refFrameSpinBox->setValue(1);


        ui->frameBox->setRange(1,frames);
        ui->refFrameSpinBox->setRange(1,frames);
        if(frames == 1)
        {
            ui->frameControlBox->setEnabled(false);
            ui->frameDiffButton->setEnabled(false);
        }
        else
        {
            ui->frameDiffButton->setEnabled(true);
            ui->frameControlBox->setEnabled(true);
            ui->frameBox->setEnabled(true);
        }

        ui->frameBox->blockSignals(false);
        ui->refFrameSpinBox->blockSignals(false);

        if(config.type() == BlackChirp::FtmwPeakUp)
        {
            ui->peakUpControlBox->show();

            blockSignals(true);
            ui->rollingAverageSpinbox->setValue(config.targetShots());
            blockSignals(false);

            connect(ui->rollingAverageSpinbox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::rollingAverageShotsChanged,Qt::UniqueConnection);
            connect(ui->rollingAverageResetButton,&QPushButton::clicked,this,&FtmwViewWidget::rollingAverageReset,Qt::UniqueConnection);
        }
        else
        {
            ui->peakUpControlBox->hide();

            disconnect(ui->rollingAverageSpinbox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::rollingAverageShotsChanged);
            disconnect(ui->rollingAverageResetButton,&QPushButton::clicked,this,&FtmwViewWidget::rollingAverageReset);
        }
    }
    else
    {        
        ui->frameBox->blockSignals(true);
        ui->frameBox->setRange(0,0);
        ui->frameBox->setValue(0);
        ui->frameBox->blockSignals(false);
        ui->frameControlBox->setEnabled(false);

        ui->peakUpControlBox->hide();

        ui->controlButton->blockSignals(true);
        ui->controlButton->setChecked(false);
        togglePanel(false);
        ui->controlButton->blockSignals(false);
        ui->controlButton->setEnabled(false);

        disconnect(ui->rollingAverageSpinbox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::rollingAverageShotsChanged);
        disconnect(ui->rollingAverageResetButton,&QPushButton::clicked,this,&FtmwViewWidget::rollingAverageReset);

    }

}

void FtmwViewWidget::togglePanel(bool on)
{
    for(int i=0; i<ui->rightLayout->count(); i++)
    {
        QWidget *w = ui->rightLayout->itemAt(i)->widget();
        if(w != nullptr)
            w->setVisible(on);
    }
}

void FtmwViewWidget::newFidList(QList<Fid> fl)
{
    d_currentFidList = fl;
    if(ui->frameBox->maximum() != d_currentFidList.size())
    {
        if(ui->frameBox->value() > d_currentFidList.size())
        {
            blockSignals(true);
            ui->frameBox->setValue(d_currentFidList.size());
            blockSignals(false);
        }
        ui->frameBox->setMaximum(d_currentFidList.size());
    }
    showFrame(ui->frameBox->value());
}

void FtmwViewWidget::updateShotsLabel(qint64 shots)
{
    ui->shotsLabel->setText(QString("Shots: %1").arg(shots));
}

void FtmwViewWidget::showFrame(int num)
{
    if(d_currentFidList.size() <= num)
        return;

    d_currentFid = d_currentFidList.at(num-1);
    //process FID
    if(!d_processing)
        updateFtPlot();
    else
        d_replotWhenDone = true;

    ui->fidPlot->receiveData(d_currentFid);
}

void FtmwViewWidget::ftStartChanged(double s)
{
    QMetaObject::invokeMethod(p_ftw,"setStart",Q_ARG(double,s));
    if(d_currentFidList.isEmpty())
        return;

    if(!d_processing)
        updateFtPlot();
    else
        d_replotWhenDone = true;
}

void FtmwViewWidget::ftEndChanged(double e)
{
    QMetaObject::invokeMethod(p_ftw,"setEnd",Q_ARG(double,e));
    if(d_currentFidList.isEmpty())
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
    if(d_currentFidList.isEmpty())
        return;

    if(!d_processing)
        updateFtPlot();
    else
        d_replotWhenDone = true;
}

void FtmwViewWidget::updateFtPlot()
{
    if(d_currentFidList.size() >= ui->frameBox->value())
    {
        QMetaObject::invokeMethod(p_ftw,"doFT",Q_ARG(const Fid,d_currentFid));
        d_processing = true;
        d_replotWhenDone = false;
    }
}

void FtmwViewWidget::ftDone(QVector<QPointF> ft, double max)
{
    d_processing = false;
    ui->ftPlot->newFt(ft,max);

    if(d_replotWhenDone)
        updateFtPlot();
}

