#include "ftmwviewwidget.h"
#include "ui_ftmwviewwidget.h"

#include <QThread>
#include <QMessageBox>

#include "ftworker.h"
#include "ftmwsnapshotwidget.h"

FtmwViewWidget::FtmwViewWidget(QWidget *parent, QString path) :
    QWidget(parent),
    ui(new Ui::FtmwViewWidget), d_mode(BlackChirp::FtmwViewLive), d_replotWhenDone(false), d_processing(false),
    d_pzf(0), d_currentExptNum(-1), p_snapWidget(nullptr), d_path(path)
{
    ui->setupUi(this);

    connect(ui->frameBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::updateFtPlot);
    connect(ui->refFrameSpinBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::updateFtPlot);

    ui->peakUpControlBox->hide();

    connect(ui->fidPlot,&FidPlot::ftStartChanged,this,&FtmwViewWidget::ftStartChanged);
    connect(ui->fidPlot,&FidPlot::ftEndChanged,this,&FtmwViewWidget::ftEndChanged);
    connect(ui->fidPlot,&FidPlot::removeDcChanged,this,&FtmwViewWidget::removeDcChanged);
    connect(ui->fidPlot,&FidPlot::showProcessedChanged,this,&FtmwViewWidget::showProcessedChanged);
    ui->exptLabel->setVisible(false);

    p_ftw = new FtWorker();
    //make signal/slot connections
    connect(p_ftw,&FtWorker::ftDone,this,&FtmwViewWidget::ftDone);
    connect(p_ftw,&FtWorker::ftDiffDone,this,&FtmwViewWidget::ftDiffDone);
    connect(p_ftw,&FtWorker::fidDone,ui->fidPlot,&FidPlot::receiveProcessedFid);
    p_ftThread = new QThread(this);
    connect(p_ftThread,&QThread::finished,p_ftw,&FtWorker::deleteLater);
    p_ftw->moveToThread(p_ftThread);
    p_ftThread->start();

    connect(ui->ftPlot,&FtPlot::pzfChanged,this,&FtmwViewWidget::pzfChanged);
    connect(ui->ftPlot,&FtPlot::unitsChanged,this,&FtmwViewWidget::scalingChanged);
    connect(ui->ftPlot,&FtPlot::scalingChange,ui->peakFindWidget,&PeakFindWidget::changeScaleFactor);
    connect(ui->ftPlot,&FtPlot::winfChanged,this,&FtmwViewWidget::winfChanged);
    connect(ui->controlButton,&QToolButton::toggled,this,&FtmwViewWidget::togglePanel);
    togglePanel(false);

    connect(ui->liveUpdateButton,&QRadioButton::clicked,this,&FtmwViewWidget::modeChanged);
    connect(ui->singleFrameButton,&QRadioButton::clicked,this,&FtmwViewWidget::modeChanged);
    connect(ui->frameDiffButton,&QRadioButton::clicked,this,&FtmwViewWidget::modeChanged);
    connect(ui->snapDiffButton,&QRadioButton::clicked,this,&FtmwViewWidget::modeChanged);
    connect(ui->snapshotCheckbox,&QCheckBox::toggled,this,&FtmwViewWidget::modeChanged);

    connect(ui->peakFindWidget,&PeakFindWidget::peakList,ui->ftPlot,&FtPlot::newPeakList);
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
    if(config.type() == BlackChirp::FtmwPeakUp)
        ui->exptLabel->setText(QString("Peak Up Mode"));
    else
        ui->exptLabel->setText(QString("Experiment %1").arg(e.number()));

    if(!ui->exptLabel->isVisible())
        ui->exptLabel->setVisible(true);

    ui->shotsLabel->setText(QString("Shots: 0"));

    ui->fidPlot->prepareForExperiment(e);
    ui->peakFindWidget->prepareForExperiment(e);
    ui->ftPlot->prepareForExperiment(e);

    ui->liveUpdateButton->blockSignals(true);
    ui->liveUpdateButton->setEnabled(true);
    ui->liveUpdateButton->setChecked(true);
    d_mode = BlackChirp::FtmwViewLive;
    ui->liveUpdateButton->blockSignals(false);

    ui->snapDiffButton->setEnabled(false);
    ui->snapshotCheckbox->setEnabled(false);

    if(p_snapWidget != nullptr)
    {
        p_snapWidget->setEnabled(false);
        p_snapWidget->deleteLater();
        p_snapWidget = nullptr;
    }

    d_currentFidList.clear();
    if(config.isEnabled())
    {
        d_currentExptNum = e.number();
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
        d_currentExptNum = -1;

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

void FtmwViewWidget::newFidList(FidList fl)
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
    if(d_mode == BlackChirp::FtmwViewLive)
        updateFtPlot();

    if(p_snapWidget != nullptr)
        p_snapWidget->setFidList(fl);
}

void FtmwViewWidget::updateShotsLabel(qint64 shots)
{
    ui->shotsLabel->setText(QString("Shots: %1").arg(shots));
}

void FtmwViewWidget::ftStartChanged(double s)
{
    QMetaObject::invokeMethod(p_ftw,"setStart",Q_ARG(double,s));
    updateFtPlot();
}

void FtmwViewWidget::ftEndChanged(double e)
{
    QMetaObject::invokeMethod(p_ftw,"setEnd",Q_ARG(double,e));
    updateFtPlot();
}

void FtmwViewWidget::removeDcChanged(bool rdc)
{
    QMetaObject::invokeMethod(p_ftw,"setRemoveDc",Q_ARG(bool,rdc));
    updateFtPlot();
}

void FtmwViewWidget::showProcessedChanged(bool p)
{
    QMetaObject::invokeMethod(p_ftw,"setShowProcessed",Q_ARG(bool,p));
    updateFtPlot();
}

void FtmwViewWidget::pzfChanged(int zpf)
{
    d_pzf = zpf;
    QMetaObject::invokeMethod(p_ftw,"setPzf",Q_ARG(int,zpf));
    updateFtPlot();
}

void FtmwViewWidget::scalingChanged(double scf)
{
    QMetaObject::invokeMethod(p_ftw,"setScaling",Q_ARG(double,scf));
    updateFtPlot();
}

void FtmwViewWidget::winfChanged(BlackChirp::FtWindowFunction f)
{
    QMetaObject::invokeMethod(p_ftw,"setWindowFunction",Q_ARG(BlackChirp::FtWindowFunction,f));
    updateFtPlot();
}

void FtmwViewWidget::updateFtPlot()
{
    if(d_processing)
    {
        d_replotWhenDone = true;
        return;
    }

    if(d_currentFidList.isEmpty())
        return;

    if(d_mode == BlackChirp::FtmwViewLive)
    {
        if(d_currentFidList.size() >= ui->frameBox->value())
            d_currentFid = d_currentFidList.at(ui->frameBox->value()-1);
        else
            d_currentFid = d_currentFidList.first();

        QMetaObject::invokeMethod(p_ftw,"doFT",Q_ARG(const Fid,d_currentFid));
        d_processing = true;
        d_replotWhenDone = false;
        ui->fidPlot->receiveData(d_currentFid);
    }
    else if(d_mode == BlackChirp::FtmwViewSingle)
    {
        if(ui->snapshotCheckbox->isChecked() && p_snapWidget != nullptr)
            d_currentFid = p_snapWidget->getSnapFid(ui->frameBox->value()-1);
        else
            d_currentFid = d_currentFidList.at(ui->frameBox->value()-1);

        QMetaObject::invokeMethod(p_ftw,"doFT",Q_ARG(const Fid,d_currentFid));
        d_processing = true;
        d_replotWhenDone = false;
        ui->fidPlot->receiveData(d_currentFid);
    }
    else if(d_mode == BlackChirp::FtmwViewFrameDiff)
    {
        if(ui->snapshotCheckbox->isChecked() && p_snapWidget != nullptr)
        {
            d_currentFid = p_snapWidget->getSnapFid(ui->frameBox->value()-1);
            d_currentRefFid = p_snapWidget->getSnapFid(ui->refFrameSpinBox->value()-1);
        }
        else
        {
            d_currentFid = d_currentFidList.at(ui->frameBox->value()-1);
            d_currentRefFid = d_currentFidList.at(ui->refFrameSpinBox->value()-1);
        }

        QMetaObject::invokeMethod(p_ftw,"doFtDiff",Q_ARG(const Fid,d_currentRefFid),Q_ARG(const Fid,d_currentFid));
        d_processing = true;
        d_replotWhenDone = false;
        ui->fidPlot->receiveData(d_currentFid);
    }
    else if(d_mode == BlackChirp::FtmwViewSnapDiff)
    {
        //note that the current and ref fids are set on mode change and then from signals from snap widget
        QMetaObject::invokeMethod(p_ftw,"doFtDiff",Q_ARG(const Fid,d_currentRefFid),Q_ARG(const Fid,d_currentFid));
        d_processing = true;
        d_replotWhenDone = false;
        ui->fidPlot->receiveData(d_currentFid);
    }
}

void FtmwViewWidget::ftDone(QVector<QPointF> ft, double max)
{
    d_processing = false;
    ui->ftPlot->newFt(ft,max);
    ui->peakFindWidget->newFt(ft);

    if(d_replotWhenDone)
        updateFtPlot();
}

void FtmwViewWidget::ftDiffDone(QVector<QPointF> ft, double min, double max)
{
    d_processing = false;
    ui->ftPlot->newFtDiff(ft,min,max);

    if(d_replotWhenDone)
        updateFtPlot();
}

void FtmwViewWidget::modeChanged()
{
    if(ui->liveUpdateButton->isChecked())
    {
        d_mode = BlackChirp::FtmwViewLive;
        ui->refFrameSpinBox->setEnabled(false);
        ui->snapshotCheckbox->blockSignals(true);
        ui->snapshotCheckbox->setChecked(false);
        ui->snapshotCheckbox->blockSignals(false);
        ui->snapshotCheckbox->setEnabled(false);

        if(p_snapWidget != nullptr)
        {
            p_snapWidget->setSelectionEnabled(false);
            p_snapWidget->setDiffMode(false);
        }

    }
    else if(ui->singleFrameButton->isChecked())
    {
        d_mode = BlackChirp::FtmwViewSingle;
        ui->refFrameSpinBox->setEnabled(false);

        if(p_snapWidget != nullptr)
        {
            ui->snapshotCheckbox->setEnabled(true);
            p_snapWidget->setSelectionEnabled(ui->snapshotCheckbox->isChecked());
            p_snapWidget->setDiffMode(false);
        }
        else
        {
            ui->snapshotCheckbox->blockSignals(true);
            ui->snapshotCheckbox->setChecked(false);
            ui->snapshotCheckbox->blockSignals(false);
            ui->snapshotCheckbox->setEnabled(false);
        }

    }
    else if(ui->frameDiffButton->isChecked())
    {
        d_mode = BlackChirp::FtmwViewFrameDiff;
        ui->refFrameSpinBox->setEnabled(true);

        if(p_snapWidget != nullptr)
        {
            ui->snapshotCheckbox->setEnabled(true);
            p_snapWidget->setSelectionEnabled(ui->snapshotCheckbox->isChecked());
            p_snapWidget->setDiffMode(false);
        }
        else
        {
            ui->snapshotCheckbox->blockSignals(true);
            ui->snapshotCheckbox->setChecked(false);
            ui->snapshotCheckbox->blockSignals(false);
            ui->snapshotCheckbox->setEnabled(false);
        }
    }
    else if(ui->snapDiffButton->isChecked())
    {
        d_mode = BlackChirp::FtmwViewSnapDiff;
        ui->snapshotCheckbox->blockSignals(true);
        ui->snapshotCheckbox->setChecked(false);
        ui->snapshotCheckbox->blockSignals(false);
        ui->snapshotCheckbox->setEnabled(false);

        Q_ASSERT(p_snapWidget != nullptr);
        p_snapWidget->setSelectionEnabled(false);
        p_snapWidget->setDiffMode(true);

        d_currentFid = p_snapWidget->getDiffFid(ui->frameBox->value()-1);
        d_currentRefFid = p_snapWidget->getRefFid(ui->frameBox->value()-1);
    }

    updateFtPlot();
}

void FtmwViewWidget::snapshotTaken()
{
    if(d_currentExptNum < 1)
        return;

    if(p_snapWidget == nullptr)
    {
        p_snapWidget = new FtmwSnapshotWidget(d_currentExptNum,d_path,this);
        connect(p_snapWidget,&FtmwSnapshotWidget::loadFailed,this,&FtmwViewWidget::snapshotLoadError);
        connect(p_snapWidget,&FtmwSnapshotWidget::snapListChanged,this,&FtmwViewWidget::snapListUpdate);
        connect(p_snapWidget,&FtmwSnapshotWidget::refChanged,this,&FtmwViewWidget::snapRefChanged);
        connect(p_snapWidget,&FtmwSnapshotWidget::diffChanged,this,&FtmwViewWidget::snapDiffChanged);
        connect(p_snapWidget,&FtmwSnapshotWidget::finalizedList,this,&FtmwViewWidget::finalizedSnapList);
        connect(p_snapWidget,&FtmwSnapshotWidget::experimentLogMessage,this,&FtmwViewWidget::experimentLogMessage);
        p_snapWidget->setSelectionEnabled(false);
        p_snapWidget->setDiffMode(false);
        p_snapWidget->setFinalizeEnabled(false);
        if(!ui->controlFrame->isVisible())
            p_snapWidget->setVisible(false);
        ui->rightLayout->addWidget(p_snapWidget);
    }

    p_snapWidget->setFidList(d_currentFidList);

    if(p_snapWidget->readSnapshots())
        ui->snapDiffButton->setEnabled(true);

}

void FtmwViewWidget::experimentComplete()
{
    if(d_mode == BlackChirp::FtmwViewLive)
    {
        ui->singleFrameButton->setChecked(true);
        modeChanged();
    }

    ui->liveUpdateButton->setEnabled(false);

    if(p_snapWidget != nullptr)
    {
        p_snapWidget->updateSnapList();
        p_snapWidget->setFinalizeEnabled(true);
    }

    updateFtPlot();
}

void FtmwViewWidget::snapshotLoadError(QString msg)
{
    p_snapWidget->setEnabled(false);
    p_snapWidget->deleteLater();
    p_snapWidget = nullptr;
    if(ui->snapDiffButton->isChecked())
        ui->singleFrameButton->setChecked(true);

    ui->snapDiffButton->setEnabled(false);

    QMessageBox::warning(this,QString("Snapshot Load Error"),msg,QMessageBox::Ok);
}

void FtmwViewWidget::snapListUpdate()
{
    if(d_mode == BlackChirp::FtmwViewSnapDiff || ui->snapshotCheckbox->isChecked())
        updateFtPlot();
}

void FtmwViewWidget::snapRefChanged()
{
    if(d_mode == BlackChirp::FtmwViewSnapDiff)
    {
        d_currentRefFid = p_snapWidget->getRefFid(ui->frameBox->value()-1);
        updateFtPlot();
    }
}

void FtmwViewWidget::snapDiffChanged()
{
    if(d_mode == BlackChirp::FtmwViewSnapDiff)
    {
        d_currentFid = p_snapWidget->getDiffFid(ui->frameBox->value()-1);
        updateFtPlot();
    }
}

void FtmwViewWidget::finalizedSnapList(const FidList l)
{
    Q_ASSERT(l.size() > 0);
    d_currentFidList = l;
    updateShotsLabel(l.first().shots());

    if(ui->snapshotCheckbox->isChecked())
    {
        ui->snapshotCheckbox->blockSignals(true);
        ui->snapshotCheckbox->setChecked(false);
        ui->snapshotCheckbox->setEnabled(false);
        ui->snapshotCheckbox->blockSignals(false);
    }

    if(d_mode == BlackChirp::FtmwViewSnapDiff)
    {
        ui->singleFrameButton->blockSignals(true);
        ui->singleFrameButton->setChecked(true);
        ui->singleFrameButton->blockSignals(false);
        d_mode = BlackChirp::FtmwViewSingle;
    }

    removeSnapWidget();

    emit finalized(d_currentExptNum);
}

void FtmwViewWidget::removeSnapWidget()
{
    ui->snapDiffButton->setEnabled(false);
    ui->snapshotCheckbox->setEnabled(false);

    if(p_snapWidget != nullptr)
    {
        p_snapWidget->setEnabled(false);
        p_snapWidget->deleteLater();
        p_snapWidget = nullptr;
    }
}

void FtmwViewWidget::checkRemoveSnapWidget(int num)
{
    if(num == d_currentExptNum)
    {
        if(d_mode == BlackChirp::FtmwViewSnapDiff || ui->snapshotCheckbox->isChecked())
        {
            ui->snapshotCheckbox->blockSignals(true);
            ui->snapshotCheckbox->setChecked(false);
            ui->snapshotCheckbox->blockSignals(false);

            ui->singleFrameButton->blockSignals(true);
            ui->singleFrameButton->setChecked(true);
            ui->singleFrameButton->blockSignals(false);

        }

        removeSnapWidget();

        modeChanged();
    }
}

