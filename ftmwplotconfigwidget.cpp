#include "ftmwplotconfigwidget.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>
#include <QCheckBox>
#include <QListWidgetItem>
#include <QFile>
#include <QThread>
#include <QMessageBox>

#include "snapworker.h"

FtmwPlotConfigWidget::FtmwPlotConfigWidget(int id, QString path, QWidget *parent) : QWidget(parent), d_num(-1), d_id(id), d_busy(false), d_updateWhenDone(false), d_path(path)
{
    auto vbl = new QVBoxLayout;

    auto fl = new QFormLayout;

    p_frameBox = new QSpinBox;
    p_frameBox->setRange(1,1);
    p_frameBox->setEnabled(false);
    p_frameBox->setKeyboardTracking(false);
    connect(p_frameBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,[=](int v){
        emit frameChanged(d_id,v-1);
    });

    auto flbl = new QLabel("Frame Number");
    flbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(flbl,p_frameBox);

    p_segmentBox = new QSpinBox;
    p_segmentBox->setRange(1,1);
    p_segmentBox->setEnabled(false);
    p_segmentBox->setKeyboardTracking(false);
    connect(p_segmentBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,[=](int v){
        emit segmentChanged(d_id,v-1);
    });

    auto slbl = new QLabel("Segment Number");
    slbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(slbl,p_segmentBox);

    vbl->addLayout(fl);

    fl = new QFormLayout;
    auto gb = new QGroupBox(QString("Snapshot Control"));
    p_allButton = new QRadioButton;
    p_allButton->setChecked(true);
    p_allButton->setToolTip(QString("Include all shots taken since beginning of experiment."));
    connect(p_allButton,&QRadioButton::toggled,this,&FtmwPlotConfigWidget::configureSnapControls);
    connect(p_allButton,&QRadioButton::toggled,this,&FtmwPlotConfigWidget::process);

    auto allL = new QLabel(QString("All"));
    allL->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(allL,p_allButton);

    p_recentButton = new QRadioButton;
    p_recentButton->setToolTip(QString("Only show shots taken after the most recent snapshot."));
    connect(p_recentButton,&QRadioButton::toggled,this,&FtmwPlotConfigWidget::configureSnapControls);
    connect(p_recentButton,&QRadioButton::toggled,this,&FtmwPlotConfigWidget::process);

    auto rl = new QLabel(QString("Current Shots"));
    rl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(rl,p_recentButton);

    p_selectedButton = new QRadioButton;
    p_selectedButton->setToolTip(QString("Only show shots taken suring the intervals selected below."));
    auto sl = new QLabel(QString("Selected"));
    sl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(sl,p_selectedButton);
    connect(p_selectedButton,&QRadioButton::toggled,this,&FtmwPlotConfigWidget::configureSnapControls);
    connect(p_selectedButton,&QRadioButton::toggled,this,&FtmwPlotConfigWidget::process);

    gb->setLayout(fl);
    vbl->addWidget(gb);

    auto hbl = new QHBoxLayout;
    p_selectAllButton = new QPushButton(QString("Select All"));
    p_selectNoneButton = new QPushButton(QString("Select None"));
    connect(p_selectAllButton,&QPushButton::clicked,this,&FtmwPlotConfigWidget::selectAll);
    connect(p_selectNoneButton,&QPushButton::clicked,this,&FtmwPlotConfigWidget::selectNone);
    hbl->addWidget(p_selectAllButton);
    hbl->addWidget(p_selectNoneButton);
    vbl->addLayout(hbl);

    p_lw = new QListWidget(this);
    connect(p_lw,&QListWidget::itemChanged,this,&FtmwPlotConfigWidget::process);
    vbl->addWidget(p_lw,1);

    p_remainderBox = new QCheckBox(QString("Include FIDs since last snapshot?"));
    p_remainderBox->setToolTip(QString("Check to include all shots taken after the most recent snapshot."));
    p_remainderBox->setChecked(true);
    connect(p_remainderBox,&QCheckBox::toggled,this,&FtmwPlotConfigWidget::process);

    vbl->addWidget(p_remainderBox,0);

    p_finalizeButton = new QPushButton(QString(" Finalize"));
    p_finalizeButton->setEnabled(false);
    p_finalizeButton->setIcon(QIcon(QString(":/icons/check.png")));
    connect(p_finalizeButton,&QPushButton::clicked,this,&FtmwPlotConfigWidget::finalizeSnapshots);
    vbl->addWidget(p_finalizeButton,0);

    setLayout(vbl);

    p_allButton->setEnabled(false);
    p_recentButton->setEnabled(false);
    p_selectedButton->setEnabled(false);
    p_selectAllButton->setEnabled(false);
    p_selectNoneButton->setEnabled(false);
    p_lw->setEnabled(false);
    p_remainderBox->setEnabled(false);
    p_finalizeButton->setEnabled(false);

    p_workerThread = new QThread(this);
    p_sw = new SnapWorker;
    connect(p_workerThread,&QThread::finished,p_sw,&SnapWorker::deleteLater);
    connect(p_sw,&SnapWorker::processingComplete,this,&FtmwPlotConfigWidget::processingComplete);
    connect(p_sw,&SnapWorker::finalProcessingComplete,this,&FtmwPlotConfigWidget::finalizeComplete);
    p_sw->moveToThread(p_workerThread);
    p_workerThread->start();
}

FtmwPlotConfigWidget::~FtmwPlotConfigWidget()
{
    p_workerThread->quit();
    p_workerThread->wait();
}

void FtmwPlotConfigWidget::prepareForExperiment(const Experiment e)
{
    blockSignals(true);

    d_num = e.number();

    p_lw->clear();

    //these things only become enabled once snapshots have been taken
    configureSnapControls();
    p_remainderBox->setChecked(true);
    p_finalizeButton->setEnabled(false);

    if(e.ftmwConfig().isEnabled())
    {
        p_frameBox->setRange(1,e.ftmwConfig().numFrames());
        p_frameBox->setEnabled(true);

        p_segmentBox->setRange(1,e.ftmwConfig().numSegments());
        p_segmentBox->setEnabled(true);

        p_allButton->setEnabled(true);
    }
    else
    {
        p_frameBox->setRange(1,1);
        p_frameBox->setEnabled(false);

        p_segmentBox->setRange(1,1);
        p_segmentBox->setEnabled(false);

        p_allButton->setEnabled(false);
    }


    blockSignals(false);
}

void FtmwPlotConfigWidget::experimentComplete(const Experiment e)
{
    processFtmwConfig(e.ftmwConfig());

    if(p_lw->count() > 0)
        p_finalizeButton->setEnabled(true);
}

void FtmwPlotConfigWidget::snapshotTaken()
{
    QFile snp(BlackChirp::getExptFile(d_num,BlackChirp::SnapFile,d_path));
    if(snp.open(QIODevice::ReadOnly))
    {
        int numSnaps = 0;
        while(!snp.atEnd())
        {
            QString line = snp.readLine();
            if(line.startsWith(QString("fid")) || line.startsWith(QString("mfd")))
            {
                QStringList l = line.split(QString("\t"));
                bool ok = false;
                int n = l.constLast().trimmed().toInt(&ok);
                if(ok)
                {
                    numSnaps = n;
                    break;
                }
            }
        }


        if(numSnaps > 0)
        {
            for(int i = p_lw->count(); i < numSnaps; i++)
            {
                QListWidgetItem *item = new QListWidgetItem(QString("Snapshot %1").arg(i));
                item->setFlags(Qt::ItemIsEnabled|Qt::ItemIsUserCheckable);
                if(!p_lw->isEnabled())
                    item->setCheckState(Qt::Checked);
                else
                    item->setCheckState(Qt::Unchecked);
                p_lw->insertItem(i,item);
            }
        }
        snp.close();
    }

    configureSnapControls();
}

bool FtmwPlotConfigWidget::isSnapshotActive()
{
    return !p_allButton->isChecked();
}

void FtmwPlotConfigWidget::configureSnapControls()
{
    if(p_lw->count() > 0)
    {
        p_recentButton->setEnabled(true);
        p_selectedButton->setEnabled(true);
        p_selectAllButton->setEnabled(p_selectedButton->isChecked());
        p_selectNoneButton->setEnabled(p_selectedButton->isChecked());
        p_lw->setEnabled(p_selectedButton->isChecked());
        p_remainderBox->setEnabled(p_selectedButton->isChecked());
    }
    else
    {
        p_allButton->blockSignals(true);
        p_allButton->setChecked(true);
        p_allButton->blockSignals(false);

        p_recentButton->setEnabled(false);
        p_selectedButton->setEnabled(false);
        p_selectAllButton->setEnabled(false);
        p_selectNoneButton->setEnabled(false);
        p_lw->setEnabled(false);
        p_remainderBox->setEnabled(false);
    }
}

void FtmwPlotConfigWidget::process()
{
    if(d_ftmwToProcess.completedShots() < 1)
        return;

    if(d_busy)
        d_updateWhenDone = true;
    else
    {
        d_updateWhenDone = false;
        processFtmwConfig(d_ftmwToProcess);
    }
}

void FtmwPlotConfigWidget::processFtmwConfig(const FtmwConfig ref)
{
    d_ftmwToProcess = ref;
    if(d_busy)
        d_updateWhenDone = true;
    else
    {
        d_updateWhenDone = false;
        if(isSnapshotActive())
        {
            bool rem = p_remainderBox->isChecked();
            QList<int> snaps;
            if(p_selectedButton->isChecked())
            {
                Qt::CheckState c = Qt::Checked;
                if(rem)
                    c = Qt::Unchecked;

                for(int i=0; i<p_lw->count(); i++)
                {
                    if(p_lw->item(i)->checkState() == c)
                        snaps << i;
                }
            }
            else
            {
                rem = true;
                for(int i=0; i<p_lw->count(); i++)
                    snaps << i;
            }

            QMetaObject::invokeMethod(p_sw,"calculateSnapshots",Q_ARG(FtmwConfig,ref),Q_ARG(QList<int>,snaps),Q_ARG(bool,rem),Q_ARG(int,d_num),Q_ARG(QString,d_path));

            setCursor(Qt::BusyCursor);
        }
        else
            processingComplete(d_ftmwToProcess);
    }
}

void FtmwPlotConfigWidget::processingComplete(const FtmwConfig out)
{
    d_busy = false;
    unsetCursor();

    emit snapshotsProcessed(d_id,out);

    if(d_updateWhenDone)
        processFtmwConfig(d_ftmwToProcess);

}

void FtmwPlotConfigWidget::selectAll()
{
    p_lw->blockSignals(true);
    for(int i=0; i<p_lw->count(); i++)
        p_lw->item(i)->setCheckState(Qt::Checked);
    p_lw->blockSignals(false);

    process();
}

void FtmwPlotConfigWidget::selectNone()
{
    p_lw->blockSignals(true);
    for(int i=0; i<p_lw->count(); i++)
        p_lw->item(i)->setCheckState(Qt::Unchecked);
    p_lw->blockSignals(false);

    process();
}

void FtmwPlotConfigWidget::finalizeSnapshots()
{
    int ret = QMessageBox::question(this,QString("Finalize Snapshots?"),QString("If you continue, the currently-selected snapshots will be combined, and the output file overwritten.\nThe snapshots themselves will be deleted.\n\nAre you sure you wish to proceed?"),QMessageBox::Yes|QMessageBox::No,QMessageBox::No);

    if(ret == QMessageBox::No)
        return;

    d_updateWhenDone = false;
    d_busy = false;

    if(isSnapshotActive())
    {
        bool rem = p_remainderBox->isChecked();
        QList<int> snaps;
        if(p_selectedButton->isChecked())
        {
            Qt::CheckState c = Qt::Checked;
            if(rem)
                c = Qt::Unchecked;

            for(int i=0; i<p_lw->count(); i++)
            {
                if(p_lw->item(i)->checkState() == c)
                    snaps << i;
            }
        }
        else
        {
            rem = true;
            for(int i=0; i<p_lw->count(); i++)
                snaps << i;
        }

        QMetaObject::invokeMethod(p_sw,"finalizeSnapshots",Q_ARG(FtmwConfig,d_ftmwToProcess),Q_ARG(QList<int>,snaps),Q_ARG(bool,rem),Q_ARG(int,d_num),Q_ARG(QString,d_path));

        setCursor(Qt::BusyCursor);
    }
    else
        finalizeComplete(d_ftmwToProcess);

}

void FtmwPlotConfigWidget::finalizeComplete(const FtmwConfig out)
{
    unsetCursor();
    clearAll();

    emit snapshotsFinalized(out);
}

void FtmwPlotConfigWidget::clearAll()
{
    p_lw->blockSignals(true);
    p_lw->clear();
    p_lw->blockSignals(false);

    p_finalizeButton->setEnabled(false);
    p_allButton->setChecked(true);
    configureSnapControls();
}
