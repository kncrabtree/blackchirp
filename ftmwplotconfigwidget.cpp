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

FtmwPlotConfigWidget::FtmwPlotConfigWidget(QString path, QWidget *parent) : QWidget(parent), d_num(-1), d_busy(false), d_updateWhenDone(false), d_path(path)
{
    auto vbl = new QVBoxLayout;

    auto fl = new QFormLayout;

    p_frameBox = new QSpinBox;
    p_frameBox->setRange(1,1);
    p_frameBox->setEnabled(false);
    p_frameBox->setKeyboardTracking(false);
    connect(p_frameBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,[=](int v){
        emit frameChanged(v-1);
    });

    auto flbl = new QLabel("Frame Number");
    flbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(flbl,p_frameBox);

    p_segmentBox = new QSpinBox;
    p_segmentBox->setRange(1,1);
    p_segmentBox->setEnabled(false);
    p_segmentBox->setKeyboardTracking(false);
    connect(p_segmentBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,[=](int v){
        emit segmentChanged(v-1);
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

    auto allL = new QLabel(QString("All"));
    allL->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(allL,p_allButton);

    p_recentButton = new QRadioButton;
    p_recentButton->setToolTip(QString("Only show shots taken after the most recent snapshot."));
    connect(p_recentButton,&QRadioButton::toggled,this,&FtmwPlotConfigWidget::configureSnapControls);

    auto rl = new QLabel(QString("Current Shots"));
    rl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(rl,p_recentButton);

    p_selectedButton = new QRadioButton;
    p_selectedButton->setToolTip(QString("Only show shots taken suring the intervals selected below."));
    auto sl = new QLabel(QString("Selected"));
    sl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(sl,p_selectedButton);
    connect(p_selectedButton,&QRadioButton::toggled,this,&FtmwPlotConfigWidget::configureSnapControls);

    gb->setLayout(fl);
    vbl->addWidget(gb);

    auto hbl = new QHBoxLayout;
    p_selectAllButton = new QPushButton(QString("Select All"));
    p_selectNoneButton = new QPushButton(QString("Select None"));
    hbl->addWidget(p_selectAllButton);
    hbl->addWidget(p_selectNoneButton);
    vbl->addLayout(hbl);

    p_lw = new QListWidget(this);
//    connect(p_lw,&QListWidget::itemChanged,this,&FtmwSnapshotWidget::updateSnapList);
    vbl->addWidget(p_lw,1);

    p_remainderBox = new QCheckBox(QString("Include FIDs since last snapshot?"));
    p_remainderBox->setToolTip(QString("Check to include all shots taken after the most recent snapshot."));
    p_remainderBox->setChecked(true);

    vbl->addWidget(p_remainderBox,0);

    p_finalizeButton = new QPushButton(QString(" Finalize"));
    p_finalizeButton->setEnabled(false);
    p_finalizeButton->setIcon(QIcon(QString(":/icons/check.png")));
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
    }
    else
    {
        p_frameBox->setRange(1,1);
        p_frameBox->setEnabled(false);

        p_segmentBox->setRange(1,1);
        p_segmentBox->setEnabled(false);
    }


    blockSignals(false);
}

void FtmwPlotConfigWidget::experimentComplete(const Experiment e)
{
    Q_UNUSED(e)

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
//        updateSnapList();
        snp.close();
    }

    configureSnapControls();
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
