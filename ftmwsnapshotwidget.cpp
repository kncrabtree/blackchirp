#include "ftmwsnapshotwidget.h"

#include <QGroupBox>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QListWidget>
#include <QSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QListWidgetItem>
#include <QFile>
#include <QThread>

#include "datastructs.h"
#include "snapworker.h"

FtmwSnapshotWidget::FtmwSnapshotWidget(int num, QWidget *parent) : QWidget(parent), d_num(num), d_busy(false),
    d_updateWhenDone(false)
{
    QVBoxLayout *vl = new QVBoxLayout;
    QGroupBox *gb = new QGroupBox(QString("Snapshot Control"));
    vl->addWidget(gb);

    QVBoxLayout *vbl = new QVBoxLayout;

    p_lw = new QListWidget(this);
    connect(p_lw,&QListWidget::itemChanged,this,&FtmwSnapshotWidget::updateSnapList);
    vbl->addWidget(p_lw,1);

    QFormLayout *fl = new QFormLayout;
    p_refBox = new QSpinBox(this);
    p_refBox->setRange(1,1);
    p_refBox->setEnabled(false);
    connect(p_refBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwSnapshotWidget::refChanged);
    fl->addRow(QString("Ref Snapshot"),p_refBox);

    p_diffBox = new QSpinBox(this);
    p_diffBox->setRange(1,1);
    p_diffBox->setEnabled(false);
    connect(p_diffBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwSnapshotWidget::diffChanged);
    fl->addRow(QString("Diff Snapshot"),p_diffBox);

    vbl->addLayout(fl,0);

    p_finalizeButton = new QPushButton(QString("Finalize"));
    p_finalizeButton->setEnabled(false);
    vbl->addWidget(p_finalizeButton,0);

    gb->setLayout(vbl);

    setLayout(vl);

    p_workerThread = new QThread(this);
    p_sw = new SnapWorker();
    p_sw->moveToThread(p_workerThread);
    connect(p_workerThread,&QThread::finished,p_sw,&SnapWorker::deleteLater);
    connect(p_sw,&SnapWorker::fidListComplete,this,&FtmwSnapshotWidget::snapListUpdated);
    p_workerThread->start();

}

FtmwSnapshotWidget::~FtmwSnapshotWidget()
{
    p_workerThread->quit();
    p_workerThread->wait();
}

int FtmwSnapshotWidget::count() const
{
    return p_lw->count();
}

bool FtmwSnapshotWidget::isEmpty() const
{
    return p_lw->count() == 0;
}

QList<Fid> FtmwSnapshotWidget::getSnapList() const
{
    return d_snapList;
}

int FtmwSnapshotWidget::snapListSize() const
{
    return d_snapList.size();
}

Fid FtmwSnapshotWidget::getSnapFid(int i) const
{
    Q_ASSERT(i < d_snapList.size());
    return d_snapList.at(i);
}

Fid FtmwSnapshotWidget::getRefFid(int i)
{
    Q_ASSERT(p_refBox->value() < count());
    return p_sw->parseFile(d_num,p_refBox->value()).at(i);
}

Fid FtmwSnapshotWidget::getDiffFid(int i)
{
    Q_ASSERT(p_diffBox->value() < count());
    return p_sw->parseFile(d_num,p_diffBox->value()).at(i);
}

QSize FtmwSnapshotWidget::sizeHint() const
{
    return QSize(100,300);
}

void FtmwSnapshotWidget::setSelectionEnabled(bool en)
{
    p_lw->setEnabled(en);
}

void FtmwSnapshotWidget::setDiffMode(bool en)
{
    if(count() > 1)
    {
        p_refBox->setEnabled(en);
        p_diffBox->setEnabled(en);
    }
    else
    {
        p_refBox->setEnabled(false);
        p_diffBox->setEnabled(false);
    }
}

void FtmwSnapshotWidget::setFinalizeEnabled(bool en)
{
    p_finalizeButton->setEnabled(en);
}

bool FtmwSnapshotWidget::readSnapshots()
{
    bool out = false;
    QFile snp(BlackChirp::getExptFile(d_num,BlackChirp::SnapFile));
    if(snp.open(QIODevice::ReadOnly))
    {
        bool parseSuccess = false;
        int numSnaps = 0;
        while(!snp.atEnd())
        {
            QString line = snp.readLine();
            if(line.startsWith(QString("fid")))
            {
                QStringList l = line.split(QString("\t"));
                bool ok = false;
                int n = l.last().trimmed().toInt(&ok);
                if(ok)
                {
                    parseSuccess = true;
                    numSnaps = n;
                    break;
                }
            }
        }

        if(!parseSuccess)
        {
            emit loadFailed(QString("Could not parse .snp file (%1)").arg(snp.fileName()));
            snp.close();
            return false;
        }


        if(numSnaps > 0)
        {
            if(isEmpty())
            {
                QListWidgetItem *item = new QListWidgetItem(QString("Remainder"));
                item->setFlags(Qt::ItemIsEnabled|Qt::ItemIsUserCheckable);
                item->setCheckState(Qt::Checked);
                p_lw->addItem(item);
            }

            out = true;

            for(int i = count()-1; i < numSnaps; i++)
            {
                QFile f(BlackChirp::getExptFile(d_num,BlackChirp::FidFile,i));
                if(f.exists())
                {
                    QListWidgetItem *item = new QListWidgetItem(QString("Snapshot %1").arg(i));
                    item->setFlags(Qt::ItemIsEnabled|Qt::ItemIsUserCheckable);
                    if(!p_lw->isEnabled())
                        item->setCheckState(Qt::Checked);
                    else
                        item->setCheckState(Qt::Unchecked);
                    p_lw->insertItem(i,item);
                }
                else
                {
                    out = false;
                    emit loadFailed(QString("FID snapshot file %1 does not exist!").arg(f.fileName()));
                    break;
                }
            }
        }
        p_refBox->setRange(0,count()-2);
        p_diffBox->setRange(0,count()-2);
        updateSnapList();
        snp.close();
    }
    else
        emit loadFailed(QString("Could not open .snp file (%1)").arg(snp.fileName()));

    return out;
}

void FtmwSnapshotWidget::updateSnapList()
{
    if(count() < 1)
        return;

    if(d_busy)
    {
        d_updateWhenDone = true;
        return;
    }

    bool subtract = p_lw->item(p_lw->count()-1)->data(Qt::CheckStateRole) == Qt::Checked ? true : false;
    QList<int> snapList;
    for(int i = count()-2; i>=0; i--)
    {
        if(subtract)
        {
            if(p_lw->item(i)->data(Qt::CheckStateRole) == Qt::Unchecked)
                snapList.append(i);
        }
        else
        {
            if(p_lw->item(i)->data(Qt::CheckStateRole) == Qt::Checked)
                snapList.append(i);
        }
    }

    if(!subtract && snapList.isEmpty())
    {
        subtract = true;
        p_lw->blockSignals(true);
        for(int i=0; i<count(); i++)
            p_lw->item(i)->setData(Qt::CheckStateRole,Qt::Checked);
        p_lw->blockSignals(false);
    }

    QMetaObject::invokeMethod(p_sw,"calculateFidList",Q_ARG(int,d_num),Q_ARG(const QList<int>,snapList),
                              Q_ARG(bool,subtract));
    d_busy = true;
    d_updateWhenDone = false;
    setEnabled(false);
    setCursor(Qt::BusyCursor);


}

void FtmwSnapshotWidget::snapListUpdated(const QList<Fid> l)
{
    d_busy = false;

    d_snapList = l;
    emit snapListChanged();

    if(d_updateWhenDone)
        updateSnapList();
    else
    {
        setEnabled(true);
        unsetCursor();
    }
}
