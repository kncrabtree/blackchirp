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
#include <QMessageBox>

#include "datastructs.h"
#include "experiment.h"
#include "snapworker.h"

FtmwSnapshotWidget::FtmwSnapshotWidget(int num, const QString path, QWidget *parent) : QWidget(parent), d_num(num), d_busy(false),
    d_updateWhenDone(false), d_path(path)
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

    p_finalizeButton = new QPushButton(QString(" Finalize"));
    p_finalizeButton->setEnabled(false);
    p_finalizeButton->setIcon(QIcon(QString(":/icons/check.png")));
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
    return p_sw->parseFile(d_num,p_refBox->value(),d_path).at(i);
}

Fid FtmwSnapshotWidget::getDiffFid(int i)
{
    Q_ASSERT(p_diffBox->value() < count());
    return p_sw->parseFile(d_num,p_diffBox->value(),d_path).at(i);
}

QSize FtmwSnapshotWidget::sizeHint() const
{
    return QSize(100,300);
}

void FtmwSnapshotWidget::setFidList(const QList<Fid> l)
{
    d_totalFidList = l;
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
    if(en)
        connect(p_finalizeButton,&QPushButton::clicked,this,&FtmwSnapshotWidget::finalize, Qt::UniqueConnection);
    else
        disconnect(p_finalizeButton,&QPushButton::clicked,this,&FtmwSnapshotWidget::finalize);
}

bool FtmwSnapshotWidget::readSnapshots()
{
    bool out = false;
    QFile snp(BlackChirp::getExptFile(d_num,BlackChirp::SnapFile,d_path));
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
                QFile f(BlackChirp::getExptFile(d_num,BlackChirp::FidFile,d_path,i));
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

    QMetaObject::invokeMethod(p_sw,"calculateFidList",Q_ARG(int,d_num),Q_ARG(const QList<Fid>,d_totalFidList),
                              Q_ARG(const QList<int>,snapList),Q_ARG(bool,subtract));
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

void FtmwSnapshotWidget::finalize()
{
    if(d_snapList.isEmpty())
    {
        QMessageBox::critical(qobject_cast<QWidget*>(parent()),QString("Finalize Error"),QString("Cannot finalize because the snapshot list is empty."),QMessageBox::Ok);
        return;
    }

    int ret = QMessageBox::question(qobject_cast<QWidget*>(parent()),QString("Discard snapshots?"),QString("If you continue, the currently-selected snapshots will be combined, and the FID output file overwritten.\nThe snapshots themselves will be deleted.\n\nAre you sure you wish to proceed?"),QMessageBox::Yes|QMessageBox::No,QMessageBox::No);

    if(ret == QMessageBox::No)
        return;

    //write fid file
    if(!FtmwConfig::writeFidFile(d_num,d_snapList,d_path))
    {
        QMessageBox::critical(qobject_cast<QWidget*>(parent()),QString("Save failed!"),QString("Could not write FID file!"),QMessageBox::Ok);
        return;
    }

    bool remainderKept = (p_lw->item(count()-1)->data(Qt::CheckStateRole) == Qt::Checked ? true : false);
    QList<int> snaps;
    for(int i=count()-2; i>=0; i--)
    {
        if(p_lw->item(i)->data(Qt::CheckStateRole) == Qt::Unchecked)
            snaps.append(i);
    }

    emit experimentLogMessage(d_num,QString("Finalizing snapshots...."),BlackChirp::LogNormal,d_path);

    if(snaps.isEmpty() && remainderKept)
        emit experimentLogMessage(d_num,QString("All snapshots kept."),BlackChirp::LogNormal,d_path);
    else
    {
        if(snaps.isEmpty())
        {
            if(remainderKept)
                emit experimentLogMessage(d_num,QString("All snapshots removed."),BlackChirp::LogNormal,d_path);
            else
                emit experimentLogMessage(d_num,QString("All snapshots kept."),BlackChirp::LogNormal,d_path);
        }
        else if(snaps.size() == 1)
            emit experimentLogMessage(d_num,QString("Removed snapshot %1.").arg(snaps.first()),BlackChirp::LogNormal,d_path);
        else if(snaps.size() == 2)
        {
            std::stable_sort(snaps.begin(),snaps.end());
            emit experimentLogMessage(d_num,QString("Removed snapshots %1 and %2.").arg(snaps.first()).arg(snaps.last()),BlackChirp::LogNormal,d_path);
        }
        else
        {
            std::stable_sort(snaps.begin(),snaps.end());
            QString snapString = QString("and %1").arg(snaps.last());
            for(int i = snaps.size()-2; i>=0; i--)
                snapString.prepend(QString("%1, ").arg(snaps.at(i)));
            emit experimentLogMessage(d_num,QString("Removed snapshots %1.").arg(snapString),BlackChirp::LogNormal,d_path);
        }
    }

    if(remainderKept)
        emit experimentLogMessage(d_num,QString("Remainder of shots kept."),BlackChirp::LogNormal,d_path);
    else
        emit experimentLogMessage(d_num,QString("Remainder of shots removed."),BlackChirp::LogNormal,d_path);

    emit experimentLogMessage(d_num,QString("Final number of shots: %1").arg(d_snapList.first().shots()),BlackChirp::LogNormal,d_path);


    //delete snapshot files
    for(int i=0; i<count()-1; i++)
    {
        QFile snap(BlackChirp::getExptFile(d_num,BlackChirp::FidFile,d_path,i));
        if(snap.exists())
            snap.remove();
    }

    //rewrite or delete snp file
    QFile snp(BlackChirp::getExptFile(d_num,BlackChirp::SnapFile,d_path));
    if(snp.exists())
    {
        if(snp.open(QIODevice::ReadOnly))
        {
            QByteArrayList l;
            while(!snp.atEnd())
            {
                QByteArray line = snp.readLine();
                if(!line.isEmpty() && !line.startsWith("fid"))
                    l.append(line);
            }
            snp.close();

            //if there's anything left (eg LIF snapshots), rewrite the file with those
            if(!l.isEmpty())
            {
                snp.open(QIODevice::WriteOnly);
                while(!l.isEmpty())
                    snp.write(l.takeFirst());
                snp.close();
            }
            else
                snp.remove();

        }
        else
            snp.remove();
    }

    //rewrite experiment header
    Experiment e(d_num,d_path);
    e.saveHeader();

    emit finalizedList(d_snapList);

}
