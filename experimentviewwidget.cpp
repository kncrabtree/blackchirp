#include "experimentviewwidget.h"

#include <QTabWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QLabel>
#include <QFile>
#include <QTextEdit>
#include <QFileDialog>
#include <QToolBar>
#include <QAction>
#include <QMessageBox>

#include "ftmwviewwidget.h"
#include "trackingviewwidget.h"
#include "loghandler.h"

#ifdef BC_LIF
#include "lifdisplaywidget.h"
#endif

#ifdef BC_MOTOR
#include "motordisplaywidget.h"
#endif

ExperimentViewWidget::ExperimentViewWidget(int num, QString path, QWidget *parent) : QWidget(parent), p_ftmw(nullptr), p_lh(nullptr)
{
    d_experiment = Experiment(num,path);
    setWindowFlags(Qt::Window);
    setWindowTitle(QString("Experiment %1").arg(num));
    setAttribute(Qt::WA_DeleteOnClose);

    QToolBar *toolBar = new QToolBar(this);
    toolBar->setMovable(false);
    toolBar->setAllowedAreas(Qt::TopToolBarArea);
    QAction *exportAction = toolBar->addAction(QIcon(QString(":/icons/export.png")),QString("Export Experiment"));
    connect(exportAction,&QAction::triggered,this,&ExperimentViewWidget::exportAscii);
    toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);



    QVBoxLayout *vbl = new QVBoxLayout;
    vbl->addWidget(toolBar);

    if(d_experiment.number() < 1)
    {
        QLabel *errLabel = new QLabel(d_experiment.errorString());
        errLabel->setAlignment(Qt::AlignCenter);
        errLabel->setWordWrap(true);
        vbl->addWidget(errLabel);
        resize(500,500);
        setLayout(vbl);
        return;
    }

    //tabwidget creation
    p_tabWidget = new QTabWidget(this);
    p_tabWidget->setTabPosition(QTabWidget::East);

    QWidget *hdr = buildHeaderWidget();
    if(hdr != nullptr)
        p_tabWidget->addTab(hdr,QIcon(QString(":/icons/header.png")),QString("Header"));

    if(d_experiment.ftmwConfig().isEnabled())
    {
        QWidget *ftmw = buildFtmwWidget(path);
        if(ftmw != nullptr)
            p_tabWidget->addTab(ftmw,QIcon(QString(":/icons/chirp.png")),QString("CP-FTMW"));
    }

#ifdef BC_LIF
    if(d_experiment.lifConfig().isEnabled())
    {
        QWidget *lif = buildLifWidget();
        if(lif != nullptr)
            p_tabWidget->addTab(lif,QIcon(QString(":/icons/laser.png")),QString("LIF"));
    }
#endif

#ifdef BC_MOTOR
    if(d_experiment.motorScan().isEnabled())
    {
        QWidget *motor = buildMotorWidget();
        if(motor != nullptr)
            p_tabWidget->addTab(motor,QIcon(QString(":/icons/motorscan.png")),QString("Motor"));
    }
#endif

    QWidget *tracking = buildTrackingWidget();
    if(tracking != nullptr)
        p_tabWidget->addTab(tracking,QIcon(QString(":/icons/dataplots.png")),QString("Tracking"));

    QWidget *log = buildLogWidget(path);
    if(log != nullptr)
    {
        p_tabWidget->addTab(log,QIcon(QString(":/icons/log.png")),QString("Log"));
        if(p_ftmw != nullptr)
            connect(p_ftmw,&FtmwViewWidget::experimentLogMessage,p_lh,&LogHandler::experimentLogMessage);
    }

    vbl->addWidget(p_tabWidget);
    setLayout(vbl);
}

QSize ExperimentViewWidget::sizeHint() const
{
    return QSize(1024,768);
}

void ExperimentViewWidget::exportAscii()
{
    QString path = BlackChirp::getExportDir();

    QString name = QFileDialog::getSaveFileName(this,QString("Export Experiment"),path + QString("/expt%1.txt").arg(d_experiment.number()));

    if(name.isEmpty())
        return;

    QFile f(name);
    if(!f.open(QIODevice::WriteOnly))
    {
        QMessageBox::critical(this,QString("Export Failed"),QString("Could not open file %1 for writing. Please choose a different filename.").arg(name));
        return;
    }

    f.close();

    BlackChirp::setExportDir(name);

    d_experiment.exportAscii(name);
}

void ExperimentViewWidget::ftmwFinalized(int num)
{
    if(num == d_experiment.number())
    {
        p_tabWidget->removeTab(0);
        p_tabWidget->insertTab(0,buildHeaderWidget(),QIcon(QString(":/icons/header.png")),QString("Header"));
        if(p_ftmw != nullptr)
            p_ftmw->checkRemoveSnapWidget(num);

        emit notifyUiFinalized(num);

        update();
    }
}

QWidget *ExperimentViewWidget::buildHeaderWidget()
{
    QWidget *hdr = new QWidget();
    QVBoxLayout *hdrvl = new QVBoxLayout();

    //header page
    QTableWidget *tw = new QTableWidget(this);
    tw->setColumnCount(3);
    tw->setEditTriggers(QTableWidget::NoEditTriggers);
    tw->setSelectionBehavior(QAbstractItemView::SelectRows);

    tw->setHorizontalHeaderItem(0,new QTableWidgetItem(QString("Key")));
    tw->setHorizontalHeaderItem(1,new QTableWidgetItem(QString("Value")));
    tw->setHorizontalHeaderItem(2,new QTableWidgetItem(QString("Unit")));
    tw->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    auto header = d_experiment.headerMap();
    auto hdrit = header.constBegin();
    tw->setRowCount(header.size());
    for(int i=0; hdrit != header.constEnd(); i++, hdrit++)
    {
        tw->setItem(i,0,new QTableWidgetItem(hdrit.key()));
        tw->setItem(i,1,new QTableWidgetItem(hdrit.value().first.toString()));
        tw->setItem(i,2,new QTableWidgetItem(hdrit.value().second));
    }
    hdrvl->addWidget(tw);
    hdr->setLayout(hdrvl);

    return hdr;
}

QWidget *ExperimentViewWidget::buildFtmwWidget(QString path)
{
    QWidget *out = nullptr;
    p_ftmw = nullptr;
    if(d_experiment.ftmwConfig().isEnabled())
    {
        out = new QWidget;
        QVBoxLayout *vbl = new QVBoxLayout;
        p_ftmw = new FtmwViewWidget(out,path);
        connect(p_ftmw,&FtmwViewWidget::finalized,this,&ExperimentViewWidget::ftmwFinalized);
        vbl->addWidget(p_ftmw);
        out->setLayout(vbl);

        p_ftmw->prepareForExperiment(d_experiment);
        p_ftmw->newFidList(d_experiment.ftmwConfig().fidList());
        if(!d_experiment.ftmwConfig().fidList().isEmpty())
            p_ftmw->updateShotsLabel(d_experiment.ftmwConfig().fidList().first().shots());

        //check for snap file
        QFile snp(BlackChirp::getExptFile(d_experiment.number(),BlackChirp::SnapFile,path));
        if(snp.exists() && snp.open(QIODevice::ReadOnly))
        {
            bool fids = false;
            while(!snp.atEnd())
            {
                QByteArray line = snp.readLine();
                if(line.startsWith("fid"))
                {
                    QByteArrayList l = line.split('\t');
                    if(!l.isEmpty() && l.last().trimmed().toInt() > 0)
                    {
                        fids = true;
                        break;
                    }
                }
            }
            snp.close();

            if(fids)
                p_ftmw->snapshotTaken();
        }

        p_ftmw->experimentComplete();
    }


    return out;
}

#ifdef BC_LIF
QWidget *ExperimentViewWidget::buildLifWidget()
{
    QWidget *out = nullptr;
    if(d_experiment.lifConfig().isEnabled())
    {
        out = new QWidget;
        QVBoxLayout *vbl = new QVBoxLayout;
        LifDisplayWidget *lif = new LifDisplayWidget(out);
        vbl->addWidget(lif);
        out->setLayout(vbl);

        lif->prepareForExperiment(d_experiment.lifConfig());

        auto d = d_experiment.lifConfig().lifData();
        QPoint p;
        for(int i = 0; i<d.size(); i++)
        {
            auto dat = d.at(i);
            p.setX(i);
            for(int j=0; j<dat.size(); j++)
            {
                p.setY(j);
                lif->updatePoint(qMakePair(p,dat.at(j)));
            }
        }
    }

    return out;
}
#endif

QWidget *ExperimentViewWidget::buildTrackingWidget()
{
    //tracking page
    QWidget *tracking = nullptr;
    auto timeData = d_experiment.timeDataMap();
    bool showWidget = false;
    auto trkit = timeData.constBegin();
    if(!timeData.isEmpty())
    {
        for(;trkit != timeData.constEnd(); trkit++)
        {
            if(trkit.value().second == true && trkit.value().first.size() > 1)
            {
                showWidget = true;
                break;
            }

        }
    }

    if(showWidget)
    {
        tracking = new QWidget;
        QVBoxLayout *trackingvl = new QVBoxLayout;

        TrackingViewWidget *tvw = new TrackingViewWidget(true,tracking);
        trackingvl->addWidget(tvw);

        auto timestampList = timeData.value(QString("exptTimeStamp")).first;

        trkit = timeData.constBegin();
        for(;trkit != timeData.constEnd(); trkit++)
        {
            if(!trkit.value().second)
                continue;

            auto list = trkit.value().first;
            if(list.size() != timestampList.size())
                continue;

            for(int i=0; i<list.size(); i++)
            {
                bool ok = false;
                double d = list.at(i).toDouble(&ok);
                if(ok)
                {
                    QList<QPair<QString,QVariant>> newList;
                    newList.append(qMakePair(trkit.key(),d));
                    tvw->pointUpdated(newList,true,timestampList.at(i).toDateTime());
                }
            }
        }

        tracking->setLayout(trackingvl);
    }

    return tracking;
}

QWidget *ExperimentViewWidget::buildLogWidget(QString path)
{
    QWidget *log = new QWidget;
    QBoxLayout *vbl = new QVBoxLayout;
    QTextEdit *te = new QTextEdit(log);
    te->setReadOnly(true);
    p_lh = new LogHandler(false,log);
    connect(p_lh,&LogHandler::sendLogMessage,te,&QTextEdit::append);
    vbl->addWidget(te);
    log->setLayout(vbl);

    QFile f(BlackChirp::getExptFile(d_experiment.number(),BlackChirp::LogFile,path));
    if(f.open(QIODevice::ReadOnly))
    {
        while(!f.atEnd())
        {
            QString line = QString(f.readLine());
            if(line.isEmpty())
                continue;

            if(line.contains(QString("[DEBUG]")))
                continue;
            if(line.contains(QString(": [WARNING] ")))
            {
                QStringList l = line.split(QString(": [WARNING] "));
                if(l.size() < 2)
                    continue;

                p_lh->logMessageWithTime(l.last(),BlackChirp::LogWarning,QDateTime::fromString(l.first()));
                continue;
            }
            if(line.contains(QString(": [ERROR] ")))
            {
                QStringList l = line.split(QString(": [ERROR] "));
                if(l.size() < 2)
                    continue;

                p_lh->logMessageWithTime(l.last(),BlackChirp::LogError,QDateTime::fromString(l.first()));
                continue;
            }
            if(line.contains(QString(": [HIGHLIGHT] ")))
            {
                QStringList l = line.split(QString(": [HIGHLIGHT] "));
                if(l.size() < 2)
                    continue;

                p_lh->logMessageWithTime(l.last(),BlackChirp::LogHighlight,QDateTime::fromString(l.first()));
                continue;
            }
            else
            {
                QStringList l = line.split(QString(": "));
                if(l.size() < 2)
                    continue;

                QString theLine;
                theLine += l.at(1);
                for(int i=2; i<l.size(); i++)
                    theLine+=QString(": ")+l.at(i);

                p_lh->logMessageWithTime(theLine,BlackChirp::LogNormal,QDateTime::fromString(l.first()));
            }
        }
        f.close();
    }

    return log;

}

#ifdef BC_MOTOR
QWidget *ExperimentViewWidget::buildMotorWidget()
{
    QWidget *out = nullptr;
    if(d_experiment.motorScan().isEnabled())
    {
        out = new QWidget;
        QVBoxLayout *vbl = new QVBoxLayout;
        MotorDisplayWidget *motor = new MotorDisplayWidget(out);
        vbl->addWidget(motor);
        out->setLayout(vbl);

        motor->prepareForScan(d_experiment.motorScan());
        motor->newMotorData(d_experiment.motorScan());
    }

    return out;
}
#endif

