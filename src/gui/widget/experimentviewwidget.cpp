#include <gui/widget/experimentviewwidget.h>

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

#include <gui/widget/ftmwviewwidget.h>
#include <gui/widget/trackingviewwidget.h>
#include <data/loghandler.h>

#ifdef BC_LIF
#include <modules/lif/gui/lifdisplaywidget.h>
#endif

#ifdef BC_MOTOR
#include <modules/motor/gui/motordisplaywidget.h>
#endif

ExperimentViewWidget::ExperimentViewWidget(int num, QString path, QWidget *parent) : QWidget(parent), p_ftmw(nullptr), p_lh(nullptr)
{
    d_experiment = Experiment(num,path);
    setWindowFlags(Qt::Window);
    setWindowTitle(QString("Experiment %1").arg(num));
    setAttribute(Qt::WA_DeleteOnClose);


    QVBoxLayout *vbl = new QVBoxLayout;

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

void ExperimentViewWidget::ftmwFinalized(int num)
{
    if(num == d_experiment.number())
    {
        p_tabWidget->removeTab(0);
        p_tabWidget->insertTab(0,buildHeaderWidget(),QIcon(QString(":/icons/header.png")),QString("Header"));
        if(p_ftmw != nullptr)
            p_ftmw->snapshotsFinalizedUpdateUi(num);

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
//        p_ftmw->updateFtmw(d_experiment.ftmwConfig());
//        if(!d_experiment.ftmwConfig().fidList().isEmpty())
//            p_ftmw->updateShotsLabel(d_experiment.ftmwConfig().fidList().constFirst().shots());


        p_ftmw->snapshotTaken();
        p_ftmw->experimentComplete(d_experiment);
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
        lif->updatePoint(d_experiment.lifConfig());
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

        TrackingViewWidget *tvw = new TrackingViewWidget(BC::Key::trackingWidget,tracking,true);
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

                p_lh->logMessageWithTime(l.constLast(),BlackChirp::LogWarning,QDateTime::fromString(l.constFirst()));
                continue;
            }
            if(line.contains(QString(": [ERROR] ")))
            {
                QStringList l = line.split(QString(": [ERROR] "));
                if(l.size() < 2)
                    continue;

                p_lh->logMessageWithTime(l.constLast(),BlackChirp::LogError,QDateTime::fromString(l.constFirst()));
                continue;
            }
            if(line.contains(QString(": [HIGHLIGHT] ")))
            {
                QStringList l = line.split(QString(": [HIGHLIGHT] "));
                if(l.size() < 2)
                    continue;

                p_lh->logMessageWithTime(l.constLast(),BlackChirp::LogHighlight,QDateTime::fromString(l.constFirst()));
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

                p_lh->logMessageWithTime(theLine,BlackChirp::LogNormal,QDateTime::fromString(l.constFirst()));
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

