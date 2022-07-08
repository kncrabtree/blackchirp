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
#include <gui/widget/experimentsummarywidget.h>
#include <gui/widget/auxdataviewwidget.h>
#include <data/storage/blackchirpcsv.h>
#include <data/loghandler.h>

#ifdef BC_LIF
#include <modules/lif/gui/lifdisplaywidget.h>
#endif

ExperimentViewWidget::ExperimentViewWidget(int num, QString path, QWidget *parent) : QWidget(parent), p_ftmw(nullptr), p_lh(nullptr)
{
    pu_experiment = std::make_unique<Experiment>(num,path);
    setWindowFlags(Qt::Window);
    setWindowTitle(QString("Experiment %1").arg(num));
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowIcon(QIcon(QString(":/icons/bc_logo_small.png")));


    QVBoxLayout *vbl = new QVBoxLayout;

    if(pu_experiment->d_number < 1)
    {
        QLabel *errLabel = new QLabel(pu_experiment->d_errorString);
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

    if(pu_experiment->ftmwEnabled())
    {
        QWidget *ftmw = buildFtmwWidget(path);
        if(ftmw != nullptr)
            p_tabWidget->addTab(ftmw,QIcon(QString(":/icons/chirp.png")),QString("CP-FTMW"));
    }

#ifdef BC_LIF
    if(pu_experiment->lifEnabled())
    {
        QWidget *lif = buildLifWidget();
        if(lif != nullptr)
            p_tabWidget->addTab(lif,QIcon(QString(":/icons/laser.png")),QString("LIF"));
    }
#endif

    QWidget *tracking = buildTrackingWidget();
    if(tracking != nullptr)
        p_tabWidget->addTab(tracking,QIcon(QString(":/icons/dataplots.png")),QString("Tracking"));

    QWidget *log = buildLogWidget(path);
    if(log != nullptr)
        p_tabWidget->addTab(log,QIcon(QString(":/icons/log.png")),QString("Log"));

    vbl->addWidget(p_tabWidget);
    setLayout(vbl);
}

QSize ExperimentViewWidget::sizeHint() const
{
    return QSize(1024,768);
}

QWidget *ExperimentViewWidget::buildHeaderWidget()
{
    QWidget *hdr = new QWidget();
    QVBoxLayout *hdrvl = new QVBoxLayout();

    auto esw = new ExperimentSummaryWidget;
    esw->setExperiment(pu_experiment.get());
    hdrvl->addWidget(esw);
    hdr->setLayout(hdrvl);

    return hdr;
}

QWidget *ExperimentViewWidget::buildFtmwWidget(QString path)
{
    QWidget *out = nullptr;
    p_ftmw = nullptr;
    if(pu_experiment->ftmwEnabled())
    {
        out = new QWidget;
        QVBoxLayout *vbl = new QVBoxLayout;
        p_ftmw = new FtmwViewWidget(out,path);
        vbl->addWidget(p_ftmw);
        out->setLayout(vbl);

        p_ftmw->prepareForExperiment(*pu_experiment);
        p_ftmw->updateBackups();
        p_ftmw->experimentComplete();
    }

    return out;
}

#ifdef BC_LIF
QWidget *ExperimentViewWidget::buildLifWidget()
{
    QWidget *out = nullptr;
    if(pu_experiment->lifEnabled())
    {
        out = new QWidget;
        QVBoxLayout *vbl = new QVBoxLayout;
        LifDisplayWidget *lif = new LifDisplayWidget(out);
        vbl->addWidget(lif);
        out->setLayout(vbl);

        lif->prepareForExperiment(*pu_experiment);
        lif->updatePoint();
    }

    return out;
}
#endif

QWidget *ExperimentViewWidget::buildTrackingWidget()
{
    //tracking page
    QWidget *tracking = nullptr;
    auto auxData = pu_experiment->auxData()->savedData();
    if(auxData.size() > 0)
    {
        tracking = new QWidget;
        QVBoxLayout *trackingvl = new QVBoxLayout;

        AuxDataViewWidget *tvw = new AuxDataViewWidget(BC::Key::auxDataWidget,tracking,true);
        trackingvl->addWidget(tvw);

        for(auto &[ts,m] : auxData)
            tvw->pointUpdated(m,ts);

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

    auto csv = std::make_shared<BlackchirpCSV>(pu_experiment->d_number,path);

    QFile f(BlackchirpCSV::exptDir(pu_experiment->d_number,path).absoluteFilePath("log.csv"));
    if(f.open(QIODevice::ReadOnly))
    {
        while(!f.atEnd())
        {
            auto line = csv->readLine(f);
            if(line.isEmpty() || line.size() != 4 || line.constFirst().toString().contains("Timestamp"))
                continue;

            auto dt = QDateTime::fromMSecsSinceEpoch(line.at(1).toLongLong());
            auto code = line.at(2).value<LogHandler::MessageCode>();
            auto msg = line.at(3).toString();

            p_lh->logMessageWithTime(msg,code,dt);
        }
    }

    return log;

}

