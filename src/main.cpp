#include <data/storage/settingsstorage.h>
#include <gui/mainwindow.h>
#include <gui/dialog/bcsavepathdialog.h>

#include <memory>

#include <QApplication>
#include <QMessageBox>
#include <QDir>
#include <QSharedMemory>
#include <QLocalServer>
#include <QLocalSocket>
#include <QProcess>

#include <gsl/gsl_errno.h>

#ifdef Q_OS_UNIX
#include <signal.h>
#endif

int main(int argc, char *argv[])
{
#ifdef Q_OS_UNIX
    signal(SIGPIPE,SIG_IGN);
#endif

    QApplication a(argc, argv);
    a.setFont(QFont(QString("sans-serif"),8));



    const QString appName = QString("Blackchirp");

    std::unique_ptr<QSharedMemory> m;
    std::unique_ptr<QLocalServer> ls;
    struct Mem {
        char name[32];
    };

#ifdef Q_OS_UNIX
    //This will delete the shared memory if Blackchirp crashed last time
    m = std::make_unique<QSharedMemory>(appName);
    m->attach();
    m.reset();
#endif
    m = std::make_unique<QSharedMemory>(appName);
    if(m->create(sizeof(Mem)))
    {
        if(!m->lock())
            return -255;

        auto mem = static_cast<Mem*>(m->data());
        sprintf(mem->name,"Blackchirp");

        QLocalServer::removeServer(appName);
        ls = std::make_unique<QLocalServer>();
        ls->setSocketOptions(QLocalServer::WorldAccessOption);
        ls->listen(appName);
        m->unlock();
    }
    else
    {
        if(m->error() == QSharedMemory::AlreadyExists)
        {
            auto socket = std::make_unique<QLocalSocket>();
            socket->connectToServer(appName);
            socket->waitForConnected(1000);
            return 0;
        }

        return -255;
    }

    //QSettings information
    QApplication::setApplicationName(appName);
    QApplication::setOrganizationDomain(QString("crabtreelab.ucdavis.edu"));
    QApplication::setOrganizationName(QString("CrabtreeLab"));

    SettingsStorage s;
    auto savePath = s.get(BC::Key::savePath,QString(""));

    if(savePath.isEmpty())
    {
        QMessageBox::information(nullptr,QString("Welcome to Blackchirp!"),
                                 QString(
R"000(It appears you are running Blackchirp for the first time, or you have just upgraded from a previous version. To get started, you first need to choose a directory where Blackchirp will store its data. In the directory you choose, four folders will be created:

        experiments - Location where all experimental data are recorded
        log - Location of log messages
        rollingdata - Location for temporal monitoring data
        textexports - Default location for manually exported csv files

Please note that if you are upgrading from an old version (<1.0.0) of Blackchirp, it is not recommended that you use the same storage folder as your old version, as all file formats have changed.
)000"));

        BCSavePathDialog d;
        int ret = d.exec();
        if(ret == QDialog::Rejected)
            return 0;

        MainWindow w;
        w.initializeHardware();

        QMessageBox::information(nullptr,QString("Hardware Configuration"),QString(
R"000(Next, you can configure the communication settings for the hardware connected to your computer. After exiting the following dialog, Blackchirp will restart and will use the settings you have chosen. You may change these later in the  Hardware > Communication menu.
)000"));

        w.launchCommunicationDialog(false);

        qApp->quit();
        QProcess::startDetached(qApp->arguments().constFirst(),qApp->arguments().mid(1));

    }

    qRegisterMetaType<std::shared_ptr<Experiment>>();
    qRegisterMetaType<LogHandler::MessageCode>();
    qRegisterMetaType<Fid>("Fid");
    qRegisterMetaType<FidList>("FidList");
    qRegisterMetaType<FtWorker::FidProcessingSettings>("FtWorker::FidProcessingSettings");
    qRegisterMetaType<Ft>("Ft");
    qRegisterMetaType<QVector<QPointF> >("QVector<QPointF>");
    qRegisterMetaType<QVector<double>>("Vector<double>");
    qRegisterMetaType<QHash<RfConfig::ClockType, RfConfig::ClockFreq>>();
    qRegisterMetaType<PulseGenConfig>();
    qRegisterMetaType<PulseGenConfig::Setting>();
    qRegisterMetaType<AuxDataStorage::AuxDataMap>();
    qRegisterMetaType<FlowConfig::FlowChSetting>();
#ifdef BC_LIF
    qRegisterMetaType<LifTrace>("LifTrace");
    qRegisterMetaType<LifConfig>("LifConfig");
    qRegisterMetaType<BlackChirp::LifScopeConfig>("BlackChirp::LifScopeConfig");
#endif
#ifdef BC_MOTOR
    qRegisterMetaType<MotorScan::MotorAxis>("MotorScan::MotorAxis");
    qRegisterMetaType<MotorScan>("MotorScan");
#endif

#ifndef QT_DEBUG
    gsl_set_error_handler_off();
#else
    //comment this line out to enable the gsl error handler
    gsl_set_error_handler_off();
#endif

    MainWindow w;
    QApplication::connect(ls.get(),&QLocalServer::newConnection,[&w](){
        w.setWindowState(Qt::WindowMaximized|Qt::WindowActive);
        w.raise();
        w.show();
    });

    w.showMaximized();
    w.initializeHardware();
    int ret = a.exec();

    return ret;
}
