#include <gui/mainwindow.h>
#include <QApplication>
#include <QFile>
#include <QMessageBox>
#include <QSettings>
#include <QDesktopServices>
#include <QDateTime>
#include <QDir>
#include <QProcessEnvironment>
#include <gsl/gsl_errno.h>
#include <QSharedMemory>
#include <QLocalServer>
#include <QLocalSocket>

#ifdef Q_OS_UNIX
#include <sys/stat.h>
#include <signal.h>
#endif

int main(int argc, char *argv[])
{
    //all files (besides lock file) created with this program should have 664 permissions (directories 775)
#ifdef Q_OS_UNIX
    umask(S_IWOTH);
    signal(SIGPIPE,SIG_IGN);
#endif

    QApplication a(argc, argv);
    a.setFont(QFont(QString("sans-serif"),8));


    //QSettings information
    const QString appName = QString("BlackChirp");

    QSharedMemory *m;
    QLocalServer *ls;
    struct Mem {
        char name[32];
    };

#ifdef Q_OS_UNIX
    //This will delete the shared memory if Blackchirp crashed last time
    m = new QSharedMemory(appName);
    m->attach();
    delete m;
#endif
    m = new QSharedMemory(appName);
    if(m->create(sizeof(Mem)))
    {
        if(!m->lock())
        {
            delete m;
            return -255;
        }

        auto mem = static_cast<Mem*>(m->data());
        sprintf(mem->name,"Blackchirp");

        QLocalServer::removeServer(appName);
        ls = new QLocalServer;
        ls->setSocketOptions(QLocalServer::WorldAccessOption);
        ls->listen(appName);
        m->unlock();
    }
    else
    {
        if(m->error() == QSharedMemory::AlreadyExists)
        {
            auto socket = new QLocalSocket;
            socket->connectToServer(appName);
            socket->waitForConnected(1000);
            delete m;
            delete socket;
            return 0;
        }
    }


#ifdef Q_OS_MSDOS
    QString appDataPath = QString("c:/data");
#else
    QString appDataPath = QString("/home/data");
#endif
    QApplication::setApplicationName(appName);
    QApplication::setOrganizationDomain(QString("crabtreelab.ucdavis.edu"));
    QApplication::setOrganizationName(QString("CrabtreeLab"));
    QSettings::setPath(QSettings::NativeFormat,QSettings::SystemScope,appDataPath);

    QProcessEnvironment se = QProcessEnvironment::systemEnvironment();
    if(se.contains(QString("BC_DATADIR")))
    {
        QString ad = se.value(QString("BC_DATADIR"));
        if(ad.endsWith(QChar('/')))
            ad.chop(1);

        appDataPath = ad;
    }
    const QString lockFilePath = QString("%1/%2").arg(appDataPath).arg(QApplication::organizationName());

    //test to make sure data path is writable
    QDir home(appDataPath);
    if(!home.exists())
    {
        QMessageBox::critical(nullptr,QString("%1 Error").arg(appName),QString("The directory %1 does not exist!\n\nIn order to run %2, the directory %1 must exist and be writable by all users.").arg(appDataPath).arg(appName));
        return -1;
    }

    if(!home.cd(QApplication::organizationName()))
    {
        if(!home.mkpath(QApplication::organizationName()))
            QMessageBox::critical(nullptr,QString("%1 Error").arg(appName),QString("Could not create folder %1 for application configuration storage. Check the permissions of the path and try again.").arg(lockFilePath));
    }

    QFile testFile(QString("%1/test").arg(home.absolutePath()));
    if(!testFile.open(QIODevice::WriteOnly))
    {
        QMessageBox::critical(nullptr,QString("%1 Error").arg(appName),QString("Could not write to directory %1!\n\nIn order to run %2, the directory %1 must exist and be writable by all users.").arg(appDataPath).arg(appName));
        return -1;
    }
    testFile.close();
    testFile.remove();

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.setValue(QString("lastRun"),QDateTime::currentDateTime().toString(Qt::ISODate));
    s.setValue(QString("savePath"),QString("%1/%2").arg(appDataPath).arg(appName.toLower()));

    //create needed directories
    QDir saveDir(s.value(QString("savePath")).toString());
    saveDir.mkpath(QString("log"));

    qRegisterMetaType<std::shared_ptr<Experiment>>();
    qRegisterMetaType<Fid>("Fid");
    qRegisterMetaType<FidList>("FidList");
    qRegisterMetaType<FtWorker::FidProcessingSettings>("FtWorker::FidProcessingSettings");
    qRegisterMetaType<Ft>("Ft");
    qRegisterMetaType<QVector<QPointF> >("QVector<QPointF>");
    qRegisterMetaType<QVector<double>>("Vector<double>");
    qRegisterMetaType<QList<QPair<QString,QVariant> >>("QList<QPair<QString,QVariant> >");
    qRegisterMetaType<QHash<RfConfig::ClockType, RfConfig::ClockFreq>>();
    qRegisterMetaType<PulseGenConfig>("PulseGenConfig");
    qRegisterMetaType<QList<QPointF>>("QList<QPointF>");
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
    QApplication::connect(ls,&QLocalServer::newConnection,[&w](){
        w.setWindowState(Qt::WindowMaximized|Qt::WindowActive);
        w.raise();
    });

    w.showMaximized();
    w.initializeHardware();
    int ret = a.exec();
//    lockFile.remove();
    delete m;
    delete ls;
    return ret;
}
