#include <src/gui/mainwindow.h>
#include <QApplication>
#include <QFile>
#include <QMessageBox>
#include <QSettings>
#include <QDesktopServices>
#include <QDateTime>
#include <QDir>
#include <QProcessEnvironment>
#include <gsl/gsl_errno.h>

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
    const QString lockFileName = QString("lock_blackchirp.lock");
#ifdef Q_OS_MSDOS
    QString appDataPath = QString("c:/data");
#else
    QString appDataPath = QString("/home/data");
#endif

    QProcessEnvironment se = QProcessEnvironment::systemEnvironment();
    if(se.contains(QString("BC_DATADIR")))
    {
        QString ad = se.value(QString("BC_DATADIR"));
        if(ad.endsWith(QChar('/')))
            ad.chop(1);

        appDataPath = ad;
    }

    QApplication::setApplicationName(appName);
    QApplication::setOrganizationDomain(QString("crabtreelab.ucdavis.edu"));
    QApplication::setOrganizationName(QString("CrabtreeLab"));
    QSettings::setPath(QSettings::NativeFormat,QSettings::SystemScope,appDataPath);
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

    //look for lock files from other applications that need same hardware resources

    //list containing lockfile names and application names
    QList<QPair<QString,QString> > incompatibleApps;
    //add other apps here
    //if the app is from a different organization (e.g. CfA Spectroscopy Lab instead of CrabtreeLab), enter full path to lockfile and prepend with !
    incompatibleApps.append(qMakePair(lockFileName,appName));
    incompatibleApps.append(qMakePair(QString("lock_qtftm.lock"),QString("QtFTM")));


    QFile lockFile;
    for(int i=0;i<incompatibleApps.size();i++)
    {
        QString title = incompatibleApps.at(i).second;
        QString fileName;
        if(incompatibleApps.at(i).first.startsWith(QString("!")))
            fileName = incompatibleApps.at(i).first.mid(1);
        else
            fileName = QString("%1/%2").arg(lockFilePath).arg(incompatibleApps.at(i).first);

        lockFile.setFileName(fileName);
        qint64 pid = 0;
        bool ok = false;
        if(lockFile.exists())
        {
            QString uName = QFileInfo(lockFile).owner();

            if(lockFile.open(QIODevice::ReadOnly))
                pid = lockFile.readLine().trimmed().toInt(&ok);

            if(!ok)
                QMessageBox::critical(nullptr,QString("%1 Error").arg(appName),QString("An instance of %1 is running as user %2, and it must be closed before %3 be started.\n\nIf you are sure no other instance is running, delete the lock file (%4) and restart.").arg(title).arg(uName).arg(appName).arg(fileName));
            else
                QMessageBox::critical(nullptr,QString("%1 Error").arg(appName),QString("An instance of %1 is running under PID %2 as user %3, and it must be closed before %4 can be started.\n\nIf process %2 has been terminated, delete the lock file (%5) and restart.").arg(title).arg(pid).arg(uName).arg(appName).arg(fileName));

            if(lockFile.isOpen())
                lockFile.close();

            return -1;
        }
    }

    QString fileName = QString("%1/%2").arg(lockFilePath).arg(lockFileName);
    lockFile.setFileName(fileName);

    if(!lockFile.open(QIODevice::WriteOnly))
    {
        QMessageBox::critical(nullptr,QString("%1 Error").arg(appName),QString("Could not write lock file to %1. Ensure this location has write permissions enabled for your user.").arg(fileName));
        return -1;
    }
    lockFile.write(QString("%1\n\nStarted by user %2 at %3.").arg(a.applicationPid()).arg(QFileInfo(lockFile).owner()).arg(QDateTime::currentDateTime().toString(Qt::ISODate)).toLatin1());
    lockFile.setPermissions(QFileDevice::ReadOwner|QFileDevice::WriteOwner|QFileDevice::ReadGroup|QFileDevice::ReadOther);
    lockFile.close();

    qRegisterMetaType<Experiment>("Experiment");
    qRegisterMetaType<Fid>("Fid");
    qRegisterMetaType<FidList>("FidList");
    qRegisterMetaType<FtWorker::FidProcessingSettings>("FtWorker::FidProcessingSettings");
    qRegisterMetaType<Ft>("Ft");
    qRegisterMetaType<FtmwConfig>("FtmwConfig");
    qRegisterMetaType<RfConfig>("RfConfig");
    qRegisterMetaType<QVector<QPointF> >("QVector<QPointF>");
    qRegisterMetaType<QVector<double>>("Vector<double>");
    qRegisterMetaType<QList<QPair<QString,QVariant> >>("QList<QPair<QString,QVariant> >");
    qRegisterMetaType<BlackChirp::PulseSetting>("BlackChirp::PulseSetting");
    qRegisterMetaType<BlackChirp::PulseRole>("BlackChirp::PulseRole");
    qRegisterMetaType<PulseGenConfig>("PulseGenConfig");
    qRegisterMetaType<BlackChirp::FtWindowFunction>("BlackChirp::FtWindowFunction");
    qRegisterMetaType<QList<QPointF>>("QList<QPointF>");
    qRegisterMetaType<BlackChirp::ClockType>("BlackChirp::ClockType");
    qRegisterMetaType<BlackChirp::Sideband>("BlackChirp::Sideband");
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
    w.showMaximized();
    w.initializeHardware();
    int ret = a.exec();
    lockFile.remove();
    return ret;
}
