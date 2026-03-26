#include <data/storage/settingsstorage.h>
#include <data/storage/applicationconfigmanager.h>
#include <gui/mainwindow.h>
#include <gui/dialog/applicationconfigdialog.h>
#include <gui/dialog/runtimehardwareconfigdialog.h>
#include <gui/plot/curveappearancepresetmanager.h>
#include <data/processing/parsers/fileparserregistry.h>
#include <data/processing/parsers/spcatparser.h>
#include <data/processing/parsers/xiamparser.h>
#include <data/processing/parsers/genericxyparser.h>

#include <memory>

#include <QApplication>
#include <QMessageBox>
#include <QDir>
#include <QSharedMemory>
#include <QLocalServer>
#include <QLocalSocket>
#include <QtGlobal>

#if QT_VERSION <= 0x060000
#include <QTextCodec>
#endif

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
#if QT_VERSION <= 0x060000
    QTextCodec::setCodecForLocale(QTextCodec::codecForName("UTF-8"));
#endif

    const QString appName = QString("Blackchirp");

    //QSettings information
    QApplication::setApplicationName(appName);
    QApplication::setOrganizationDomain(QString("crabtreelab.ucdavis.edu"));
    QApplication::setOrganizationName(QString("CrabtreeLab"));

    SettingsStorage s;
    auto f = ApplicationConfigManager::instance().getOptionValue(BC::Key::AppConfig::appFont).value<QFont>();
    a.setFont(f);

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


    auto savePath = s.get(BC::Key::savePath,QString(""));

    if(savePath.isEmpty())
    {
        ApplicationConfigDialog configDialog(true);
        if(configDialog.exec() == QDialog::Rejected)
            return 0;

        QMessageBox::information(nullptr, QString("Hardware Selection"),
                                 QString(
R"000(Next, select the instrument models connected to your system. For each hardware type, you can add a profile for your instrument model and activate it.

If you are unsure which hardware to select, you can leave the defaults (virtual instruments) and configure this later from the Hardware > Configure Hardware menu.
)000"));

        RuntimeHardwareConfigDialog hwDialog;
        hwDialog.exec();
        // Fall through to normal startup — the saved hardware config will be loaded by
        // HardwareManager::initialize(). If any hardware fails to connect, the
        // Communication dialog will open automatically.
    }

    qRegisterMetaType<std::shared_ptr<Experiment>>();
    qRegisterMetaType<LogHandler::MessageCode>();
    qRegisterMetaType<Fid>("Fid");
    qRegisterMetaType<FidList>("FidList");
    qRegisterMetaType<FtWorker::FidProcessingSettings>("FtWorker::FidProcessingSettings");
    qRegisterMetaType<Ft>("Ft");
    qRegisterMetaType<QVector<QPointF> >("QVector<QPointF>");
    qRegisterMetaType<QVector<double>>("QVector<double>");
    qRegisterMetaType<QVector<qint8>>("QVector<qint8>");
    qRegisterMetaType<QHash<RfConfig::ClockType, RfConfig::ClockFreq>>();
    qRegisterMetaType<PulseGenConfig>();
    qRegisterMetaType<PulseGenConfig::Setting>();
    qRegisterMetaType<AuxDataStorage::AuxDataMap>();
    qRegisterMetaType<FlowConfig::FlowChSetting>();
    qRegisterMetaType<LifTrace>("LifTrace");
    qRegisterMetaType<LifConfig>("LifConfig");

#ifndef QT_DEBUG
    gsl_set_error_handler_off();
#else
    //comment this line out to enable the gsl error handler
    gsl_set_error_handler_off();
#endif

    // Register catalog parsers
    auto registry = FileParserRegistry::instance();
    registry->registerParser(std::make_unique<SPCATParser>());
    registry->registerParser(std::make_unique<XIAMParser>());
    registry->registerParser(std::make_unique<GenericXYParser>());

    MainWindow w;
    QApplication::connect(ls.get(),&QLocalServer::newConnection,[&w](){
        w.setWindowState(Qt::WindowMaximized|Qt::WindowActive);
        w.raise();
        w.show();
    });

    w.showMaximized();
    w.initializeHardware();
    int ret = a.exec();
    
    // Cleanup global instances before application shutdown
    FileParserRegistry::cleanup();
    CurveAppearancePresetManager::cleanup();

    return ret;
}
