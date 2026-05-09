#include <data/storage/settingsstorage.h>
#include "viewermainwindow.h"
#include <gui/plot/curveappearancepresetmanager.h>
#include <data/processing/parsers/fileparserregistry.h>
#include <data/processing/parsers/spcatparser.h>
#include <data/processing/parsers/xiamparser.h>
#include <data/processing/parsers/genericxyparser.h>

#include <memory>

#include <QApplication>
#include <QCommandLineParser>
#include <QMessageBox>
#include <QDir>
#include <QtGlobal>

#if QT_VERSION <= 0x060000
#include <QTextCodec>
#endif

#include <gsl/gsl_errno.h>

#ifdef Q_OS_UNIX
#include <signal.h>
#endif

#ifdef Q_OS_WIN
#include <cstdio>
#include <windows.h>
#endif

#define _BC_STR(x) #x
#define BC_STRINGIFY(x) _BC_STR(x)

int main(int argc, char *argv[])
{
#ifdef Q_OS_UNIX
    signal(SIGPIPE,SIG_IGN);
#endif
#ifdef Q_OS_WIN
    // GUI-subsystem binaries don't inherit a console; reattach to the
    // parent shell's so --version / --help can write to stdout.
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        freopen("CONOUT$", "w", stdout);
        freopen("CONOUT$", "w", stderr);
    }
#endif

    QApplication a(argc, argv);
#if QT_VERSION <= 0x060000
    QTextCodec::setCodecForLocale(QTextCodec::codecForName("UTF-8"));
#endif

    // Handle --version / --help before any setup runs. Version string
    // embeds BCV_BUILD_VERSION (git SHA at compile time) so it can be
    // matched against the symbol artifacts captured in CI.
    QCoreApplication::setApplicationVersion(QString::fromLatin1(
        "%1.%2.%3-%4 (build %5)")
        .arg(BCV_MAJOR_VERSION).arg(BCV_MINOR_VERSION).arg(BCV_PATCH_VERSION)
        .arg(QLatin1StringView(BC_STRINGIFY(BCV_RELEASE_VERSION)))
        .arg(QLatin1StringView(BCV_BUILD_VERSION)));
    {
        QCommandLineParser parser;
        parser.setApplicationDescription(
            QStringLiteral("Blackchirp data viewer"));
        parser.addHelpOption();
        parser.addVersionOption();
        parser.process(a);
    }

    const QString appName = QString("Blackchirp Viewer");

    //QSettings information
    QApplication::setApplicationName(appName);
    QApplication::setOrganizationDomain(QString("crabtreelab.ucdavis.edu"));
    QApplication::setOrganizationName(QString("CrabtreeLab"));

    SettingsStorage s;
    auto f = s.get(BC::Key::appFont,QFont(QString("sans-serif"),8));
    a.setFont(f);

    {
        QSettings vset;
        vset.setFallbacksEnabled(false);
        vset.beginGroup(BC::Key::BC);
        vset.setValue(BC::Key::versionMajor, BCV_MAJOR_VERSION);
        vset.setValue(BC::Key::versionMinor, BCV_MINOR_VERSION);
        vset.setValue(BC::Key::versionPatch, BCV_PATCH_VERSION);
        vset.setValue(BC::Key::versionRelease, QLatin1StringView(BC_STRINGIFY(BCV_RELEASE_VERSION)));
        vset.endGroup();
        vset.sync();
    }

    // Register meta types for Qt signal/slot system
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

    ViewerMainWindow w;
    w.show();

    int ret = a.exec();
    
    // Cleanup global instances before application shutdown
    FileParserRegistry::cleanup();
    CurveAppearancePresetManager::cleanup();

    return ret;
}