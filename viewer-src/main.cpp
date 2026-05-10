#include <data/storage/settingsstorage.h>
#include "viewermainwindow.h"
#include <gui/plot/curveappearancepresetmanager.h>
#include <data/processing/parsers/fileparserregistry.h>
#include <data/processing/parsers/spcatparser.h>
#include <data/processing/parsers/xiamparser.h>
#include <data/processing/parsers/genericxyparser.h>

#include <cstdio>
#include <cstring>
#include <memory>

#include <QApplication>
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
#include <windows.h>
#endif

#define _BC_STR(x) #x
#define BC_STRINGIFY(x) _BC_STR(x)

int main(int argc, char *argv[])
{
#ifdef Q_OS_WIN
    // GUI-subsystem binaries don't inherit a console; reattach to the
    // parent shell's so --version / --help can write to stdout — but
    // only when stdout has no real handle yet, so a `Start-Process
    // -RedirectStandardOutput` or shell `>` doesn't get clobbered.
    {
        HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD type = (h == NULL || h == INVALID_HANDLE_VALUE)
                       ? FILE_TYPE_UNKNOWN
                       : GetFileType(h);
        if (type == FILE_TYPE_UNKNOWN
            && AttachConsole(ATTACH_PARENT_PROCESS)) {
            freopen("CONOUT$", "w", stdout);
            freopen("CONOUT$", "w", stderr);
        }
    }
#endif

    // Handle --version / --help before constructing QApplication so the
    // call doesn't need a Qt platform plugin (xcb aborts on a headless
    // box). Version string embeds BCV_BUILD_VERSION (git SHA at compile
    // time) so it can be matched against the symbol artifacts captured
    // in CI.
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (std::strcmp(arg, "--version") == 0 || std::strcmp(arg, "-v") == 0) {
            std::printf("blackchirp-viewer %d.%d.%d-%s (build %s)\n",
                        BCV_MAJOR_VERSION, BCV_MINOR_VERSION, BCV_PATCH_VERSION,
                        BC_STRINGIFY(BCV_RELEASE_VERSION), BCV_BUILD_VERSION);
            return 0;
        }
        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            std::printf(
                "Usage: blackchirp-viewer [options]\n"
                "Blackchirp data viewer\n\n"
                "Options:\n"
                "  -h, --help     Display this help.\n"
                "  -v, --version  Display version information.\n");
            return 0;
        }
    }

#ifdef Q_OS_UNIX
    signal(SIGPIPE,SIG_IGN);
#endif

    QApplication a(argc, argv);
#if QT_VERSION <= 0x060000
    QTextCodec::setCodecForLocale(QTextCodec::codecForName("UTF-8"));
#endif

    // applicationName is the QSettings storage key, so it must match
    // blackchirp's ("Blackchirp<major>") for the viewer to read settings
    // that the acquisition app wrote. applicationDisplayName drives the
    // window-manager / taskbar string, which legitimately differs.
    const QString settingsName = QString("Blackchirp%1").arg(BC_MAJOR_VERSION);
    const QString displayName = QString("Blackchirp Viewer");

    QApplication::setApplicationName(settingsName);
    QApplication::setApplicationDisplayName(displayName);
    QApplication::setOrganizationDomain(QString("crabtreelab.ucdavis.edu"));
    QApplication::setOrganizationName(QString("CrabtreeLab"));

    // Tie the running window to share/applications/blackchirp-viewer.desktop
    // so Wayland (xdg_toplevel.app_id) and X11 (WM_CLASS) can both look up
    // the Icon= entry. applicationName is "Blackchirp<major>" — shared with
    // the acquisition app for QSettings — so it can't drive this lookup.
    QGuiApplication::setDesktopFileName(QString("blackchirp-viewer"));

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