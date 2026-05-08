#include "crashhandler.h"
#include "crashhandler_p.h"

#include <atomic>
#include <cstdio>

#include <algorithm>

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QString>
#include <QStringList>
#include <QByteArray>
#include <QtGlobal>

using namespace Qt::Literals::StringLiterals;

#include <data/storage/settingsstorage.h>

#define _BC_CRASH_STR(x) #x
#define BC_CRASH_STRINGIFY(x) _BC_CRASH_STR(x)

namespace {

std::atomic<int> g_activeExperiment{0};
std::atomic<bool> g_installed{false};

// Build identity formatted once at install() time. Read by the
// platform handler from non-allocating context.
char g_buildHeader[512]{};

// UTF-8 null-terminated path to <savePath>/log/crashes/. Updated only
// from non-handler context (CrashHandler::reopen). The handler reads
// it via a relaxed load.
QByteArray g_crashesDir;

// UTF-8 null-terminated per-run filename stem (e.g.,
// "crash-20260508-022919-5c8837e"). Used by both the per-platform
// reopenPlatform() and by collectPriorArtifacts() to filter out the
// current process's own files.
QByteArray g_currentBasename;

// Null-terminated build SHA copy so the handler can read it without
// allocating; the BC_BUILD_VERSION macro is a string literal but the
// handler benefits from a dedicated symbol it can take the address of.
char g_buildSha[64]{};

}

namespace BC::CrashInternal {

const char *buildHeaderUtf8()     { return g_buildHeader; }
const char *crashesDirUtf8()      { return g_crashesDir.constData(); }
const char *currentBasenameUtf8() { return g_currentBasename.constData(); }
const char *buildVersionUtf8()    { return g_buildSha; }
int activeExperimentNumber()      { return g_activeExperiment.load(std::memory_order_relaxed); }

}

namespace CrashHandler {

void install()
{
    bool expected = false;
    if(!g_installed.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        return;

    std::snprintf(g_buildHeader, sizeof(g_buildHeader),
                  "Blackchirp %d.%d.%d-%s (build %s)\n"
                  "Qt %s\n",
                  BC_MAJOR_VERSION, BC_MINOR_VERSION, BC_PATCH_VERSION,
                  BC_CRASH_STRINGIFY(BC_RELEASE_VERSION),
                  BC_BUILD_VERSION,
                  qVersion());

    std::snprintf(g_buildSha, sizeof(g_buildSha), "%s", BC_BUILD_VERSION);

    BC::CrashInternal::installPlatformHandlers();
}

void reopen(const QString &savePath)
{
    if(!g_installed.load(std::memory_order_acquire))
        return;
    if(savePath.isEmpty())
        return;

    QDir d(savePath);
    if(!d.exists())
        return;
    if(!d.exists(BC::Key::logDir) && !d.mkpath(BC::Key::logDir))
        return;
    if(!d.cd(BC::Key::logDir))
        return;
    if(!d.exists("crashes"_L1) && !d.mkpath("crashes"_L1))
        return;
    if(!d.cd("crashes"_L1))
        return;

    g_crashesDir = (d.absolutePath() + u'/').toUtf8();

    auto stamp = QDateTime::currentDateTimeUtc().toString("yyyyMMdd-HHmmss"_L1);
    QString sha = QString::fromUtf8(g_buildSha);
    if(sha.size() > 7) sha = sha.left(7);
    g_currentBasename = (u"crash-"_s + stamp + u'-' + sha).toUtf8();

    BC::CrashInternal::reopenPlatform();
}

void setActiveExperiment(int num)
{
    g_activeExperiment.store(num, std::memory_order_relaxed);
}

void shutdown()
{
    if(!g_installed.load(std::memory_order_acquire))
        return;
    BC::CrashInternal::shutdownPlatform();
}

QString crashesDirectory()
{
    return QString::fromUtf8(g_crashesDir);
}

QString artifactTimestamp(const QString &path)
{
    QFileInfo fi(path);
    auto base = fi.completeBaseName();
    static const auto prefix = "crash-"_L1;
    constexpr int tsLen = 15; // yyyyMMdd-HHmmss
    if(!base.startsWith(prefix) || base.size() < prefix.size() + tsLen)
        return {};
    return base.mid(prefix.size(), tsLen);
}

QStringList collectPriorArtifacts()
{
    QStringList prior;
    if(g_crashesDir.isEmpty())
        return prior;

    QDir d(QString::fromUtf8(g_crashesDir));
    if(!d.exists())
        return prior;

    auto current = QString::fromUtf8(g_currentBasename);
    QStringList filters;
    filters << "crash-*.log"_L1
            << "crash-*.dmp"_L1;
    auto entries = d.entryInfoList(filters,
                                   QDir::Files | QDir::NoDotAndDotDot,
                                   QDir::NoSort);
    for(const auto &fi : entries)
    {
        if(!current.isEmpty() && fi.completeBaseName() == current)
            continue;
        if(fi.size() == 0)
        {
            QFile::remove(fi.absoluteFilePath());
            continue;
        }
        prior << fi.absoluteFilePath();
    }

    std::sort(prior.begin(), prior.end(),
              [](const QString &a, const QString &b) {
                  return artifactTimestamp(a) > artifactTimestamp(b);
              });
    return prior;
}

}
