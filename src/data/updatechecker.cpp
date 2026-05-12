#include "updatechecker.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QRegularExpression>

#include "loghandler.h"

using namespace Qt::Literals::StringLiterals;

// Two-step stringification so the preprocessor expands BC_RELEASE_VERSION
// from the compile definition before quoting it. Matches main.cpp / mainwindow.cpp.
#define _BC_STR(x) #x
#define BC_STRINGIFY(x) _BC_STR(x)

namespace {
// GitHub /releases/latest excludes pre-releases; /releases would include
// them. The endpoint is unauthenticated (60 req/hour/IP limit, ample for
// daily checks) and returns 404 when the repository has no stable release.
constexpr auto k_releasesUrl =
    "https://api.github.com/repos/kncrabtree/blackchirp/releases/latest";

// GitHub requires a User-Agent on every request; rejects with 403 otherwise.
QByteArray userAgent()
{
    return QByteArrayLiteral("blackchirp/")
         + QByteArray::number(BC_MAJOR_VERSION) + '.'
         + QByteArray::number(BC_MINOR_VERSION) + '.'
         + QByteArray::number(BC_PATCH_VERSION);
}

constexpr int k_timeoutMs = 10000;
}

UpdateChecker::UpdateChecker(QObject *parent)
    : QObject(parent),
      SettingsStorage(QStringList{BC::Key::Update::updateChecker.toString()}),
      p_nam(new QNetworkAccessManager(this))
{
}

UpdateChecker::~UpdateChecker() = default;

QString UpdateChecker::skippedVersion() const
{
    return get<QString>(BC::Key::Update::skipVersion);
}

void UpdateChecker::setSkippedVersion(const QString &version)
{
    set(BC::Key::Update::skipVersion, version, true);
}

QDateTime UpdateChecker::lastCheckedAt() const
{
    return get<QDateTime>(BC::Key::Update::lastRun);
}

void UpdateChecker::recordCheckCompleted()
{
    set(BC::Key::Update::lastRun, QDateTime::currentDateTimeUtc(), true);
}

UpdateChecker::Version UpdateChecker::localVersion()
{
    Version v;
    v.major = BC_MAJOR_VERSION;
    v.minor = BC_MINOR_VERSION;
    v.patch = BC_PATCH_VERSION;
    v.releaseTag = QString::fromLatin1(BC_STRINGIFY(BC_RELEASE_VERSION));
    return v;
}

UpdateChecker::Version UpdateChecker::parseTag(const QString &tag)
{
    // "v2.0.0", "v2.0.0-alpha", "2.0.0-rc1" — capture major.minor.patch
    // and an optional pre-release suffix delimited by '-'.
    static const QRegularExpression re(
        u"^v?(\\d+)\\.(\\d+)\\.(\\d+)(?:-([A-Za-z0-9.]+))?$"_s);
    const auto m = re.match(tag.trimmed());
    if(!m.hasMatch())
        return {};

    Version v;
    v.major = m.captured(1).toInt();
    v.minor = m.captured(2).toInt();
    v.patch = m.captured(3).toInt();
    v.releaseTag = m.captured(4);
    return v;
}

bool UpdateChecker::Version::isOlder(const Version &a, const Version &b)
{
    if(a.major != b.major) return a.major < b.major;
    if(a.minor != b.minor) return a.minor < b.minor;
    return a.patch < b.patch;
}

QString UpdateChecker::Version::toString() const
{
    QString base = u"%1.%2.%3"_s.arg(major).arg(minor).arg(patch);
    if(!releaseTag.isEmpty())
        base += u'-' + releaseTag;
    return base;
}

void UpdateChecker::checkNow()
{
    if(p_reply)
        return;

    QNetworkRequest req{QUrl(QLatin1StringView(k_releasesUrl))};
    req.setHeader(QNetworkRequest::UserAgentHeader, userAgent());
    req.setRawHeader("Accept", "application/vnd.github+json");
    req.setRawHeader("X-GitHub-Api-Version", "2022-11-28");
    req.setTransferTimeout(k_timeoutMs);

    p_reply = p_nam->get(req);
    connect(p_reply, &QNetworkReply::finished,
            this, &UpdateChecker::onReplyFinished);
}

void UpdateChecker::onReplyFinished()
{
    QNetworkReply *reply = p_reply;
    p_reply = nullptr;
    reply->deleteLater();

    const auto httpStatus = reply
        ->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();

    // 404: repository has no published non-prerelease yet. Treat as
    // up-to-date — the user is already on the most recent published
    // artifact regardless of what the local version says.
    if(httpStatus == 404)
    {
        bcDebug(u"Update check: no stable release published; treating as up to date."_s);
        recordCheckCompleted();
        emit upToDate();
        return;
    }

    if(reply->error() != QNetworkReply::NoError)
    {
        const QString reason = reply->errorString();
        bcDebug(u"Update check failed: "_s + reason);
        emit checkFailed(reason);
        return;
    }

    const QByteArray body = reply->readAll();
    QJsonParseError err{};
    const auto doc = QJsonDocument::fromJson(body, &err);
    if(err.error != QJsonParseError::NoError || !doc.isObject())
    {
        const QString reason = u"Malformed JSON response from GitHub: "_s
                             + err.errorString();
        bcDebug(reason);
        emit checkFailed(reason);
        return;
    }

    const auto obj = doc.object();
    const QString tag  = obj.value(u"tag_name"_s).toString();
    const QString url  = obj.value(u"html_url"_s).toString();
    const QString name = obj.value(u"name"_s).toString(tag);

    const Version remote = parseTag(tag);
    if(remote.major == 0 && remote.minor == 0 && remote.patch == 0
       && remote.releaseTag.isEmpty())
    {
        const QString reason = u"Could not parse release tag '"_s + tag + u"'"_s;
        bcDebug(reason);
        emit checkFailed(reason);
        return;
    }

    const Version local = localVersion();
    recordCheckCompleted();
    if(Version::isOlder(local, remote))
    {
        bcDebug(u"Update check: remote "_s + remote.toString()
              + u" is newer than local "_s + local.toString());
        emit updateAvailable(remote, QUrl(url), name);
    }
    else
    {
        bcDebug(u"Update check: local "_s + local.toString()
              + u" is up to date (remote "_s + remote.toString() + u')');
        emit upToDate();
    }
}
