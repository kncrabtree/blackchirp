#ifndef UPDATECHECKER_H
#define UPDATECHECKER_H

#include <QDateTime>
#include <QLatin1StringView>
#include <QObject>
#include <QString>
#include <QUrl>

#include <data/storage/settingsstorage.h>

class QNetworkAccessManager;
class QNetworkReply;

/*!
 * \brief Asynchronously checks GitHub for a newer Blackchirp release.
 *
 * UpdateChecker queries the project's GitHub repository for the latest
 * non-prerelease tag and compares it against the application's compiled-in
 * version. The check is one-shot per call to checkNow(); the caller decides
 * whether to invoke it manually (Help menu action) or automatically
 * (throttled to once per day on startup).
 *
 * The GitHub /releases/latest endpoint excludes pre-releases by definition,
 * so alpha/beta/rc tags published on the same repository never surface to
 * users running a stable build. During a pre-release-only period the
 * endpoint returns 404, which is treated as "up to date" (the user is
 * already on the most recent published artifact).
 *
 * Network failures (DNS, TLS, timeout, non-200 response other than 404) are
 * logged at Debug severity via bcDebug and surfaced only through the
 * checkFailed() signal. Callers that want a silent startup check should
 * ignore checkFailed; callers acting on a user-initiated request should
 * connect to it to display an error.
 */
class UpdateChecker : public QObject, public SettingsStorage
{
    Q_OBJECT
public:
    /*!
     * \brief Parsed semantic version.
     *
     * \c releaseTag is empty for stable releases and carries the suffix
     * (e.g., "alpha", "beta", "rc1") otherwise. The compiled-in local
     * version always has a non-empty releaseTag during pre-1.0 development.
     */
    struct Version {
        int major{0};
        int minor{0};
        int patch{0};
        QString releaseTag;

        /*!
         * \brief Compare on (major, minor, patch) only; releaseTag is
         * ignored. Returns true when \a a is strictly older than \a b.
         */
        static bool isOlder(const Version &a, const Version &b);

        /*!
         * \brief Format as "<major>.<minor>.<patch>" or
         * "<major>.<minor>.<patch>-<tag>" when releaseTag is non-empty.
         */
        QString toString() const;
    };

    explicit UpdateChecker(QObject *parent = nullptr);
    ~UpdateChecker() override;

    /*!
     * \brief Compiled-in application version from BC_*_VERSION macros.
     */
    static Version localVersion();

    /*!
     * \brief Parse a GitHub tag string (e.g., "v2.0.0", "v2.0.0-alpha").
     *
     * Accepts an optional leading 'v'. Returns a zero-initialized Version
     * on parse failure.
     */
    static Version parseTag(const QString &tag);

    /*!
     * \brief Version string the user opted out of notifications for.
     *
     * Empty when nothing is skipped. Read from QSettings; persistent.
     * Callers consult this to decide whether to suppress notifications on
     * startup-throttled checks. Manual checks generally ignore it.
     */
    QString skippedVersion() const;

    /*!
     * \brief Record a version string as "do not notify again".
     *
     * Persists to QSettings. Pass an empty string to clear the skip.
     */
    void setSkippedVersion(const QString &version);

    /*!
     * \brief Timestamp of the last successful check (UTC).
     *
     * Invalid QDateTime when no check has completed. Used by callers that
     * implement once-per-day throttling on startup.
     */
    QDateTime lastCheckedAt() const;

public slots:
    /*!
     * \brief Issue a single HTTP GET against the GitHub releases endpoint.
     *
     * Emits exactly one of updateAvailable(), upToDate(), or checkFailed()
     * once the request resolves. Subsequent calls before a pending request
     * resolves are no-ops.
     */
    void checkNow();

signals:
    /*!
     * \brief A newer release is available on GitHub.
     * \param remoteVersion  Parsed version from the release's tag_name.
     * \param releaseUrl     html_url of the release on GitHub.
     * \param releaseName    Human-readable release title.
     */
    void updateAvailable(const UpdateChecker::Version &remoteVersion,
                         const QUrl &releaseUrl,
                         const QString &releaseName);

    /*!
     * \brief Remote version is not newer than the local version, or the
     * repository has no stable release yet.
     */
    void upToDate();

    /*!
     * \brief The check could not complete (network, TLS, parse, or
     * non-200/404 HTTP response).
     * \param reason Human-readable description suitable for display.
     */
    void checkFailed(const QString &reason);

private slots:
    void onReplyFinished();

private:
    void recordCheckCompleted();

    QNetworkAccessManager *p_nam{nullptr};
    QNetworkReply *p_reply{nullptr};
};

/*!
 * \brief QSettings keys owned by UpdateChecker for its internal state.
 *
 * The user-visible toggle that controls whether checks happen at all
 * lives in BC::Key::AppConfig::updateCheckEnabled — these keys are the
 * operational bookkeeping the checker manages itself.
 */
namespace BC::Key::Update {
    inline constexpr QLatin1StringView updateChecker{"updateChecker"};  /*!< Settings group */
    inline constexpr QLatin1StringView lastRun{"lastRun"};              /*!< ISO-8601 timestamp of the last completed check */
    inline constexpr QLatin1StringView skipVersion{"skipVersion"};      /*!< Version string the user opted out of notifications for */
}

#endif // UPDATECHECKER_H
