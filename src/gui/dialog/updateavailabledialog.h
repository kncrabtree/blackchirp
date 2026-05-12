#ifndef UPDATEAVAILABLEDIALOG_H
#define UPDATEAVAILABLEDIALOG_H

#include <QDialog>
#include <QString>
#include <QUrl>

class UpdateChecker;

/*!
 * \brief Modal dialog notifying the user that a newer release is available.
 *
 * Presents three actions:
 * - **Download** — opens the release page in the user's browser via
 *   QDesktopServices, then accepts the dialog.
 * - **Remind Me Later** — closes the dialog. The next check (manual or
 *   the next daily-throttled startup) will notify again.
 * - **Skip This Version** — closes the dialog and asks the caller to
 *   record the remote version in updateCheckSkipVersion so the user is
 *   not notified again about this specific tag.
 *
 * The caller distinguishes the three outcomes through outcome().
 */
class UpdateAvailableDialog : public QDialog
{
    Q_OBJECT
public:
    enum Outcome {
        Dismissed,       ///< Dialog closed without action ("Remind me later")
        Downloaded,      ///< User opened the release page
        SkippedVersion,  ///< User opted out of further notifications for this version
    };
    Q_ENUM(Outcome)

    explicit UpdateAvailableDialog(const QString &remoteVersion,
                                   const QString &localVersion,
                                   const QUrl &releaseUrl,
                                   const QString &releaseName,
                                   QWidget *parent = nullptr);

    Outcome outcome() const { return d_outcome; }

    /*!
     * \brief Trigger an update check and wire the three outcomes to the
     * standard "user clicked Check for Updates" UX.
     *
     * Connects \a checker's three signals to a per-call receiver scoped
     * to \a parent and immediately calls \c checkNow():
     * - \c updateAvailable shows this dialog modally; SkippedVersion is
     *   persisted via \c UpdateChecker::setSkippedVersion.
     * - \c upToDate shows QMessageBox::information.
     * - \c checkFailed shows QMessageBox::warning.
     *
     * The receiver is deleted after the first emission, so repeated calls
     * do not accumulate connections.
     */
    static void triggerManualCheck(UpdateChecker *checker, QWidget *parent);

    /*!
     * \brief Trigger an unobtrusive startup update check.
     *
     * Performs the once-per-day throttle and skip-version filtering that
     * the manual path omits:
     * - Returns immediately if \c lastCheckedAt() is less than 24h old.
     * - On \c updateAvailable, suppresses the dialog when the remote tag
     *   matches \c skippedVersion(). Otherwise shows the same modal as the
     *   manual path.
     * - \c upToDate and \c checkFailed are silent (only the receiver is
     *   cleaned up). \c bcDebug entries inside UpdateChecker remain.
     *
     * Like \c triggerManualCheck, uses a per-call receiver scoped to
     * \a parent so connections do not accumulate.
     */
    static void triggerStartupCheck(UpdateChecker *checker, QWidget *parent);

private:
    Outcome d_outcome{Dismissed};
};

#endif // UPDATEAVAILABLEDIALOG_H
