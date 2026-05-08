#ifndef CRASHREPORTDIALOG_H
#define CRASHREPORTDIALOG_H

#include <QDialog>
#include <QStringList>

#include <data/storage/settingsstorage.h>

/*!
 * \brief Settings keys owned by the crash-report dialog.
 */
namespace BC::Key::CrashDialog {
    inline constexpr QLatin1StringView crashDialog{"CrashDialog"}; /*!< Group key */
    inline constexpr QLatin1StringView lastSeen{"lastSeen"};       /*!< Most recent
        crash-artifact UTC timestamp (yyyyMMdd-HHmmss) acknowledged by the user. */
}

/*!
 * \brief Modeless notification that one or more crash artifacts from
 *        prior process runs are present.
 *
 * Constructed by \c main shortly after the main window is shown when
 * \c CrashHandler::collectPriorArtifacts() returns artifacts whose
 * filename timestamp is strictly newer than the value previously
 * persisted at \c BC::Key::CrashDialog::lastSeen.
 *
 * The dialog inherits \c SettingsStorage and owns the \c lastSeen
 * key. On dismissal it writes the topmost (most recent) artifact's
 * timestamp into that key, so the next launch only re-prompts when a
 * still-newer crash has occurred. Reports are not deleted; users are
 * expected to email them to the developer or remove them by hand.
 *
 * Actions offered:
 *   - Open Folder: opens the crash directory in the platform file
 *     manager via \c QDesktopServices.
 *   - View Most Recent: opens the newest crash artifact in the
 *     platform-default text editor.
 *   - Dismiss: closes the dialog without modifying any files.
 */
class CrashReportDialog : public QDialog, public SettingsStorage
{
    Q_OBJECT
public:
    explicit CrashReportDialog(const QStringList &artifacts,
                               const QString &crashesDir,
                               QWidget *parent = nullptr);

private:
    QString d_crashesDir;
    QStringList d_artifacts;
};

#endif
