#ifndef EXPERIMENTCHOOSERDIALOG_H
#define EXPERIMENTCHOOSERDIALOG_H

#include <vector>

#include <QDialog>
#include <QString>

class QSpinBox;
class QCheckBox;
class QLineEdit;
class QToolButton;
class QComboBox;

/*!
 * \brief Modal "pick a saved experiment" dialog shared by both apps.
 *
 * Used by the acquisition app's View Experiment workflow and the
 * standalone viewer's Open Experiment workflow. The dialog is
 * deliberately storage-free: the host reads the spinbox bound, the
 * browse start directory, and the recent-experiment list from its own
 * SettingsStorage scope, constructs the dialog with them, and — on
 * acceptance — reads back the chosen number/path and performs the
 * actual open and recent-history write itself. The dialog never touches
 * SettingsStorage, so the two apps' independent histories stay in their
 * own scopes (see \ref BC::RecentExperiments).
 */
class ExperimentChooserDialog : public QDialog
{
    Q_OBJECT
public:
    //! One entry for the optional recent-experiment combo.
    struct RecentEntry { int num; QString path; };

    /*!
     * \param upperBound Highest selectable experiment number. A value
     *        below 1 means no experiments are available: the number
     *        control is disabled and the dialog opens in custom-path
     *        mode.
     * \param browseStartDir Initial directory for the folder picker.
     * \param recent Entries for the recent combo; an empty list
     *        suppresses the combo entirely.
     */
    explicit ExperimentChooserDialog(int upperBound,
                                     const QString &browseStartDir,
                                     const std::vector<RecentEntry> &recent,
                                     QWidget *parent = nullptr);

    //! Chosen experiment number, or 0 when a custom path was selected.
    int experimentNumber() const { return d_resultNum; }
    //! Chosen custom path, or empty when an experiment number was selected.
    QString experimentPath() const { return d_resultPath; }

signals:
    //! Emitted when the user picks a new directory in the folder picker.
    void browseDirChanged(const QString &dir);

protected:
    void accept() override;

private:
    QSpinBox *p_numBox;
    QCheckBox *p_pathBox;
    QLineEdit *p_pathEdit;
    QToolButton *p_browseButton;
    QComboBox *p_recentBox = nullptr;

    QString d_browseDir;
    std::vector<RecentEntry> d_recent;

    int d_resultNum = 0;
    QString d_resultPath;
};

#endif // EXPERIMENTCHOOSERDIALOG_H
