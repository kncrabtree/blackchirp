#include "crashreportdialog.h"

#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QFileInfo>
#include <QFont>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPushButton>
#include <QScrollBar>
#include <QUrl>
#include <QVBoxLayout>

#include <data/crashhandler.h>

using namespace Qt::Literals::StringLiterals;

CrashReportDialog::CrashReportDialog(const QStringList &artifacts,
                                     const QString &crashesDir,
                                     QWidget *parent)
    : QDialog(parent),
      SettingsStorage(BC::Key::CrashDialog::crashDialog),
      d_crashesDir(crashesDir),
      d_artifacts(artifacts)
{
    setWindowTitle("Crash Reports Detected"_L1);
    setAttribute(Qt::WA_DeleteOnClose);

    auto layout = new QVBoxLayout(this);

    auto header = new QLabel(
        "Blackchirp detected crash reports from a prior session. The reports "
        "below were written by an earlier run that did not exit cleanly. You "
        "can open the directory to inspect or email the report files, view "
        "the most recent report in your default text editor, or dismiss this "
        "notice. Reports are not deleted automatically."_L1, this);
    header->setWordWrap(true);
    layout->addWidget(header);

    auto list = new QListWidget(this);
    QFont boldFont = list->font();
    boldFont.setBold(true);
    for(int i = 0; i < d_artifacts.size(); ++i)
    {
        QFileInfo fi(d_artifacts.at(i));
        auto *item = new QListWidgetItem(fi.fileName(), list);
        if(i == 0)
            item->setFont(boldFont);
    }
    layout->addWidget(list, 1);

    auto reportLink = new QLabel(
        "To report a crash, open an issue at "
        "<a href=\"https://github.com/kncrabtree/blackchirp/issues\">"
        "github.com/kncrabtree/blackchirp/issues</a> and attach the most "
        "recent report file along with a short description of what you were "
        "doing when the crash occurred."_L1, this);
    reportLink->setOpenExternalLinks(true);
    reportLink->setTextInteractionFlags(Qt::TextBrowserInteraction);
    reportLink->setWordWrap(true);
    layout->addWidget(reportLink);

    // Anchor the dialog width to the widest crash filename so that the
    // wrapped labels above and below line up with the list rather than
    // collapsing to a narrow column. The list's frame and scrollbar
    // margin are added so an entry never overlaps the scrollbar.
    int contentsWidth = list->sizeHintForColumn(0)
                      + 2 * list->frameWidth()
                      + list->verticalScrollBar()->sizeHint().width();
    setMinimumWidth(qMax(contentsWidth + layout->contentsMargins().left()
                                       + layout->contentsMargins().right(),
                         480));

    auto buttons = new QDialogButtonBox(this);
    auto openFolder = buttons->addButton("Open &Folder"_L1,
                                         QDialogButtonBox::ActionRole);
    auto openMostRecent = buttons->addButton("View Most &Recent"_L1,
                                             QDialogButtonBox::ActionRole);
    auto dismiss = buttons->addButton("&Dismiss"_L1,
                                      QDialogButtonBox::AcceptRole);
    layout->addWidget(buttons);

    connect(openFolder, &QPushButton::clicked, this, [this]() {
        if(!d_crashesDir.isEmpty())
            QDesktopServices::openUrl(QUrl::fromLocalFile(d_crashesDir));
    });
    connect(openMostRecent, &QPushButton::clicked, this, [this]() {
        if(!d_artifacts.isEmpty())
            QDesktopServices::openUrl(QUrl::fromLocalFile(d_artifacts.first()));
    });
    connect(dismiss, &QPushButton::clicked, this, &QDialog::accept);

    connect(this, &QDialog::finished, this, [this](int) {
        if(d_artifacts.isEmpty())
            return;
        auto ts = CrashHandler::artifactTimestamp(d_artifacts.first());
        if(!ts.isEmpty())
            set(BC::Key::CrashDialog::lastSeen, ts, true);
    });
}
