#include "updateavailabledialog.h"

#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

#include <data/updatechecker.h>

using namespace Qt::Literals::StringLiterals;

UpdateAvailableDialog::UpdateAvailableDialog(const QString &remoteVersion,
                                             const QString &localVersion,
                                             const QUrl &releaseUrl,
                                             const QString &releaseName,
                                             QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("Update Available"_L1);
    setModal(true);

    auto *layout = new QVBoxLayout(this);
    layout->setSpacing(10);

    auto *header = new QLabel(
        u"<b>A new version of Blackchirp is available.</b>"_s, this);
    layout->addWidget(header);

    auto *body = new QLabel(
        u"Installed: %1<br>Latest release: <b>%2</b>"_s
            .arg(localVersion, remoteVersion),
        this);
    body->setTextFormat(Qt::RichText);
    layout->addWidget(body);

    if(!releaseName.isEmpty() && releaseName != remoteVersion)
    {
        auto *name = new QLabel(u"Release notes: "_s + releaseName, this);
        name->setWordWrap(true);
        layout->addWidget(name);
    }

    auto *hint = new QLabel(
        u"Choose <b>Download</b> to open the release page in your browser. "
         "Blackchirp does not install updates automatically."_s, this);
    hint->setWordWrap(true);
    layout->addWidget(hint);

    auto *buttons = new QDialogButtonBox(this);
    auto *downloadBtn = buttons->addButton("Download"_L1,
                                            QDialogButtonBox::AcceptRole);
    auto *skipBtn     = buttons->addButton("Skip This Version"_L1,
                                            QDialogButtonBox::DestructiveRole);
    auto *laterBtn    = buttons->addButton("Remind Me Later"_L1,
                                            QDialogButtonBox::RejectRole);
    downloadBtn->setDefault(true);

    connect(downloadBtn, &QPushButton::clicked, this, [this, releaseUrl]() {
        QDesktopServices::openUrl(releaseUrl);
        d_outcome = Downloaded;
        accept();
    });
    connect(skipBtn, &QPushButton::clicked, this, [this]() {
        d_outcome = SkippedVersion;
        accept();
    });
    connect(laterBtn, &QPushButton::clicked, this, [this]() {
        d_outcome = Dismissed;
        reject();
    });

    layout->addWidget(buttons);
}

void UpdateAvailableDialog::triggerManualCheck(UpdateChecker *checker, QWidget *parent)
{
    if(!checker)
        return;

    // Per-call receiver. UpdateChecker emits exactly one of the three
    // signals per checkNow(), so whichever lambda fires schedules `r` for
    // deletion; the other two connections die with the receiver and do
    // not accumulate across calls.
    auto *r = new QObject(parent);

    QObject::connect(checker, &UpdateChecker::updateAvailable, r,
                     [checker, parent, r](const UpdateChecker::Version &remote,
                                          const QUrl &url, const QString &name) {
        UpdateAvailableDialog dlg(remote.toString(),
                                  UpdateChecker::localVersion().toString(),
                                  url, name, parent);
        dlg.exec();
        if(dlg.outcome() == UpdateAvailableDialog::SkippedVersion)
            checker->setSkippedVersion(remote.toString());
        r->deleteLater();
    });

    QObject::connect(checker, &UpdateChecker::upToDate, r, [parent, r]() {
        QMessageBox::information(parent, "Check for Updates"_L1,
            u"You are running the latest version of Blackchirp (%1)."_s
                .arg(UpdateChecker::localVersion().toString()));
        r->deleteLater();
    });

    QObject::connect(checker, &UpdateChecker::checkFailed, r,
                     [parent, r](const QString &reason) {
        QMessageBox::warning(parent, "Check for Updates"_L1,
            u"Could not check for updates:\n\n%1"_s.arg(reason));
        r->deleteLater();
    });

    checker->checkNow();
}
