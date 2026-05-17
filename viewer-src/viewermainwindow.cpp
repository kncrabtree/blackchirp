#include "viewermainwindow.h"

#include <QApplication>
#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QDir>
#include <QFontMetrics>
#include <QFormLayout>
#include <QListWidgetItem>
#include <QMenu>
#include <QMouseEvent>
#include <QPushButton>
#include <QToolBar>
#include <QUrl>

#include <gui/dialog/aboutdialog.h>
#include <gui/dialog/updateavailabledialog.h>
#include <gui/dialog/experimentchooserdialog.h>
#include <gui/util/recentexperiments.h>
#include <data/storage/applicationconfigmanager.h>
#include <data/storage/blackchirpcsv.h>
#include <data/updatechecker.h>
#include <QTimer>

#define _STR(x) #x
#define STRINGIFY(x) _STR(x)

ViewerMainWindow::ViewerMainWindow(QWidget *parent)
    : QMainWindow(parent), SettingsStorage(BC::Key::Viewer::viewer)
{
    setWindowTitle(QString("Blackchirp Viewer"));
    setWindowIcon(ThemeColors::createThemedIcon(":/icons/bc_logo_trans.svg", ThemeColors::IconPrimary, this));

    loadActiveDataPath();

    if (d_dataPath.isEmpty()) {
        QMessageBox::warning(this, "Data Path Not Found",
            "Could not find a Blackchirp data path. Run Blackchirp first to "
            "configure one, set an override under Settings → Set Data Path…, "
            "or specify a path when opening individual experiments.");
    }

    setupUI();
    setupMenuBar();
    updateButtonStates();

    resize(400, 300);

    // Throttled startup check. The updateCheckEnabled toggle is owned by
    // the acquisition app's ApplicationConfigManager (read here from the
    // same Blackchirp2.conf), so the viewer and main app share a single
    // user preference. UpdateChecker's lastRun/skipVersion are shared the
    // same way, so a recent check from either app suppresses the next.
    if(ApplicationConfigManager::instance().isUpdateCheckEnabled())
    {
        QTimer::singleShot(2000, this, [this]() {
            UpdateAvailableDialog::triggerStartupCheck(p_updateChecker, this);
        });
    }
}

ViewerMainWindow::~ViewerMainWindow()
{
    // Disconnect widgetClosing first so unique_ptr destruction does not
    // re-enter onExperimentWidgetClosing and mutate d_openExperiments mid-clear.
    for (auto& [displayText, widget] : d_openExperiments) {
        if (widget) {
            disconnect(widget.get(), &ExperimentViewWidget::widgetClosing,
                       this, &ViewerMainWindow::onExperimentWidgetClosing);
        }
    }
    d_openExperiments.clear();
}

void ViewerMainWindow::setupUI()
{
    p_centralWidget = new QWidget(this);
    setCentralWidget(p_centralWidget);
    
    QVBoxLayout *mainLayout = new QVBoxLayout(p_centralWidget);
    
    // Create shared actions
    p_openAction = new QAction(ThemeColors::createThemedIcon(":/icons/document-plus.svg", ThemeColors::IconPrimary, this), 
                               "Open Experiment", this);
    p_openAction->setShortcut(QKeySequence::Open);
    p_openAction->setStatusTip("Open an experiment for viewing");
    connect(p_openAction, &QAction::triggered, this, &ViewerMainWindow::openExperiment);
    
    p_closeAction = new QAction(ThemeColors::createThemedIcon(":/icons/x-mark.svg", ThemeColors::IconSecondary, this), 
                                "Close Selected", this);
    p_closeAction->setStatusTip("Close the selected experiment");
    connect(p_closeAction, &QAction::triggered, this, &ViewerMainWindow::closeSelectedExperiment);
    
    // Create toolbar
    QToolBar *toolbar = addToolBar("Main");
    toolbar->addAction(p_openAction);
    toolbar->addAction(p_closeAction);
    
    // Create experiment list
    p_experimentList = new QListWidget(this);
    p_experimentList->setSelectionMode(QAbstractItemView::SingleSelection);
    connect(p_experimentList, &QListWidget::itemDoubleClicked, 
            this, &ViewerMainWindow::onListItemDoubleClicked);
    connect(p_experimentList, &QListWidget::itemSelectionChanged,
            this, &ViewerMainWindow::updateButtonStates);
    
    mainLayout->addWidget(p_experimentList);

    // Active data-path label. Shows the directory the viewer reads
    // experiments-by-number from, with a click-to-change affordance
    // (cursor + underline on hover, MouseButtonRelease handled in
    // eventFilter). Tooltip carries the full path plus the rule for
    // when this directory does and doesn't apply.
    p_dataPathLabel = new QLabel(this);
    p_dataPathLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    p_dataPathLabel->setCursor(Qt::PointingHandCursor);
    p_dataPathLabel->installEventFilter(this);
    mainLayout->addWidget(p_dataPathLabel);
    updateDataPathLabel();

    // Status label — transient activity messages (Ready / Opened X /
    // Closed Y). The data-path display moved to p_dataPathLabel above.
    p_statusLabel = new QLabel(QString("Ready"), this);
    p_statusLabel->setStyleSheet(QString("QLabel { color: %1; font-size: 10px; }")
                                .arg(ThemeColors::getThemeAwareColor(ThemeColors::SubtleText, this).name()));
    mainLayout->addWidget(p_statusLabel);
}

void ViewerMainWindow::setupMenuBar()
{
    QMenuBar *menuBar = this->menuBar();
    
    // File menu - minimal, just essential items
    QMenu *fileMenu = menuBar->addMenu("&File");
    
    QAction *exitAction = fileMenu->addAction("E&xit");
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);
    
    // Experiment menu for experiment-related actions
    QMenu *expMenu = menuBar->addMenu("&Experiment");
    expMenu->addAction(p_openAction);

    p_recentMenu = expMenu->addMenu("Open &Recent");
    updateRecentMenu();

    expMenu->addSeparator();
    expMenu->addAction(p_closeAction);

    // Settings menu — viewer-specific knobs that don't belong in
    // Blackchirp's settings file.
    QMenu *settingsMenu = menuBar->addMenu("&Settings");

    auto *setDataPathAction = settingsMenu->addAction("Set &Data Path…");
    setDataPathAction->setStatusTip(
        "Override the directory the viewer reads experiments from");
    connect(setDataPathAction, &QAction::triggered,
            this, &ViewerMainWindow::chooseDataPath);

    auto *resetDataPathAction = settingsMenu->addAction("&Reset to Blackchirp Default");
    resetDataPathAction->setStatusTip(
        "Drop the override and follow Blackchirp's configured savePath");
    connect(resetDataPathAction, &QAction::triggered,
            this, &ViewerMainWindow::resetDataPath);

    // Help menu — kept as a member so the update-check badge can mutate
    // its title from the persistent UpdateChecker connections below.
    p_helpMenu = menuBar->addMenu("&Help");
    QMenu *helpMenu = p_helpMenu;

    auto addUrl = [helpMenu, this](const QString &text, const char *url) {
        helpMenu->addAction(text, this, [url]() {
            QDesktopServices::openUrl(QUrl(QLatin1StringView(url)));
        });
    };
    addUrl(QString("&Documentation"),   "https://blackchirp.readthedocs.io/");
    addUrl(QString("&GitHub Repository"), "https://github.com/kncrabtree/blackchirp");
    addUrl(QString("Di&scord Server"),  "https://discord.gg/88CkbAKUZY");
    helpMenu->addSeparator();

    p_updateChecker = new UpdateChecker(this);
    p_checkForUpdatesAction = helpMenu->addAction(
        QString("Check for &Updates..."), this,
        &ViewerMainWindow::onCheckForUpdatesTriggered);
    // Persistent badge: themed sparkles icon on the menu action plus a
    // ★ on the Help menu title so the user notices the indicator from
    // the menu bar without opening the menu. Cleared on upToDate.
    connect(p_updateChecker, &UpdateChecker::updateAvailable, this,
            [this](const UpdateChecker::Version &remote, const QUrl &,
                   const QString &) {
        p_checkForUpdatesAction->setIcon(ThemeColors::createThemedIcon(
            ":/icons/sparkles.svg", ThemeColors::StatusInfo, this));
        p_checkForUpdatesAction->setText(
            QString("Check for &Updates... %1 available")
                .arg(remote.toString()));
        p_helpMenu->setTitle(QString("&Help ★"));
    });
    connect(p_updateChecker, &UpdateChecker::upToDate, this, [this]() {
        p_checkForUpdatesAction->setIcon(QIcon());
        p_checkForUpdatesAction->setText(QString("Check for &Updates..."));
        p_helpMenu->setTitle(QString("&Help"));
    });
    helpMenu->addSeparator();

    auto *aboutAction = helpMenu->addAction("&About Blackchirp Viewer");
    connect(aboutAction, &QAction::triggered, this, [this]() {
        AboutDialog::AppInfo info;
        info.name = QString("Blackchirp Viewer");
        info.version = QString("%1.%2.%3-%4")
            .arg(BCV_MAJOR_VERSION).arg(BCV_MINOR_VERSION)
            .arg(BCV_PATCH_VERSION).arg(STRINGIFY(BCV_RELEASE_VERSION));
        info.build = QLatin1StringView(BCV_BUILD_VERSION);
        info.description = QString("Data visualization application for Blackchirp experiments.");
        AboutDialog dlg(info, this);
        dlg.exec();
    });

    helpMenu->addAction(QString("About &Qt"), qApp, &QApplication::aboutQt);
}

void ViewerMainWindow::openExperiment()
{
    // The last experiment number Blackchirp wrote. The viewer is itself
    // a SettingsStorage scoped to [BlackchirpViewer]; a transient
    // default-ctor SettingsStorage reaches the acquisition app's
    // [Blackchirp] group in the same shared settings file.
    const int lastExptNum = SettingsStorage().get(BC::Key::exptNum, 0);

    std::vector<ExperimentChooserDialog::RecentEntry> recent;
    for(const auto &e : getArray(BC::Key::Viewer::recentExperiments))
    {
        int n;
        QString p;
        BC::RecentExperiments::decode(e, n, p);
        recent.push_back({n, p});
    }

    const QString startDir = get(BC::Key::Viewer::lastBrowseDir,
                                 QDir::homePath());

    ExperimentChooserDialog d(lastExptNum, startDir, recent, this);
    connect(&d, &ExperimentChooserDialog::browseDirChanged, this,
            [this](const QString &dir){
        set(BC::Key::Viewer::lastBrowseDir, dir, true);
    });

    if (d.exec() == QDialog::Accepted)
        openExperimentByNumPath(d.experimentNumber(), d.experimentPath());
}

void ViewerMainWindow::closeSelectedExperiment()
{
    QListWidgetItem *currentItem = p_experimentList->currentItem();
    if (!currentItem) {
        return;
    }

    QString displayText = currentItem->data(Qt::UserRole).toString();
    
    auto it = d_openExperiments.find(displayText);
    if (it != d_openExperiments.end()) {
        // Close the widget
        it->second->close();
        // The widget will emit widgetClosing signal which will handle cleanup
    }
}

void ViewerMainWindow::onExperimentWidgetClosing()
{
    ExperimentViewWidget* widget = qobject_cast<ExperimentViewWidget*>(sender());
    if (!widget) {
        return;
    }

    // Find and remove from our tracking
    QString displayTextToRemove;
    for (auto it = d_openExperiments.begin(); it != d_openExperiments.end(); ++it) {
        if (it->second.get() == widget) {
            displayTextToRemove = it->first;
            d_openExperiments.erase(it);
            break;
        }
    }

    if (!displayTextToRemove.isEmpty()) {
        removeExperimentFromList(displayTextToRemove);
        p_statusLabel->setText(QString("Closed: %1").arg(displayTextToRemove));
        updateButtonStates();
    }
}

void ViewerMainWindow::onListItemDoubleClicked(QListWidgetItem *item)
{
    if (!item) {
        return;
    }

    QString displayText = item->data(Qt::UserRole).toString();
    auto it = d_openExperiments.find(displayText);
    if (it != d_openExperiments.end()) {
        ExperimentViewWidget* widget = it->second.get();
        widget->show();
        widget->raise();
        widget->activateWindow();
    }
}

void ViewerMainWindow::updateButtonStates()
{
    bool hasSelection = p_experimentList->currentItem() != nullptr;
    p_closeAction->setEnabled(hasSelection);
    
    if (d_openExperiments.empty()) {
        p_statusLabel->setText(QString("Ready"));
    }
}

void ViewerMainWindow::updateDataPathLabel()
{
    static const QString explanation =
        QString("Experiments loaded by number are read from this directory. "
                "Other experiments can always be loaded by browsing to their "
                "folder explicitly when they live outside a Blackchirp-formatted "
                "experiments tree.");

    if (d_dataPath.isEmpty()) {
        p_dataPathLabel->setText(QString("Data Path: (click to choose)"));
        p_dataPathLabel->setToolTip(
            QString("No data path configured.\n\n%1").arg(explanation));
        return;
    }

    p_dataPathLabel->setToolTip(
        QString("%1\n\n%2").arg(d_dataPath, explanation));

    const QFontMetrics fm(p_dataPathLabel->font());
    const int w = p_dataPathLabel->width();
    const QString prefix(QString("Data Path: "));
    const int avail = (w > 0 ? w : 200) - fm.horizontalAdvance(prefix);
    p_dataPathLabel->setText(
        prefix + fm.elidedText(d_dataPath, Qt::ElideMiddle, std::max(avail, 32)));
}

void ViewerMainWindow::loadActiveDataPath()
{
    // Active data path: the viewer's own override wins; if absent or
    // empty, fall back to Blackchirp's savePath. Both reads target
    // Blackchirp2.conf — the override under [BlackchirpViewer], the
    // fallback under [Blackchirp] — so the viewer never writes into the
    // acquisition app's group.
    d_dataPath = get(BC::Key::Viewer::dataPath, QString());
    if (!d_dataPath.isEmpty())
        return;

    SettingsStorage blackchirpStore;
    d_dataPath = blackchirpStore.get(BC::Key::savePath, QString());
}

void ViewerMainWindow::chooseDataPath()
{
    QString startDir = d_dataPath.isEmpty() ? QDir::homePath() : d_dataPath;
    QString chosen = QFileDialog::getExistingDirectory(this,
        QString("Select Blackchirp data directory"), startDir);
    if (chosen.isEmpty())
        return;

    d_dataPath = chosen;
    set(BC::Key::Viewer::dataPath, chosen, true);
    updateDataPathLabel();
}

void ViewerMainWindow::resetDataPath()
{
    clearValue(BC::Key::Viewer::dataPath);
    loadActiveDataPath();
    updateDataPathLabel();

    if (d_dataPath.isEmpty()) {
        QMessageBox::information(this, QString("Data Path"),
            QString("Blackchirp has no savePath set yet, so the viewer has "
                    "no active data directory. Run Blackchirp first or use "
                    "Set Data Path… to point the viewer at an experiment "
                    "tree manually."));
    }
}

void ViewerMainWindow::onCheckForUpdatesTriggered()
{
    UpdateAvailableDialog::triggerManualCheck(p_updateChecker, this);
}

void ViewerMainWindow::removeExperimentFromList(const QString& displayText)
{
    for (int i = 0; i < p_experimentList->count(); ++i) {
        QListWidgetItem* item = p_experimentList->item(i);
        if (item && item->data(Qt::UserRole).toString() == displayText) {
            delete p_experimentList->takeItem(i);
            break;
        }
    }
}

void ViewerMainWindow::openExperimentByNumPath(int num, const QString &path)
{
    QString displayText = BC::RecentExperiments::displayText(num, path);

    // Check if experiment is already open
    auto it = d_openExperiments.find(displayText);
    if (it != d_openExperiments.end()) {
        ExperimentViewWidget* existingWidget = it->second.get();
        existingWidget->show();
        existingWidget->raise();
        existingWidget->notifyAlreadyOpen();
        return;
    }

    // Create new experiment view widget
    auto evw = std::make_unique<ExperimentViewWidget>(num, path, true);
    ExperimentViewWidget* evwPtr = evw.get();

    connect(evwPtr, &ExperimentViewWidget::widgetClosing, this, &ViewerMainWindow::onExperimentWidgetClosing);

    d_openExperiments[displayText] = std::move(evw);
    evwPtr->show();

    QListWidgetItem *item = new QListWidgetItem(displayText);
    item->setData(Qt::UserRole, displayText);
    p_experimentList->addItem(item);

    p_statusLabel->setText(QString("Opened: %1").arg(displayText));
    updateButtonStates();

    // In path mode the caller passes num == 0; the real number lives in the
    // header file that ExperimentViewWidget just loaded.
    int recentNum = num > 0 ? num : evwPtr->experimentNumber();
    addToRecentExperiments(recentNum, path);
}

void ViewerMainWindow::addToRecentExperiments(int num, const QString &path)
{
    auto recent = BC::RecentExperiments::prepend(
        getArray(BC::Key::Viewer::recentExperiments), num, path,
        MaxRecentExperiments);
    setArray(BC::Key::Viewer::recentExperiments, recent, true);
    updateRecentMenu();
}

void ViewerMainWindow::updateRecentMenu()
{
    p_recentMenu->clear();

    auto recent = getArray(BC::Key::Viewer::recentExperiments);
    if (recent.empty()) {
        p_recentMenu->addAction("(No recent experiments)")->setEnabled(false);
        return;
    }

    for (const auto &entry : recent) {
        int num;
        QString path;
        BC::RecentExperiments::decode(entry, num, path);

        auto action = p_recentMenu->addAction(
            BC::RecentExperiments::displayText(num, path));
        connect(action, &QAction::triggered, this, [this, num, path]() {
            openExperimentByNumPath(num, path);
        });
    }

    p_recentMenu->addSeparator();
    auto clearAction = p_recentMenu->addAction("Clear Recent");
    connect(clearAction, &QAction::triggered, this, [this]() {
        setArray(BC::Key::Viewer::recentExperiments, {}, true);
        updateRecentMenu();
    });
}

bool ViewerMainWindow::eventFilter(QObject *watched, QEvent *event)
{
    if (watched != p_dataPathLabel)
        return QMainWindow::eventFilter(watched, event);

    switch (event->type()) {
    case QEvent::Resize:
        updateDataPathLabel();
        break;
    case QEvent::Enter: {
        auto f = p_dataPathLabel->font();
        f.setUnderline(true);
        p_dataPathLabel->setFont(f);
        break;
    }
    case QEvent::Leave: {
        auto f = p_dataPathLabel->font();
        f.setUnderline(false);
        p_dataPathLabel->setFont(f);
        break;
    }
    case QEvent::MouseButtonRelease: {
        auto *me = static_cast<QMouseEvent *>(event);
        if (me->button() == Qt::LeftButton
            && p_dataPathLabel->rect().contains(me->pos()))
            chooseDataPath();
        break;
    }
    default:
        break;
    }

    return QMainWindow::eventFilter(watched, event);
}

void ViewerMainWindow::closeEvent(QCloseEvent *event)
{
    // Disconnect widgetClosing first so closing each widget does not re-enter
    // onExperimentWidgetClosing and erase from the map being iterated.
    for (auto& [displayText, widget] : d_openExperiments) {
        if (widget) {
            disconnect(widget.get(), &ExperimentViewWidget::widgetClosing,
                       this, &ViewerMainWindow::onExperimentWidgetClosing);
            widget->close();
        }
    }
    d_openExperiments.clear();

    QMainWindow::closeEvent(event);
}
