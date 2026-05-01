#include "viewermainwindow.h"
#include <QApplication>
#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QDir>
#include <QFormLayout>
#include <QListWidgetItem>
#include <QMenu>
#include <QPushButton>
#include <QToolBar>
#include <QUrl>

#include <gui/dialog/aboutdialog.h>

#define _STR(x) #x
#define STRINGIFY(x) _STR(x)

ViewerMainWindow::ViewerMainWindow(QWidget *parent)
    : QMainWindow(parent), SettingsStorage()
{
    setWindowTitle(QString("Blackchirp Viewer"));
    setWindowIcon(ThemeColors::createThemedIcon(":/icons/bc_logo_trans.svg", ThemeColors::IconPrimary, this));
    
    // Read Blackchirp's data storage path from its settings
    SettingsStorage blackchirpSettings("CrabtreeLab", "Blackchirp");
    d_dataPath = blackchirpSettings.get(BC::Key::savePath, QString(""));

    set(BC::Key::savePath,d_dataPath,true);
    
    if (d_dataPath.isEmpty()) {
        QMessageBox::warning(this, "Data Path Not Found", 
            "Could not find Blackchirp's data storage path in settings.\n"
            "You may need to run Blackchirp first to set up the data path,\n"
            "or manually specify paths when opening experiments.");
    }
    
    setupUI();
    setupMenuBar();
    updateButtonStates();
    
    resize(400, 300);
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
    
    
    // Status label
    QString statusText;
    if (d_dataPath.isEmpty()) {
        statusText = "Ready (No data path configured)";
    } else {
        statusText = QString("Ready (Data path: %1)").arg(d_dataPath);
    }
    p_statusLabel = new QLabel(statusText, this);
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
    
    // Help menu
    QMenu *helpMenu = menuBar->addMenu("&Help");

    auto addUrl = [helpMenu, this](const QString &text, const char *url) {
        helpMenu->addAction(text, this, [url]() {
            QDesktopServices::openUrl(QUrl(QLatin1StringView(url)));
        });
    };
    addUrl(QString("&Documentation"),   "https://blackchirp.readthedocs.io/en/latest/index.html");
    addUrl(QString("&GitHub Repository"), "https://github.com/kncrabtree/blackchirp");
    addUrl(QString("Di&scord Server"),  "https://discord.gg/88CkbAKUZY");
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
    QDialog d(this);
    d.setWindowTitle(QString("Open experiment"));
    d.setWindowIcon(ThemeColors::createThemedIcon(":/icons/document-plus.svg", ThemeColors::IconPrimary, this));
    
    QVBoxLayout *vbl = new QVBoxLayout;
    QFormLayout *fl = new QFormLayout;
    QHBoxLayout *hl = new QHBoxLayout;

    QSpinBox *numBox = new QSpinBox(&d);
    
    // Get the last experiment number from Blackchirp's settings each time dialog opens
    SettingsStorage blackchirpSettings("CrabtreeLab", "Blackchirp");
    int lastExptNum = blackchirpSettings.get(BC::Key::exptNum, 0);
    
    if (lastExptNum > 0) {
        numBox->setRange(1, lastExptNum);
        numBox->setValue(lastExptNum);
    } else {
        numBox->setRange(1, __INT_MAX__);
        numBox->setValue(1);
    }
    
    fl->addRow(QString("Experiment Number"), numBox);

    QCheckBox *pathBox = new QCheckBox(QString("Specify custom path"), &d);
    fl->addRow(pathBox);

    QLineEdit *pathEdit = new QLineEdit(&d);
    QToolButton *browseButton = new QToolButton(&d);
    browseButton->setIcon(ThemeColors::createThemedIcon(":/icons/document-magnifying-glass.svg", ThemeColors::IconSecondary, this));

    connect(browseButton, &QToolButton::clicked, this, [this, pathEdit]() {
        QString startDir = get(BC::Key::Viewer::lastBrowseDir, QDir::homePath());
        QString path = QFileDialog::getExistingDirectory(this, QString("Select experiment directory"), startDir);
        if (!path.isEmpty()) {
            pathEdit->setText(path);
            set(BC::Key::Viewer::lastBrowseDir, path, true);
        }
    });

    // Initially disable path controls
    pathEdit->setEnabled(false);
    browseButton->setEnabled(false);

    connect(pathBox, &QCheckBox::toggled, [=](bool checked) {
        if (checked) {
            pathEdit->setEnabled(true);
            browseButton->setEnabled(true);
            numBox->setEnabled(false);
        } else {
            numBox->setEnabled(true);
            pathEdit->clear();
            pathEdit->setEnabled(false);
            browseButton->setEnabled(false);
        }
    });

    hl->addWidget(pathEdit, 1);
    hl->addWidget(browseButton, 0);
    fl->addRow(hl);

    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel, &d);
    connect(bb->button(QDialogButtonBox::Open), &QPushButton::clicked, &d, &QDialog::accept);
    connect(bb->button(QDialogButtonBox::Cancel), &QPushButton::clicked, &d, &QDialog::reject);

    vbl->addLayout(fl);
    vbl->addWidget(bb);
    d.setLayout(vbl);

    if (d.exec() == QDialog::Accepted) {
        QString path;
        if (pathBox->isChecked()) {
            path = pathEdit->text();
            if (path.isEmpty()) {
                QMessageBox::critical(this, QString("Load error"), QString("Cannot open experiment with an empty path."), QMessageBox::Ok);
                return;
            }

            QDir dir(path);
            if (!dir.exists()) {
                QMessageBox::critical(this, QString("Load error"), QString("The directory %1 does not exist. Could not load experiment.").arg(dir.absolutePath()), QMessageBox::Ok);
                return;
            }
        }

        int num = numBox->value();
        if (num < 1) {
            QMessageBox::critical(this, QString("Load error"), QString("Cannot open an experiment numbered below 1. (You chose %1)").arg(num), QMessageBox::Ok);
            return;
        }

        openExperimentByNumPath(num, path);
    }
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
        p_statusLabel->setText(QString("Closed experiment %1").arg(displayTextToRemove));
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
        QString statusText;
        if (d_dataPath.isEmpty()) {
            statusText = "Ready (No data path configured)";
        } else {
            statusText = QString("Ready (Data path: %1)").arg(d_dataPath);
        }
        p_statusLabel->setText(statusText);
    }
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

QString ViewerMainWindow::createDisplayText(int expNum, const QString& path) const
{
    if (path.isEmpty()) {
        return QString("Experiment %1").arg(expNum);
    } else {
        QDir dir(path);
        return QString("%1").arg(dir.absolutePath());
    }
}

void ViewerMainWindow::openExperimentByNumPath(int num, const QString &path)
{
    QString displayText = createDisplayText(num, path);

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

    p_statusLabel->setText(QString("Opened experiment %1").arg(displayText));
    updateButtonStates();

    addToRecentExperiments(num, path);
}

void ViewerMainWindow::addToRecentExperiments(int num, const QString &path)
{
    using namespace BC::Key::Viewer;

    auto recent = getArray(recentExperiments);

    // Remove any existing entry for the same experiment
    QString displayText = createDisplayText(num, path);
    recent.erase(std::remove_if(recent.begin(), recent.end(),
        [&](const SettingsStorage::SettingsMap &m) {
            int n = 0;
            QString p;
            auto nit = m.find(recentNum);
            if (nit != m.end()) n = nit->second.toInt();
            auto pit = m.find(recentPath);
            if (pit != m.end()) p = pit->second.toString();
            return createDisplayText(n, p) == displayText;
        }), recent.end());

    // Prepend new entry
    SettingsStorage::SettingsMap entry;
    entry[recentNum] = num;
    entry[recentPath] = path;
    recent.insert(recent.begin(), entry);

    // Trim to max size
    if (static_cast<int>(recent.size()) > MaxRecentExperiments)
        recent.resize(MaxRecentExperiments);

    setArray(recentExperiments, recent, true);
    updateRecentMenu();
}

void ViewerMainWindow::updateRecentMenu()
{
    using namespace BC::Key::Viewer;

    p_recentMenu->clear();

    auto recent = getArray(recentExperiments);
    if (recent.empty()) {
        p_recentMenu->addAction("(No recent experiments)")->setEnabled(false);
        return;
    }

    for (const auto &entry : recent) {
        int num = 0;
        QString path;
        auto nit = entry.find(recentNum);
        if (nit != entry.end()) num = nit->second.toInt();
        auto pit = entry.find(recentPath);
        if (pit != entry.end()) path = pit->second.toString();

        QString label = createDisplayText(num, path);
        auto action = p_recentMenu->addAction(label);
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
