#ifndef VIEWERMAINWINDOW_H
#define VIEWERMAINWINDOW_H

#include <QMainWindow>
#include <QListWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpinBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QToolButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QMenuBar>
#include <QCloseEvent>
#include <QAction>
#include <memory>
#include <map>

#include <gui/widget/experimentviewwidget.h>
#include <gui/style/themecolors.h>
#include <data/storage/settingsstorage.h>

class QMenu;

namespace BC::Key::Viewer {
static const QString recentExperiments{"recentExperiments"};
static const QString lastBrowseDir{"lastBrowseDir"};
static const QString recentNum{"num"};
static const QString recentPath{"path"};
}

class ViewerMainWindow : public QMainWindow, public SettingsStorage
{
    Q_OBJECT

public:
    explicit ViewerMainWindow(QWidget *parent = nullptr);
    ~ViewerMainWindow();

private slots:
    void openExperiment();
    void closeSelectedExperiment();
    void onExperimentWidgetClosing();
    void onListItemDoubleClicked(QListWidgetItem *item);

private:
    QWidget *p_centralWidget;
    QListWidget *p_experimentList;
    QLabel *p_statusLabel;

    // Shared actions for menu and toolbar
    QAction *p_openAction;
    QAction *p_closeAction;
    QMenu *p_recentMenu;

    // Track open experiment view widgets by their display text
    std::map<QString, std::unique_ptr<ExperimentViewWidget>> d_openExperiments;

    // Blackchirp's data storage path
    QString d_dataPath;

    static constexpr int MaxRecentExperiments = 10;

    void setupUI();
    void setupMenuBar();
    void updateButtonStates();
    void removeExperimentFromList(const QString& displayText);
    QString createDisplayText(int expNum, const QString& path = QString()) const;
    void openExperimentByNumPath(int num, const QString &path);
    void addToRecentExperiments(int num, const QString &path);
    void updateRecentMenu();

protected:
    void closeEvent(QCloseEvent *event) override;
};

#endif // VIEWERMAINWINDOW_H
