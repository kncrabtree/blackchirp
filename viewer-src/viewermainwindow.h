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
class UpdateChecker;
class ClickableLabel;

namespace BC::Key::Viewer {
// Group name for every viewer-owned setting in Blackchirp2.conf. The
// acquisition app owns [Blackchirp]; the viewer owns [BlackchirpViewer].
// Sharing the .conf file lets the viewer read Blackchirp's savePath on
// startup with a transient default-ctor SettingsStorage, while keeping
// the two apps' writeable state physically separate.
inline constexpr QLatin1StringView viewer{"BlackchirpViewer"};

// User-supplied override of Blackchirp's savePath. Empty (or absent)
// means follow Blackchirp's value.
inline constexpr QLatin1StringView dataPath{"dataPath"};

inline constexpr QLatin1StringView recentExperiments{"recentExperiments"};
inline constexpr QLatin1StringView lastBrowseDir{"lastBrowseDir"};
inline constexpr QLatin1StringView recentNum{"num"};
inline constexpr QLatin1StringView recentPath{"path"};
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
    void chooseDataPath();
    void resetDataPath();
    void onCheckForUpdatesTriggered();

private:
    QWidget *p_centralWidget;
    QListWidget *p_experimentList;
    ClickableLabel *p_dataPathLabel;
    QToolButton *p_dataPathConfigButton{nullptr};
    QLabel *p_statusLabel;

    // Shared actions for menu and toolbar
    QAction *p_openAction;
    QAction *p_closeAction;
    QMenu *p_recentMenu;
    UpdateChecker *p_updateChecker{nullptr};
    QAction *p_checkForUpdatesAction{nullptr};
    QMenu *p_helpMenu{nullptr};

    // Track open experiment view widgets by their display text
    std::map<QString, std::unique_ptr<ExperimentViewWidget>> d_openExperiments;

    // Blackchirp's data storage path
    QString d_dataPath;

    static constexpr int MaxRecentExperiments = 10;

    void setupUI();
    void setupMenuBar();
    void updateButtonStates();
    void removeExperimentFromList(const QString& displayText);
    void openExperimentByNumPath(int num, const QString &path);
    void addToRecentExperiments(int num, const QString &path);
    void updateRecentMenu();
    void loadActiveDataPath();
    void updateDataPathLabel();

protected:
    void closeEvent(QCloseEvent *event) override;
    bool eventFilter(QObject *watched, QEvent *event) override;
};

#endif // VIEWERMAINWINDOW_H
