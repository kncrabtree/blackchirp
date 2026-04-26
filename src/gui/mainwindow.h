#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QList>
#include <QPair>
#include <QVector>
#include <QMetaObject>
#include <memory>
#include <map>

#include <gui/plot/zoompanplot.h>
#include <data/experiment/experiment.h>

class QThread;
class QCloseEvent;
class QActionGroup;
class LogHandler;
class AcquisitionManager;
class HardwareManager;
class Led;
class BatchManager;
class QLabel;
class QLineEdit;
class QDoubleSpinBox;
class QProgressBar;
class QAction;
class HWDialog;
class QuickExptDialog;
class LifControlWidget;
class RfConfigWidget;
class ExperimentViewWidget;

namespace Ui {
class MainWindow;
inline constexpr QLatin1StringView actionStr{"Action"};
inline constexpr QLatin1StringView sbStr{"StatusBox"};
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    enum ProgramState
    {
        Idle,
        Acquiring,
        Paused,
        Disconnected,
        Asleep
    };

    void initializeHardware();

signals:
    void logMessage(QString, LogHandler::MessageCode);
    void startInit();
    void closing();
    void checkSleep();

public slots:
    void startExperiment();
    void quickStart();
    void startSequence();
    bool runExperimentWizard(Experiment *exp, QuickExptDialog *qed = nullptr);
    void configureOptionalHardware(Experiment *exp, QuickExptDialog *qed = nullptr);
    void batchComplete(bool aborted);
    void experimentInitialized(std::shared_ptr<Experiment> exp);
    void hardwareInitialized(bool success);
    void clockPrompt(QHash<RfConfig::ClockType,RfConfig::ClockFreq> c);
    void pauseUi();
    void resumeUi();
    void launchCommunicationDialog(bool parent = true);
    void launchFtmwConfigDialog();
    void rebuildLoadoutMenu();
    void onLoadoutActionTriggered(QAction *act);
    void rebuildFtmwPresetMenu();
    void onFtmwPresetActionTriggered(QAction *act);
    void launchLifConfigDialog();
    void launchRuntimeHardwareConfigDialog();
    void configureLifWidget(LifControlWidget *w);
    void connectRfConfigWidget(RfConfigWidget *w);
    void setLogIcon(LogHandler::MessageCode c);
    void sleep(bool s);
    void viewExperiment();

    bool isDialogOpen(const QString key);
    HWDialog *createHWDialog(const QString key, QWidget *controlWidget = nullptr);

private:
    Ui::MainWindow *ui;
    QList<QPair<QThread*,QObject*> > d_threadObjectList;
    std::map<QString,QDialog*,std::less<>> d_openDialogs;
    LogHandler *p_lh;
    HardwareManager *p_hwm;
    AcquisitionManager *p_am;

    bool d_hardwareConnected{false};

    // Hardware UI management structures
    struct HardwareUIElements {
        QAction* menuAction;
        QWidget* statusWidget;  // GasFlowDisplayBox, PressureStatusBox, etc.
        QVector<QMetaObject::Connection> connections;
    };
    std::map<QString, HardwareUIElements, std::less<>> d_hardwareUI;  // key.label -> UI elements
    std::map<QString, bool, std::less<>> d_hardwareConnectionState;  // key.label -> connection status

    void configureUi(ProgramState s);
    void startBatch(BatchManager *bm);
    void removeExperimentWidget(const QString& path);
    void updateViewExperimentMenu();
    void showExistingExperiment(const QString& path);
    void setupThemeAwareIconStyling();
    
    // Factory method for creating experiments with proper hardware data
    std::shared_ptr<Experiment> createExperiment();
    
    // Hardware UI management methods
    void buildHardwareUI();
    void clearHardwareUI();
    void updateHardwareConnectionState(const QString& hwKey, bool connected);
    void configureUiForHardwareState();
    bool isCriticalHardwareConnected() const;
    QWidget *wrapWithPythonWidget(const QString &hwKey, QWidget *typeWidget);

    QActionGroup *p_loadoutActionGroup{nullptr};
    QActionGroup *p_ftmwPresetActionGroup{nullptr};
    ProgramState d_state{Idle};
    bool d_initialHardwareTestComplete{false};
    int d_logCount{0};
    LogHandler::MessageCode d_logIcon{LogHandler::Normal};
    int d_currentExptNum{0};
    BatchManager *p_batchManager{nullptr};
    
    // Track open experiment view widgets by experiment path
    std::map<QString, std::unique_ptr<ExperimentViewWidget>, std::less<>> d_openExperiments;


protected:
    void closeEvent(QCloseEvent *ev);
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    QString d_savePath;
    void updateSavePathLabel();

};

#endif // MAINWINDOW_H
