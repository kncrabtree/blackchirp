#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QList>
#include <QPair>

#include <gui/plot/zoompanplot.h>
#include <data/experiment/experiment.h>

class QThread;
class QCloseEvent;
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

namespace Ui {
class MainWindow;
static const QString actionStr{"Action"};
static const QString sbStr{"StatusBox"};
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
    void launchRfConfigDialog();
#ifdef BC_LIF
    void launchLifConfigDialog();
    void configureLifWidget(LifControlWidget *w);
#endif
    void setLogIcon(LogHandler::MessageCode c);
    void sleep(bool s);
    void viewExperiment();

    bool isDialogOpen(const QString key);
    HWDialog *createHWDialog(const QString key, QWidget *controlWidget = nullptr);

private:
    Ui::MainWindow *ui;
    QList<QPair<QThread*,QObject*> > d_threadObjectList;
    std::map<QString,QString> d_hardware;
    std::map<QString,QDialog*> d_openDialogs;
    LogHandler *p_lh;
    HardwareManager *p_hwm;
    AcquisitionManager *p_am;

    bool d_hardwareConnected{false};

    void configureUi(ProgramState s);
    void startBatch(BatchManager *bm);

    ProgramState d_state{Idle};
    int d_logCount{0};
    LogHandler::MessageCode d_logIcon{LogHandler::Normal};
    int d_currentExptNum{0};
    BatchManager *p_batchManager{nullptr};


protected:
    void closeEvent(QCloseEvent *ev);

};

#endif // MAINWINDOW_H
