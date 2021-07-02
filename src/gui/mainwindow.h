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

#ifdef BC_LIF
class LifControlWidget;
class LifDisplayWidget;
#endif

#ifdef BC_MOTOR
class MotorDisplayWidget;
class MotorStatusWidget;
#endif



namespace Ui {
class MainWindow;
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
        Peaking,
        Paused,
        Disconnected,
        Asleep
    };

    void initializeHardware();

signals:
    void logMessage(const QString, BlackChirp::LogMessageCode);
    void startInit();
    void statusMessage(const QString);
    void closing();
    void checkSleep();

public slots:
    void startExperiment();
    void quickStart();
    void startSequence();
    void batchComplete(bool aborted);
    void experimentInitialized(std::shared_ptr<Experiment> exp);
    void hardwareInitialized(bool success);
    void pauseUi();
    void resumeUi();
    void launchCommunicationDialog();
    void updatePulseLeds(const PulseGenConfig cc);
    void updatePulseLed(int index,PulseGenConfig::Setting s, QVariant val);
    void setLogIcon(BlackChirp::LogMessageCode c);
    void sleep(bool s);
    void viewExperiment();

#ifdef BC_PCONTROLLER
    void configPController(bool readOnly);
#endif

private:
    Ui::MainWindow *ui;
    QList<QPair<QThread*,QObject*> > d_threadObjectList;
    QList<QPair<QLabel*,Led*>> d_ledList;
    LogHandler *p_lh;
    HardwareManager *p_hwm;
    AcquisitionManager *p_am;

    bool d_hardwareConnected;
    bool d_oneExptDone;
    QThread *p_batchThread;

    void configureUi(ProgramState s);
    void startBatch(BatchManager *bm);

    ProgramState d_state;
    int d_logCount;
    BlackChirp::LogMessageCode d_logIcon;
    int d_currentExptNum;

#ifdef BC_LIF
    QWidget *p_lifTab;
    LifControlWidget *p_lifControlWidget;
    QProgressBar *p_lifProgressBar;
    QAction *p_lifAction;
    LifDisplayWidget *p_lifDisplayWidget;
#endif

#ifdef BC_MOTOR
    QAction *p_motorViewAction;
    QWidget *p_motorTab;
    MotorDisplayWidget *p_motorDisplayWidget;
    MotorStatusWidget *p_motorStatusWidget;
#endif

    QWidget *p_pcBox;

protected:
    void closeEvent(QCloseEvent *ev);

};

#endif // MAINWINDOW_H
