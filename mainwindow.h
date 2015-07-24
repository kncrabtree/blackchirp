#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QList>
#include <QPair>

#include "experiment.h"

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
        Paused,
        Disconnected,
        Asleep
    };

    struct FlowWidgets {
        QLineEdit *nameEdit;
        QDoubleSpinBox *controlBox;
        Led *led;
        QLabel *nameLabel;
        QDoubleSpinBox *displayBox;
    };

    void initializeHardware();

signals:
    void startInit();
    void statusMessage(const QString);
    void closing();

public slots:
    void startExperiment();
    void batchComplete(bool aborted);
    void experimentInitialized(const Experiment exp);
    void hardwareInitialized(bool success);
    void pauseUi();
    void resumeUi();
    void launchCommunicationDialog();
    void launchIOBoardDialog();
    void launchRfConfigDialog();
    void updatePulseLeds(const PulseGenConfig cc);
    void updatePulseLed(int index,BlackChirp::PulseSetting s, QVariant val);
    void updateFlow(int ch, double val);
    void updateFlowName(int ch, QString name);
    void updateFlowSetpoint(int ch, double val);
    void updatePressureSetpoint(double val);
    void updatePressureControl(bool en);
    void setLogIcon(BlackChirp::LogMessageCode c);
    void sleep(bool s);

private:
    Ui::MainWindow *ui;
    QList<QPair<QThread*,QObject*> > d_threadObjectList;
    QList<QPair<QLabel*,Led*>> d_ledList;
    QList<FlowWidgets> d_flowWidgets;
    LogHandler *p_lh;
    HardwareManager *p_hwm;
    AcquisitionManager *p_am;

    bool d_hardwareConnected;
    QThread *d_batchThread;

    void configureUi(ProgramState s);
    void startBatch(BatchManager *bm);
    FlowConfig getFlowConfig();

    ProgramState d_state;
    int d_logCount;
    BlackChirp::LogMessageCode d_logIcon;

protected:
    void closeEvent(QCloseEvent *ev);

private slots:
    void test();

};

#endif // MAINWINDOW_H
