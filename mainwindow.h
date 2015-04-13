#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QList>
#include <QPair>
#include <QThread>
#include <QCloseEvent>
#include "loghandler.h"
#include "hardwaremanager.h"
#include "acquisitionmanager.h"
#include "batchmanager.h"
#include "batchsingle.h"

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

signals:
    void statusMessage(const QString);

public slots:
    void startExperiment();
    void batchComplete(bool aborted);
    void experimentInitialized(Experiment exp);
    void hardwareInitialized(bool success);
    void pauseUi();
    void resumeUi();
    void launchCommunicationDialog();
    void launchRfConfigDialog();

private:
    Ui::MainWindow *ui;
    QList<QPair<QThread*,QObject*> > d_threadObjectList;
    LogHandler *p_lh;
    HardwareManager *p_hwm;
    AcquisitionManager *p_am;

    bool d_hardwareConnected;
    QThread *d_batchThread;

    void configureUi(ProgramState s);
    void startBatch(BatchManager *bm, bool sleepWhenDone = false);

    ProgramState d_state;

protected:
    void closeEvent(QCloseEvent *ev);

};

#endif // MAINWINDOW_H
