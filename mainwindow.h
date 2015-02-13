#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QList>
#include <QPair>
#include <QThread>
#include "loghandler.h"
#include "hardwaremanager.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    QList<QPair<QThread*,QObject*> > d_threadList;

    LogHandler *p_lh;
    HardwareManager *p_hwm;


};

#endif // MAINWINDOW_H
