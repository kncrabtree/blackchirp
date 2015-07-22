#ifndef EXPERIMENTVIEWWIDGET_H
#define EXPERIMENTVIEWWIDGET_H

#include <QWidget>

#include "experiment.h"
#include "datastructs.h"

class QTabWidget;
class LogHandler;
class FtmwViewWidget;

class ExperimentViewWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ExperimentViewWidget(int num, QWidget *parent = 0);

signals:
    void logMessage(QString msg, BlackChirp::LogMessageCode t = BlackChirp::LogNormal);

public slots:

private:
    Experiment d_experiment;
    QTabWidget *p_tabWidget;
    FtmwViewWidget *p_ftmw;
    LogHandler *p_lh;

    QWidget *buildHeaderWidget();
    QWidget *buildFtmwWidget();
    QWidget *buildTrackingWidget();
    QWidget *buildLogWidget();
};

#endif // EXPERIMENTVIEWWIDGET_H
