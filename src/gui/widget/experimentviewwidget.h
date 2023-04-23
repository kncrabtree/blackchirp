#ifndef EXPERIMENTVIEWWIDGET_H
#define EXPERIMENTVIEWWIDGET_H

#include <QWidget>

#include <data/experiment/experiment.h>

class QTabWidget;
class LogHandler;
class FtmwViewWidget;

class ExperimentViewWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ExperimentViewWidget(int num, QString path = QString(""), QWidget *parent = 0);

    QSize sizeHint() const;

signals:
    void logMessage(QString msg, LogHandler::MessageCode t = LogHandler::Normal);

private:
    std::unique_ptr<Experiment> pu_experiment;
    QTabWidget *p_tabWidget;
    FtmwViewWidget *p_ftmw;
    LogHandler *p_lh;

    QWidget *buildHeaderWidget();
    QWidget *buildFtmwWidget(QString path = QString(""));
    QWidget *buildTrackingWidget();
    QWidget *buildLogWidget(QString path = QString(""));

#ifdef BC_LIF
    QWidget *buildLifWidget();
#endif
};

#endif // EXPERIMENTVIEWWIDGET_H
