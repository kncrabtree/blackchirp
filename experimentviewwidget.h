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
    explicit ExperimentViewWidget(int num, QString path = QString(""), QWidget *parent = 0);

    QSize sizeHint() const;

signals:
    void logMessage(QString msg, BlackChirp::LogMessageCode t = BlackChirp::LogNormal);

public slots:
    void exportAscii();

private:
    Experiment d_experiment;
    QTabWidget *p_tabWidget;
    FtmwViewWidget *p_ftmw;
    LogHandler *p_lh;

    QWidget *buildHeaderWidget();
    QWidget *buildFtmwWidget(QString path = QString(""));
    QWidget *buildLifWidget();
    QWidget *buildTrackingWidget();
    QWidget *buildLogWidget(QString path = QString(""));
};

#endif // EXPERIMENTVIEWWIDGET_H
