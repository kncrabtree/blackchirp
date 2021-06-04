#ifndef EXPERIMENTVIEWWIDGET_H
#define EXPERIMENTVIEWWIDGET_H

#include <QWidget>

#include <src/data/experiment/experiment.h>
#include <src/data/datastructs.h>

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
    void notifyUiFinalized(int);

public slots:
    void exportAscii();
    void ftmwFinalized(int num);

private:
    Experiment d_experiment;
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

#ifdef BC_MOTOR
    QWidget *buildMotorWidget();
#endif
};

#endif // EXPERIMENTVIEWWIDGET_H
