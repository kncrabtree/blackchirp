#ifndef EXPERIMENTVIEWWIDGET_H
#define EXPERIMENTVIEWWIDGET_H

#include <QWidget>

#include <data/experiment/experiment.h>
#include <data/analysis/ftworker.h>
#include <data/analysis/ft.h>

class QTabWidget;
class LogHandler;
class FtmwViewWidget;

class ExperimentViewWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ExperimentViewWidget(int num, QString path = QString(""), bool overlaysEnabled = true, QWidget *parent = 0);

    QSize sizeHint() const;
    FtWorker::FidProcessingSettings getFtmwProcessingSettings() const;
    Ft getMainPlotFt() const;
    void setCurrentTab(const QString &tabName);

    void notifyAlreadyOpen();

signals:
    void logMessage(QString msg, LogHandler::MessageCode t = LogHandler::Normal);
    void widgetClosing();

private:
    std::unique_ptr<Experiment> pu_experiment;
    QTabWidget *p_tabWidget;
    FtmwViewWidget *p_ftmw;
    LogHandler *p_lh;
    bool d_overlaysEnabled{true};

    QWidget *buildHeaderWidget();
    QWidget *buildFtmwWidget(QString path = QString(""));
    QWidget *buildTrackingWidget();
    QWidget *buildLogWidget(QString path = QString(""));

#ifdef BC_LIF
    QWidget *buildLifWidget();
#endif

protected:
    void closeEvent(QCloseEvent *event) override;
};

#endif // EXPERIMENTVIEWWIDGET_H
