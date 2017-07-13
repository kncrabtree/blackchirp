#ifndef FTMWVIEWWIDGET_H
#define FTMWVIEWWIDGET_H

#include <QWidget>

#include <QList>

#include "experiment.h"

class FtWorker;
class QThread;
class FtmwSnapshotWidget;

namespace Ui {
class FtmwViewWidget;
}

class FtmwViewWidget : public QWidget
{
    Q_OBJECT
public:
    explicit FtmwViewWidget(QWidget *parent = 0, QString path = QString(""));
    ~FtmwViewWidget();
    void prepareForExperiment(const Experiment e);

signals:
    void rollingAverageShotsChanged(int);
    void rollingAverageReset();
    void experimentLogMessage(int,QString,BlackChirp::LogMessageCode = BlackChirp::LogNormal,QString=QString(""));
    void finalized(int);

public slots:
    void togglePanel(bool on);
    void newFidList(QList<Fid> fl);
    void updateShotsLabel(qint64 shots);
    void ftStartChanged(double s);
    void ftEndChanged(double e);
    void removeDcChanged(bool rdc);
    void showProcessedChanged(bool p);
    void pzfChanged(int zpf);
    void scalingChanged(double scf);
    void winfChanged(BlackChirp::FtWindowFunction f);
    void updateFtPlot();
    void ftDone(QVector<QPointF> ft, double max);
    void ftDiffDone(QVector<QPointF> ft, double min, double max);
    void modeChanged();
    void snapshotTaken();
    void experimentComplete();
    void snapshotLoadError(QString msg);
    void snapListUpdate();
    void snapRefChanged();
    void snapDiffChanged();
    void finalizedSnapList(const QList<Fid> l);
    void removeSnapWidget();
    void checkRemoveSnapWidget(int num);


private:
    Ui::FtmwViewWidget *ui;

    BlackChirp::FtmwViewMode d_mode;
    bool d_replotWhenDone, d_processing;
    int d_pzf, d_currentExptNum;
    QList<Fid> d_currentFidList;
    Fid d_currentFid, d_currentRefFid;
    QThread *p_ftThread;
    FtWorker *p_ftw;
    FtmwSnapshotWidget *p_snapWidget;
    QString d_path;
};

#endif // FTMWVIEWWIDGET_H
