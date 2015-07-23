#ifndef FTMWVIEWWIDGET_H
#define FTMWVIEWWIDGET_H

#include <QWidget>

#include <QList>

#include "experiment.h"

class FtWorker;
class QThread;

namespace Ui {
class FtmwViewWidget;
}

class FtmwViewWidget : public QWidget
{
    Q_OBJECT
public:
    explicit FtmwViewWidget(QWidget *parent = 0);
    ~FtmwViewWidget();
    void prepareForExperiment(const Experiment e);

signals:
    void rollingAverageShotsChanged(int);
    void rollingAverageReset();

public slots:
    void newFidList(QList<Fid> fl);
    void updateShotsLabel(qint64 shots);
    void showFrame(int num);
    void ftStartChanged(double s);
    void ftEndChanged(double e);
    void pzfChanged(int zpf);
    void updateFtPlot();
    void ftDone(QVector<QPointF> ft, double max);


private:
    Ui::FtmwViewWidget *ui;

    BlackChirp::FtmwViewMode d_mode;
    bool d_replotWhenDone, d_processing;
    int d_pzf;
    QList<Fid> d_currentFidList;
    Fid d_currentFid;
    QThread *p_ftThread;
    FtWorker *p_ftw;
};

#endif // FTMWVIEWWIDGET_H
