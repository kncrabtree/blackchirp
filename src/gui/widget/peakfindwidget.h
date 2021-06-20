#ifndef PEAKFINDWIDGET_H
#define PEAKFINDWIDGET_H

#include <QWidget>

#include <QSortFilterProxyModel>
#include <QPair>
#include <QVector>
#include <QPointF>

#include <src/data/storage/settingsstorage.h>
#include <src/data/experiment/experiment.h>
#include <src/data/analysis/peakfinder.h>
#include <src/data/model/peaklistmodel.h>

namespace Ui {
class PeakFindWidget;
}

namespace BC::Key {
static const QString peakFind("peakFind");
static const QString pfMinFreq("minFreq");
static const QString pfMaxFreq("maxFreq");
static const QString pfSnr("snr");
static const QString pfWinSize("winSize");
static const QString pfOrder("polyOrder");
}

class PeakFindWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit PeakFindWidget(Ft ft, QWidget *parent = 0);
    ~PeakFindWidget();

signals:
    void peakList(const QList<QPointF>);

public slots:
    void newFt(const Ft ft);
    void newPeakList(const QList<QPointF> pl);
    void findPeaks();
    void removeSelected();
    void updateRemoveButton();
    void changeScaleFactor(double scf);
    void launchOptionsDialog();
    void launchExportDialog();

private:
    Ui::PeakFindWidget *ui;

    PeakFinder *p_pf;
    PeakListModel *p_listModel;
    QSortFilterProxyModel *p_proxy;

    double d_minFreq;
    double d_maxFreq;
    double d_snr;
    int d_winSize;
    int d_polyOrder;
    int d_number;
    bool d_busy;
    bool d_waiting;
    Ft d_currentFt;
    QThread *p_thread;
};

#endif // PEAKFINDWIDGET_H
