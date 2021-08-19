#ifndef PEAKFINDWIDGET_H
#define PEAKFINDWIDGET_H

#include <QWidget>

#include <QSortFilterProxyModel>
#include <QPair>
#include <QVector>
#include <QPointF>
#include <QFutureWatcher>

#include <data/storage/settingsstorage.h>
#include <data/experiment/experiment.h>
#include <data/analysis/peakfinder.h>
#include <data/model/peaklistmodel.h>

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
    explicit PeakFindWidget(Ft ft, int number, QWidget *parent = nullptr);
    ~PeakFindWidget();

signals:
    void peakList(QVector<QPointF>);

public slots:
    void newFt(const Ft ft);
    void newPeakList(const QVector<QPointF> pl);
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
    std::unique_ptr<QFutureWatcher<void>> pu_watcher{std::make_unique<QFutureWatcher<void>>() };

    double d_minFreq;
    double d_maxFreq;
    double d_snr;
    int d_winSize;
    int d_polyOrder;
    int d_number;
    bool d_busy;
    bool d_waiting;
    Ft d_currentFt;
};

#endif // PEAKFINDWIDGET_H
