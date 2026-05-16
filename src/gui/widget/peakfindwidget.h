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

class QToolBar;
class QAction;
class QTableView;
class FtmwViewWidget;

namespace BC::Key {
inline constexpr QLatin1StringView peakFind{"peakFind"};
inline constexpr QLatin1StringView pfMinFreq{"minFreq"};
inline constexpr QLatin1StringView pfMaxFreq{"maxFreq"};
inline constexpr QLatin1StringView pfSnr{"snr"};
inline constexpr QLatin1StringView pfWinSize{"winSize"};
inline constexpr QLatin1StringView pfOrder{"polyOrder"};
}

class PeakFindWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit PeakFindWidget(Ft ft, int number, QWidget *parent = nullptr);
    ~PeakFindWidget();

signals:
    void peakList(QVector<QPointF>);

    /// Requests that the host pop the peak-marker appearance editor.
    /// \param globalPos Screen position to anchor the menu (the
    ///        Appearance toolbar button's lower-left corner).
    void editPeakAppearanceRequested(const QPoint &globalPos);

public slots:
    void newFt(const Ft ft);
    void newPeakList(const QVector<QPointF> pl);
    void findPeaks();
    void removeSelected();
    void updateRemoveButton();
    void changeScaleFactor(double scf);
    void launchOptionsDialog();
    void launchExportDialog();
    void raiseParent();

private:
    PeakFinder *p_pf;
    PeakListModel *p_listModel;
    QSortFilterProxyModel *p_proxy;
    std::unique_ptr<QFutureWatcher<void>> pu_watcher{std::make_unique<QFutureWatcher<void>>() };

    QToolBar *p_toolBar;
    QTableView *p_peakListView;
    QAction *p_findAction;
    QAction *p_liveAction;
    QAction *p_appearanceAction;
    QAction *p_optionsAction;
    QAction *p_exportAction;
    QAction *p_removeAction;
    QAction *p_raiseParentAction;

    double d_minFreq;
    double d_maxFreq;
    double d_snr;
    int d_winSize;
    int d_polyOrder;
    int d_number;
    bool d_busy;
    bool d_waiting;
    bool d_dockHooked{false};
    Ft d_currentFt;

    void setupUI();
    void adjustToolbarStyle();
    void updateRaiseParentVisibility();

    // Walks the parent chain to the hosting FtmwViewWidget. The direct
    // parent is the QDockWidget, so a plain parentWidget() cast won't do.
    FtmwViewWidget *findFtmwView() const;

protected:
    void resizeEvent(QResizeEvent *event) override;
    void showEvent(QShowEvent *event) override;
};

#endif // PEAKFINDWIDGET_H
