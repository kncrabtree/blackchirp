#ifndef PEAKFINDWIDGET_H
#define PEAKFINDWIDGET_H

#include <QWidget>

#include <QPair>
#include <QVector>
#include <QList>
#include <QPointF>
#include <QFutureWatcher>

#include <data/storage/settingsstorage.h>
#include <data/experiment/experiment.h>
#include <data/analysis/peakfinder.h>
#include <data/model/peaklistmodel.h>
#include <data/model/peaklistfilterproxymodel.h>

class QToolBar;
class QToolButton;
class QAction;
class QTableView;
class QTableWidget;
class QDoubleSpinBox;
class ScientificSpinBox;
class FtmwViewWidget;

namespace BC::Key {
inline constexpr QLatin1StringView peakFind{"peakFind"};
inline constexpr QLatin1StringView pfMinFreq{"minFreq"};
inline constexpr QLatin1StringView pfMaxFreq{"maxFreq"};
inline constexpr QLatin1StringView pfSnr{"snr"};
inline constexpr QLatin1StringView pfWinSize{"winSize"};
inline constexpr QLatin1StringView pfOrder{"polyOrder"};
// Display-filter state (distinct from pfMinFreq/pfMaxFreq, which bound
// the search). pfFilterMin/MaxFreq hold the spin-box values; a value at
// the box minimum means "unbounded" on that side.
inline constexpr QLatin1StringView pfFilterMinFreq{"filterMinFreq"};
inline constexpr QLatin1StringView pfFilterMaxFreq{"filterMaxFreq"};
inline constexpr QLatin1StringView pfFilterMinInt{"filterMinInt"};
inline constexpr QLatin1StringView pfFilterMaxInt{"filterMaxInt"};
inline constexpr QLatin1StringView pfFilterEnabled{"filterEnabled"};
inline constexpr QLatin1StringView pfViewSync{"filterViewSync"};
inline constexpr QLatin1StringView pfNavHalfWidth{"navHalfWidth"};
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

    /// Requests that a plot be centered on a selected peak.
    /// \param plotName  Target plot object name; empty means the main
    ///        FT plot (the double-click fast path).
    /// \param freq      Peak frequency (MHz).
    /// \param intensity Peak intensity, for y framing.
    /// \param halfWidth x half-width of the framed window (MHz).
    void centerPlotRequested(const QString &plotName, double freq,
                             double intensity, double halfWidth);

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

    /// Feeds the main FT plot's visible x range to the display filter.
    /// Connected unconditionally; only narrows the table while the
    /// "In view" control is on.
    void setMainPlotXRange(double min, double max);

private:
    PeakFinder *p_pf;
    PeakListModel *p_listModel;
    PeakListFilterProxyModel *p_proxy;
    std::unique_ptr<QFutureWatcher<void>> pu_watcher{std::make_unique<QFutureWatcher<void>>() };

    QToolBar *p_toolBar;
    QWidget *p_bottomBar;
    QList<QToolButton*> d_bottomButtons;
    QTableView *p_peakListView;
    QAction *p_findAction;
    QAction *p_liveAction;
    QAction *p_appearanceAction;
    QAction *p_filterAction;
    QAction *p_inViewAction;
    QAction *p_optionsAction;
    QAction *p_exportAction;
    QAction *p_removeAction;
    QAction *p_raiseParentAction;

    QTableWidget *p_filterGrid;
    QDoubleSpinBox *p_minFreqBox;
    QDoubleSpinBox *p_maxFreqBox;
    ScientificSpinBox *p_minIntBox;
    ScientificSpinBox *p_maxIntBox;

    double d_minFreq;
    double d_maxFreq;
    double d_snr;
    int d_winSize;
    int d_polyOrder;
    double d_navHalfWidth;
    int d_number;
    bool d_busy;
    bool d_waiting;
    bool d_dockHooked{false};
    Ft d_currentFt;

    void setupUI();
    void adjustToolbarStyle();
    void updateRaiseParentVisibility();

    // Pushes the current filter-strip controls into the proxy and
    // persists them under the peakFind group.
    void applyFilters();
    void persistFilterState();

    // Row context menu (Center on ▸) and the double-click fast path.
    void showPeakContextMenu(const QPoint &pos);
    void centerPlot(const QModelIndex &proxyIndex, const QString &plotName);

    // Keyboard navigation. The navigation cursor is tracked as a
    // source-model row so it survives proxy filtering and re-sorting;
    // Left/Right walk it by frequency across the full (unfiltered,
    // unsorted) peak list, Up/Down walk the visible table rows, Enter
    // re-centers the current peak.
    void centerSourceRow(int sourceRow);
    void navigateProxyRow(int delta);
    void navigateByFrequency(int dir);
    int d_navSrcRow{-1};

    // Walks the parent chain to the hosting FtmwViewWidget. The direct
    // parent is the QDockWidget, so a plain parentWidget() cast won't do.
    FtmwViewWidget *findFtmwView() const;

protected:
    // Compact preferred width so the dock matches the other tool
    // panels instead of stretching to fit the toolbar labels.
    QSize sizeHint() const override;
    void resizeEvent(QResizeEvent *event) override;
    void showEvent(QShowEvent *event) override;
    bool eventFilter(QObject *obj, QEvent *ev) override;
};

#endif // PEAKFINDWIDGET_H
