#ifndef ZOOMPANPLOT_H
#define ZOOMPANPLOT_H

#include <QFutureWatcher>

#include <qwt6/qwt_plot.h>
#include <qwt6/qwt_symbol.h>
#include <qwt6/qwt_plot_grid.h>
#include <qwt6/qwt_scale_engine.h>

#include <data/storage/settingsstorage.h>
#include <gui/plot/customzoomer.h>

class QMenu;
class CustomTracker;
class BlackchirpPlotCurve;
class BlackchirpPlotCurveBase;
class QMutex;


namespace BC::Key {
static const QString axes{"axes"};
static const QString bottom{"Bottom"};
static const QString top{"Top"};
static const QString left{"Left"};
static const QString right{"Right"};
static const QString zoomFactor{"zoomFactor"};
static const QString trackerDecimals{"trackerDecimals"};
static const QString trackerScientific{"trackerScientific"};
static const QString trackerEn{"trackerEnabled"};
static const QString majorGridColor{"majorGridColor"};
static const QString majorGridStyle{"majorGridStyle"};
static const QString minorGridColor{"minorGridColor"};
static const QString minorGridStyle{"minorGridStyle"};
}

class ZoomPanPlot : public QwtPlot, public SettingsStorage
{
    Q_OBJECT
public:
    explicit ZoomPanPlot(const QString name, QWidget *parent = nullptr);
    virtual ~ZoomPanPlot();

    bool isAutoScale();
    void resetPlot();
    void setSpectrogramMode(bool b = true);
    void setXRanges(const QwtScaleDiv &bottom, const QwtScaleDiv &top);
    void setMaxIndex(int i){ d_maxIndex = i; }
    void setPlotTitle(const QString text);
    void setPlotAxisTitle(QwtPlot::Axis a, const QString text);

    const QString d_name;

public slots:
    void autoScale();
    void overrideAxisAutoScaleRange(QwtPlot::Axis a, double min, double max);
    void clearAxisAutoScaleOverride(QwtPlot::Axis a);
    virtual void replot();
    void setZoomFactor(QwtPlot::Axis a, double v);
    void setTrackerEnabled(bool en);
    void setTrackerDecimals(QwtPlot::Axis a, int dec);
    void setTrackerScientific(QwtPlot::Axis a, bool sci);

    void exportCurve(BlackchirpPlotCurveBase *curve);
    void setCurveColor(BlackchirpPlotCurveBase* curve);
    void setCurveLineThickness(BlackchirpPlotCurveBase* curve, double t);
    void setCurveLineStyle(BlackchirpPlotCurveBase* curve, Qt::PenStyle s);
    void setCurveMarker(BlackchirpPlotCurveBase* curve, QwtSymbol::Style s);
    void setCurveMarkerSize(BlackchirpPlotCurveBase* curve, int s);
    void setCurveVisible(BlackchirpPlotCurveBase* curve, bool v);
    void setCurveAxisY(BlackchirpPlotCurveBase* curve, QwtPlot::Axis a);
    void configureGridMajorPen();
    void configureGridMinorPen();

signals:
    void panningStarted();
    void panningFinished();
    void plotRightClicked(QMouseEvent *ev);
    void curveMoveRequested(BlackchirpPlotCurve*, int);

protected:
    int d_maxIndex;
    QwtPlotGrid *p_grid;
    CustomTracker *p_tracker;
    CustomZoomer *p_zoomerLB, *p_zoomerRT;

    struct AxisConfig {
        QwtPlot::Axis type;
        bool autoScale {true};
        bool override {false};
        bool overrideAutoScaleRange {false};
        QRectF overrideRect{1.0,1.0,-2.0,-2.0};
        QRectF boundingRect{1.0,1.0,-2.0,-2.0};
        double zoomFactor {0.1};
        QString name;

        explicit AxisConfig(QwtPlot::Axis t, const QString n) : type(t), name(n) {}
    };

    struct PlotConfig {
        QList<AxisConfig> axisList;

        bool xDirty{false};
        bool panning{false};
        bool spectrogramMode{false};
        QPoint panClickPos;
        bool zoomXLock{false};
        bool zoomYLock{false};
    };

    void setAxisOverride(QwtPlot::Axis axis, bool override = true);

    virtual void filterData();
    virtual void resizeEvent(QResizeEvent *ev);
    virtual bool eventFilter(QObject *obj, QEvent *ev);
    virtual void pan(QMouseEvent *me);
    virtual void zoom(QWheelEvent *we);
    virtual void zoom(const QRectF &rect, QwtPlot::Axis xAx, QwtPlot::Axis yAx);

    virtual void buildContextMenu(QMouseEvent *me);
    virtual QMenu *contextMenu();

private:
    PlotConfig d_config;
    int getAxisIndex(QwtPlot::Axis a);
    QFutureWatcher<void> *p_watcher;
    bool d_busy{false};
    QMutex *p_mutex;

    // QWidget interface
public:
    virtual QSize sizeHint() const;
    virtual QSize minimumSizeHint() const;

    // QWidget interface
protected:
    virtual void showEvent(QShowEvent *event);
};

#endif // ZOOMPANPLOT_H
