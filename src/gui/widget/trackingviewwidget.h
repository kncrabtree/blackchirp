#ifndef TRACKINGVIEWWIDGET_H
#define TRACKINGVIEWWIDGET_H

#include <QWidget>

#include <QList>
#include <QDateTime>

#include <qwt6/qwt_plot.h>
#include <src/data/storage/settingsstorage.h>

class QGridLayout;
class TrackingPlot;
class QwtPlotCurve;

namespace BC::Key {
static const QString numPlots("numPlots");
static const QString viewonly("View");
static const QString plot("Plot");
}

class TrackingViewWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit TrackingViewWidget(const QString name, QWidget *parent = 0, bool viewOnly = false);
    ~TrackingViewWidget();

    //! Associates curves with which plot and axis they're displayed on
    struct CurveMetaData {
        QwtPlotCurve *curve; /*!< The curve */
        QVector<QPointF> data; /*!< The curve data */
        QString name; /*!< The name of the curve */
        int plotIndex; /*!< The index of the plot (in allPlots) the curve is plotted on */
        QwtPlot::Axis axis; /*!< The y-axis on which the curve is plotted */
        bool isVisible; /*!< Whether the curve is visible */
        double min;
        double max;
    };

    const QString d_name;

public slots:
    void initializeForExperiment();
    void pointUpdated(const QList<QPair<QString,QVariant> > list, bool plot, QDateTime t);
    void curveVisibilityToggled(QwtPlotCurve *c, bool visible);
    void curveContextMenuRequested(QwtPlotCurve *c, QMouseEvent *me);
    void changeCurveColor(int curveIndex);
    void moveCurveToPlot(int curveIndex, int newPlotIndex);
    void changeCurveAxis(int curveIndex);
    void pushXAxis(int sourcePlotIndex);
    void autoScaleAll();

    void changeNumPlots();


private:
    QGridLayout *p_gridLayout = nullptr;
    QList<CurveMetaData> d_plotCurves;
    QList<TrackingPlot*> d_allPlots;
    QPair<double,double> d_xRange;
    bool d_viewMode;

    int findCurveIndex(QwtPlotCurve* c);
    void addNewPlot();
    void configureGrid();
    void setAutoScaleYRanges(int plotIndex, QwtPlot::Axis axis);

};

#endif // TRACKINGVIEWWIDGET_H
