#ifndef TRACKINGVIEWWIDGET_H
#define TRACKINGVIEWWIDGET_H

#include <QWidget>

#include <QList>
#include <QDateTime>

#include <qwt6/qwt_plot.h>

class QGridLayout;
class TrackingPlot;
class QwtPlotCurve;

class TrackingViewWidget : public QWidget
{
    Q_OBJECT

public:
    explicit TrackingViewWidget(bool viewOnly = false, QWidget *parent = 0);
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
    QGridLayout *d_gridLayout = nullptr;
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
