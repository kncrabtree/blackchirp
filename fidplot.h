#ifndef FIDPLOT_H
#define FIDPLOT_H
#include "zoompanplot.h"
#include <QObject>
#include <QVector>
#include <qwt6/qwt_plot_curve.h>
#include <QResizeEvent>
#include <qwt6/qwt_scale_draw.h>
#include <qwt6/qwt_plot_marker.h>
#include <QPointF>
#include "fid.h"
#include <QPair>

/*!
 * \brief The FID Plot
 */
class FidPlot : public ZoomPanPlot
{
    Q_OBJECT
public:

    /*!
     * \brief Constructor. Initializes axes and plot options
     * \param parent Parent widget
     */
    explicit FidPlot(QWidget *parent = 0);

signals:
    void overrideStart(double);
    void overrideEnd(double);
    void ftStartChanged(double);
    void ftEndChanged(double);

public slots:
    void receiveData(const Fid f);
    void initialize(double chirpStart, double chirpEnd, bool displayMarkers = true);
    void setFtStart(double start);
    void setFtEnd(double end);
    void buildContextMenu(QMouseEvent *me);

private:
    Fid d_currentFid;
    QwtPlotCurve *d_curve;
    QPair<double,double> d_yMinMax;

    QPair<QwtPlotMarker*,QwtPlotMarker*> d_chirpMarkers;
    QPair<QwtPlotMarker*,QwtPlotMarker*> d_ftMarkers;
    bool d_ftEndAtFidEnd;

protected:
    void filterData();
    void resizeEvent(QResizeEvent *e);

};


class SciNotationScaleDraw : public QwtScaleDraw
{
    virtual QwtText label(double d) const { return QwtText(QString("%1").arg(d,4,'e',1)); }
};


#endif // FIDPLOT_H
