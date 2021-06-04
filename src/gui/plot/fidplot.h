#ifndef FIDPLOT_H
#define FIDPLOT_H

#include <src/gui/plot/zoompanplot.h>

#include <QPointF>
#include <QPair>

#include <src/data/experiment/experiment.h>

class QwtPlotMarker;
class QwtPlotCurve;

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
    explicit FidPlot(QString id, QWidget *parent = 0);

signals:
    void ftStartChanged(double);
    void ftEndChanged(double);


public slots:
    void receiveProcessedFid(const QVector<QPointF> d);
    void prepareForExperiment(const Experiment e);
    void setFtStart(double start);
    void setFtEnd(double end);
    void buildContextMenu(QMouseEvent *me);
    void changeFidColor();

private:
    QVector<QPointF> d_currentFid;
    QwtPlotCurve *p_curve;

    QPair<QwtPlotMarker*,QwtPlotMarker*> d_chirpMarkers;
    QPair<QwtPlotMarker*,QwtPlotMarker*> d_ftMarkers;

protected:
    void filterData();

};

#include <qwt6/qwt_scale_draw.h>

class SciNotationScaleDraw : public QwtScaleDraw
{
    virtual QwtText label(double d) const { return QwtText(QString("%1").arg(d,4,'e',1)); }
};


#endif // FIDPLOT_H
