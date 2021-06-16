#ifndef FIDPLOT_H
#define FIDPLOT_H

#include <src/gui/plot/zoompanplot.h>

#include <QPointF>
#include <QPair>

#include <src/data/experiment/experiment.h>

class QwtPlotMarker;

namespace BC::Key {
static const QString fidPlot("FidPlot");
static const QString fidCurve("FID");
}

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
    explicit FidPlot(const QString id, QWidget *parent = 0);

signals:
    void ftStartChanged(double);
    void ftEndChanged(double);


public slots:
    void receiveProcessedFid(const QVector<QPointF> d);
    void prepareForExperiment(const Experiment e);
    void setFtStart(double start);
    void setFtEnd(double end);

private:
    QVector<QPointF> d_currentFid;
    BlackchirpPlotCurve *p_curve;

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
