#ifndef FIDPLOT_H
#define FIDPLOT_H

#include "zoompanplot.h"

#include <QPointF>
#include <QPair>

#include "experiment.h"

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
    explicit FidPlot(QWidget *parent = 0);

signals:
    void overrideStart(double);
    void overrideEnd(double);
    void ftStartChanged(double);
    void ftEndChanged(double);
    void removeDcChanged(bool);

public slots:
    void receiveData(const Fid f);
    void prepareForExperiment(const Experiment e);
    void setFtStart(double start);
    void setFtEnd(double end);
    void removeDc(bool rdc);
    void buildContextMenu(QMouseEvent *me);
    void changeFidColor();

private:
    Fid d_currentFid;
    QwtPlotCurve *p_curve;

    QPair<QwtPlotMarker*,QwtPlotMarker*> d_chirpMarkers;
    QPair<QwtPlotMarker*,QwtPlotMarker*> d_ftMarkers;
    bool d_removeDc;
    bool d_ftEndAtFidEnd;
    int d_number;

protected:
    void filterData();

};

#include <qwt6/qwt_scale_draw.h>

class SciNotationScaleDraw : public QwtScaleDraw
{
    virtual QwtText label(double d) const { return QwtText(QString("%1").arg(d,4,'e',1)); }
};


#endif // FIDPLOT_H
