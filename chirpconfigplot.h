#ifndef CHIRPCONFIGPLOT_H
#define CHIRPCONFIGPLOT_H

#include "zoompanplot.h"
#include <qwt6/qwt_plot_curve.h>
#include "chirpconfig.h"

class ChirpConfigPlot : public ZoomPanPlot
{
public:
    ChirpConfigPlot(QWidget *parent =0);
    ~ChirpConfigPlot();

public slots:
    void newChirp(const ChirpConfig cc);
    void buildContextMenu(QMouseEvent *me);

private:
    QwtPlotCurve *p_twtEnableCurve, *p_protectionCurve, *p_chirpCurve;
    QVector<QPointF> d_chirpData;

    void setCurveColor(QwtPlotCurve *c);


    // ZoomPanPlot interface
protected:
    void filterData();
};

#endif // CHIRPCONFIGPLOT_H
