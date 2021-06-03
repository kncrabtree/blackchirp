#ifndef CHIRPCONFIGPLOT_H
#define CHIRPCONFIGPLOT_H

#include "zoompanplot.h"

class ChirpConfig;
class QwtPlotCurve;


class ChirpConfigPlot : public ZoomPanPlot
{
public:
    ChirpConfigPlot(QWidget *parent =0);
    ~ChirpConfigPlot();

public slots:
    void newChirp(const ChirpConfig cc);
    void buildContextMenu(QMouseEvent *me);

private:
    QwtPlotCurve *p_ampEnableCurve, *p_protectionCurve, *p_chirpCurve;
    QVector<QPointF> d_chirpData;

    void setCurveColor(QwtPlotCurve *c);


    // ZoomPanPlot interface
protected:
    void filterData();
};

#endif // CHIRPCONFIGPLOT_H
