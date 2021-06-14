#ifndef CHIRPCONFIGPLOT_H
#define CHIRPCONFIGPLOT_H

#include <src/gui/plot/zoompanplot.h>

class ChirpConfig;
class QwtPlotCurve;

namespace BC::Key {
static const QString chirpPlot("ChirpConfigPlot");
static const QString chirpColor("chirpColor");
static const QString ampColor("ampEnableColor");
static const QString protectionColor("protectionColor");
}

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


    // ZoomPanPlot interface
protected:
    void filterData();
};

#endif // CHIRPCONFIGPLOT_H
