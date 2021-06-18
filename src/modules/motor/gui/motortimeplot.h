#ifndef MOTORTIMEPLOT_H
#define MOTORTIMEPLOT_H

#include <src/gui/plot/zoompanplot.h>

#include <qwt6/qwt_plot_curve.h>

#include <src/modules/motor/data/motorscan.h>

namespace BC::Key {
static const QString motorTimePlot("MotorTimePlot");
static const QString motorTimeCurve("MotorTimeCurve");
}

class MotorTimePlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    MotorTimePlot(QWidget *parent = nullptr);

    void prepareForScan(const MotorScan s);

public slots:
    void updateData(QVector<QPointF> d);

private:
    BlackchirpPlotCurve *p_curve;
};

#endif // MOTORTIMEPLOT_H
