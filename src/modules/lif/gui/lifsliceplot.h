#ifndef LIFSLICEPLOT_H
#define LIFSLICEPLOT_H

#include <src/gui/plot/zoompanplot.h>

class QwtPlotCurve;
class QwtPlotTextLabel;

namespace BC::Key {
static const QString lifSlicePlot("lifSlicePlot");
static const QString lifSliceCurve("LifSliceCurve");
}

/// \todo This needs to take an argument to differentiate time vs freq slice plots
class LifSlicePlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    LifSlicePlot(QWidget *parent = nullptr);
    ~LifSlicePlot();

    void setXAxisTitle(QString title);

    void prepareForExperiment(double xMin, double xMax);
    void setData(const QVector<QPointF> d);
    void setPlotTitle(QString text);

public slots:
    void exportXY();

    // ZoomPanPlot interface
protected:
    void filterData() override;

    BlackchirpPlotCurve *p_curve;
    QVector<QPointF> d_currentData;

    // ZoomPanPlot interface
protected:
    QMenu *contextMenu() override;
};

#endif // LIFSLICEPLOT_H
