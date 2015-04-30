#ifndef LIFTRACEPLOT_H
#define LIFTRACEPLOT_H

#include "zoompanplot.h"

#include "liftrace.h"

class QwtPlotCurve;
class QwtPlotZoneItem;

class LifTracePlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    LifTracePlot(QWidget *parent = nullptr);
    ~LifTracePlot();

signals:
    void colorChanged();
    void integralUpdate(double);

public slots:
    void setNumAverages(int n);
    void newTrace(const LifTrace t);
    void traceProcessed(const LifTrace t);
    void buildContextMenu(QMouseEvent *me);
    void changeLifColor();
    void changeRefColor();
    void legendItemClicked(QVariant info, bool checked, int index);
    void reset();

    void changeLifGateRange();
    void changeRefGateRange();

private:
    QwtPlotCurve *p_lif, *p_ref;
    QwtPlotZoneItem *p_lifZone, *p_refZone;
    LifTrace d_currentTrace;
    int d_numAverages;
    bool d_resetNext, d_lifGateMode, d_refGateMode;
    QPair<int,int> d_lifZoneRange, d_refZoneRange;

    void initializeLabel(QwtPlotCurve *curve, bool isVisible);
    void updateLifZone();
    void updateRefZone();


    // ZoomPanPlot interface
protected:
    void filterData();
    bool eventFilter(QObject *obj, QEvent *ev);
};

#endif // LIFTRACEPLOT_H
