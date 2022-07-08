#ifndef LIFTRACEPLOT_H
#define LIFTRACEPLOT_H

#include <gui/plot/zoompanplot.h>

#include <modules/lif/data/lifconfig.h>
#include <modules/lif/data/liftrace.h>

class QwtPlotCurve;
class QwtPlotZoneItem;
class QwtPlotTextLabel;

namespace BC::Key {
static const QString lifTracePlot{"lifTracePlot"};
static const QString lifCurve{"lifCurve"};
static const QString refCurve{"lifRefCurve"};
}

class LifTracePlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    LifTracePlot(QWidget *parent = nullptr);
    ~LifTracePlot();

    void setLifGateRange(int begin, int end);
    void setRefGateRange(int begin, int end);

signals:
    void integralUpdate(double);
    void lifGateUpdated(int,int);
    void refGateUpdated(int,int);

public slots:
    void setNumAverages(int n);
    void processTrace(const LifTrace t);
    void setTrace(const LifTrace t);
    void checkColors();
    void reset();
    void setIntegralText(double d);

    void clearPlot();


private:
    BlackchirpPlotCurve *p_lif, *p_ref; ///TODO: convert to even spaced curves
    QwtPlotZoneItem *p_lifZone, *p_refZone;
    QwtPlotTextLabel *p_integralLabel;
    LifTrace d_currentTrace;
    int d_numAverages;
    bool d_resetNext;
    QPair<int,int> d_lifZoneRange, d_refZoneRange;

    void updateLifZone();
    void updateRefZone();


    // ZoomPanPlot interface
protected:
    virtual void replot() override;
};

#endif // LIFTRACEPLOT_H
