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

    LifConfig getSettings(LifConfig c);

    void setDisplayOnly(bool b) { d_displayOnly = b; }

signals:
    void integralUpdate(double);
    void lifGateUpdated(int,int);
    void refGateUpdated(int,int);

public slots:
    void setNumAverages(int n);
    void newTrace(const LifTrace t);
    void traceProcessed(const LifTrace t);
    void buildContextMenu(QMouseEvent *me) override;
    void checkColors();
    void reset();
    void setIntegralText(double d);

    void changeLifGateRange();
    void changeRefGateRange();

    void clearPlot();

    void exportXY();

private:
    BlackchirpPlotCurve *p_lif, *p_ref;
    QwtPlotZoneItem *p_lifZone, *p_refZone;
    QwtPlotTextLabel *p_integralLabel;
    LifTrace d_currentTrace;
    int d_numAverages;
    bool d_resetNext, d_lifGateMode, d_refGateMode;
    QPair<int,int> d_lifZoneRange, d_refZoneRange;
    bool d_displayOnly;

    void updateLifZone();
    void updateRefZone();


    // ZoomPanPlot interface
protected:
    bool eventFilter(QObject *obj, QEvent *ev) override;
    virtual void replot() override;
};

#endif // LIFTRACEPLOT_H
