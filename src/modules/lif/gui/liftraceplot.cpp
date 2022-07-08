#include "liftraceplot.h"

#include <QApplication>
#include <QMenu>
#include <QColorDialog>
#include <QMouseEvent>
#include <QWidgetAction>
#include <QSpinBox>
#include <QFormLayout>
#include <QFileDialog>
#include <QMessageBox>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_zoneitem.h>
#include <qwt6/qwt_legend.h>
#include <qwt6/qwt_legend_label.h>
#include <qwt6/qwt_plot_textlabel.h>

#include <gui/plot/blackchirpplotcurve.h>


LifTracePlot::LifTracePlot(QWidget *parent) :
    ZoomPanPlot(BC::Key::lifTracePlot,parent), d_resetNext(true)
{

    setPlotAxisTitle(QwtPlot::xBottom,QString("Time (ns)"));
    setPlotAxisTitle(QwtPlot::yLeft,QString("LIF (V)"));

    p_integralLabel = new QwtPlotTextLabel();
    p_integralLabel->setZ(10.0);
    p_integralLabel->attach(this);
    p_integralLabel->setItemAttribute(QwtPlotItem::AutoScale,false);

    p_lif = new BlackchirpPlotCurve(BC::Key::lifCurve,"LIF");
    p_lif->setZ(1.0);

    p_ref = new BlackchirpPlotCurve(BC::Key::refCurve,"Ref");
    p_ref->setZ(1.0);

    p_lifZone = new QwtPlotZoneItem();
    p_lifZone->setOrientation(Qt::Vertical);
    p_lifZone->setZ(2.0);
    p_lifZone->setItemAttribute(QwtPlotItem::AutoScale,false);



    p_refZone = new QwtPlotZoneItem();
    p_refZone->setOrientation(Qt::Vertical);
    p_refZone->setZ(2.0);
    p_refZone->setItemAttribute(QwtPlotItem::AutoScale,false);


    insertLegend( new QwtLegend(this),QwtPlot::BottomLegend);

    connect(this,&LifTracePlot::integralUpdate,this,&LifTracePlot::setIntegralText);
}

LifTracePlot::~LifTracePlot()
{
    p_lif->detach();
    delete p_lif;

    p_ref->detach();
    delete p_ref;

    p_lifZone->detach();
    delete p_lifZone;

    p_refZone->detach();
    delete p_refZone;

    p_integralLabel->detach();
    delete p_integralLabel;
}

void LifTracePlot::setLifGateStart(int n)
{
    d_procSettings.lifGateStart = n;
    updateLifZone();
}

void LifTracePlot::setLifGateEnd(int n)
{
    d_procSettings.lifGateEnd = n;
    updateLifZone();
}

void LifTracePlot::setRefGateStart(int n)
{
    d_procSettings.refGateStart = n;
    updateRefZone();
}

void LifTracePlot::setRefGateEnd(int n)
{
    d_procSettings.refGateEnd = n;
    updateRefZone();
}

void LifTracePlot::setNumAverages(int n)
{
    d_numAverages = n;
}

void LifTracePlot::setAllProcSettings(const LifTrace::LifProcSettings &s)
{
    d_procSettings = s;
    replot();
}

void LifTracePlot::processTrace(const LifTrace t)
{
    if(t.size() == 0)
        return;

    if(d_resetNext || d_currentTrace.size() == 0)
    {
        d_resetNext = false;
        setTrace(t);
    }
    else
    {
        d_currentTrace.rollAvg(t,d_numAverages);
        setTrace(d_currentTrace);
    }

}

void LifTracePlot::setTrace(const LifTrace t)
{
    d_currentTrace = t;

    ///TODO: update when curves are changed to fixed spacing
    p_lif->setCurveData(t.lifToXY(d_procSettings));
    p_ref->setCurveData(t.refToXY(d_procSettings));


    if(p_lif->plot() != this)
        p_lif->attach(this);

    if(p_lifZone->plot() != this)
        p_lifZone->attach(this);

    if(t.hasRefData())
    {
        if(p_ref->plot() != this)
            p_ref->attach(this);

        if(p_refZone->plot() != this)
            p_refZone->attach(this);
    }
    else
    {
        if(p_ref->plot() == this)
            p_ref->detach();
        if(p_refZone->plot() == this)
            p_refZone->detach();
    }

    replot();
}

void LifTracePlot::checkColors()
{

    p_lif->updateFromSettings();
    auto lc = p_lif->pen().color();
    p_lifZone->setPen(QPen(lc,2.0));
    lc.setAlpha(75);
    p_lifZone->setBrush(QBrush(lc));

    p_ref->updateFromSettings();
    auto rc = p_ref->pen().color();
    p_refZone->setPen(QPen(rc,2.0));
    rc.setAlpha(75);
    p_refZone->setBrush(QBrush(rc));

    ZoomPanPlot::replot();

}

void LifTracePlot::reset()
{
    d_resetNext = true;
}

void LifTracePlot::setIntegralText(double d)
{
    QwtText t;
    QString text = QString("%1\n%2 Avgs").arg(QString::number(d,'e',3)).arg(d_currentTrace.shots());

    t.setRenderFlags(Qt::AlignRight | Qt::AlignTop);
    t.setBackgroundBrush(QBrush(QPalette().color(QPalette::Window)));
    QColor border = QPalette().color(QPalette::Text);
    border.setAlpha(0);
    t.setBorderPen(QPen(border));
    t.setColor(QPalette().color(QPalette::Text));

//    QFont f(QString("monospace"));
//    t.setFont(f);

    t.setText(text);
    p_integralLabel->setText(t);
}

void LifTracePlot::clearPlot()
{
    p_lif->detach();
    p_lifZone->detach();
    p_ref->detach();
    p_refZone->detach();
//    p_integralLabel->setText(QString(""));

    d_currentTrace = LifTrace();

    replot();
}

void LifTracePlot::updateLifZone()
{
    double x1 = static_cast<double>(d_procSettings.lifGateStart)*d_currentTrace.xSpacingns();
    double x2 = static_cast<double>(d_procSettings.lifGateEnd)*d_currentTrace.xSpacingns();
    p_lifZone->setInterval(x1,x2);
}

void LifTracePlot::updateRefZone()
{
    double x1 = static_cast<double>(d_procSettings.refGateStart)*d_currentTrace.xSpacingns();
    double x2 = static_cast<double>(d_procSettings.refGateEnd)*d_currentTrace.xSpacingns();
    p_refZone->setInterval(x1,x2);
}

void LifTracePlot::replot()
{
    //this function calls ZoomPanPlot::replot()
    updateLifZone();
    updateRefZone();
    emit integralUpdate(d_currentTrace.integrate(d_procSettings));

    checkColors();
}
