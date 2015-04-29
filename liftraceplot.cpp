#include "liftraceplot.h"

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_zoneitem.h>
#include <QSettings>

LifTracePlot::LifTracePlot(QWidget *parent) :
    ZoomPanPlot(QString("lifTrace"),parent), d_numAverages(1), d_resetNext(true)
{
    QSettings s;

    QColor lifColor = s.value(QString("lifTracePlot/lifColor"),QPalette().color(QPalette::Text)).value<QColor>();
    QColor refColor = s.value(QString("lifTracePlot/refColor"),QPalette().color(QPalette::Text)).value<QColor>();
    QColor zoneBrushColor = QColor(Qt::black);
    zoneBrushColor.setAlpha(50);

    p_lif = new QwtPlotCurve(QString("LIF"));
    p_lif->setPen(QPen(lifColor));
    p_lif->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_lif->setVisible(false);
    p_lif->attach(this);

    p_ref = new QwtPlotCurve(QString("Ref"));
    p_ref->setPen(QPen(refColor));
    p_ref->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_ref->setVisible(false);
    p_ref->attach(this);

    p_lifZone = new QwtPlotZoneItem();
    p_lifZone->setPen(QPen(lifColor));
    p_lifZone->setBrush(QBrush(zoneBrushColor));
    p_lifZone->setVisible(false);
    p_lifZone->setOrientation(Qt::Vertical);
    p_lifZone->attach(this);

    p_refZone = new QwtPlotZoneItem();
    p_refZone->setPen(QPen(lifColor));
    p_refZone->setBrush(QBrush(zoneBrushColor));
    p_refZone->setVisible(false);
    p_refZone->setOrientation(Qt::Vertical);
    p_refZone->attach(this);

}

LifTracePlot::~LifTracePlot()
{

}

void LifTracePlot::setNumAverages(int n)
{
    d_numAverages = n;
}

void LifTracePlot::newTrace(const LifTrace t)
{
    if(t.size() == 0)
        return;

//    if(d_resetNext)
    traceProcessed(t);

}

void LifTracePlot::traceProcessed(const LifTrace t)
{
    d_currentTrace = t;

    if(d_lifZoneRange.first < 0 || d_lifZoneRange.first >= t.size())
        d_lifZoneRange.first = 0;
    if(d_lifZoneRange.second < d_lifZoneRange.first || d_lifZoneRange.second >= t.size())
        d_lifZoneRange.second = t.size()-1;
    if(d_refZoneRange.first < 0 || d_refZoneRange.first >= t.size())
        d_refZoneRange.first = 0;
    if(d_refZoneRange.second < d_refZoneRange.first || d_refZoneRange.second >= t.size())
        d_refZoneRange.second = t.size()-1;

    QVector<QPointF> lif = t.lifToXY();
    setAxisAutoScaleRange(QwtPlot::xBottom,lif.at(0).x(),lif.at(lif.size()-1).x());

    p_lifZone->setInterval(lif.at(d_lifZoneRange.first).x(),lif.at(d_lifZoneRange.second).x());
    if(!p_lif->isVisible())
        p_lif->setVisible(true);
    if(!p_lifZone->isVisible())
        p_lif->setVisible(true);

    if(t.hasRefData())
    {
        QVector<QPointF> ref = t.refToXY();

        p_refZone->setInterval(lif.at(d_refZoneRange.first).x(),lif.at(d_refZoneRange.second).x());
        if(!p_ref->isVisible())
            p_ref->setVisible(true);
        if(!p_refZone->isVisible())
            p_ref->setVisible(true);
    }
    else
    {
        p_ref->setVisible(false);
        p_refZone->setVisible(false);
    }

    filterData();
    replot();
}



void LifTracePlot::filterData()
{
    if(d_currentTrace.size() < 2)
        return;

    QVector<QPointF> lifData = d_currentTrace.lifToXY();
    QVector<QPointF> refData;
    if(d_currentTrace.hasRefData())
        refData = d_currentTrace.refToXY();


    double firstPixel = 0.0;
    double lastPixel = canvas()->width();
    QwtScaleMap map = canvasMap(QwtPlot::xBottom);

    QVector<QPointF> lifFiltered, refFiltered;
    double yMin = 0.0, yMax = 0.0;

    //find first data point that is in the range of the plot
    //note: x data for lif and ref are assumed to be the same!
    int dataIndex = 0;
    while(dataIndex+1 < lifData.size() && map.transform(lifData.at(dataIndex).x()*1e6) < firstPixel)
        dataIndex++;

    //add the previous point to the filtered array
    //this will make sure the curve always goes to the edge of the plot
    if(dataIndex-1 >= 0)
    {
        lifFiltered.append(lifData.at(dataIndex-1));
        if(d_currentTrace.hasRefData())
            refFiltered.append(refData.at(dataIndex-1));
    }

    //at this point, dataIndex is at the first point within the range of the plot. loop over pixels, compressing data
    for(double pixel = firstPixel; pixel<lastPixel; pixel+=1.0)
    {
        double lifMin = lifData.at(dataIndex).y(), lifMax = lifMin;
        double refMin = 0.0, refMax = 0.0;
        if(d_currentTrace.hasRefData())
        {
            refMin = refData.at(dataIndex).y();
            refMax = refMin;
        }

        int lifMinIndex = dataIndex, lifMaxIndex = dataIndex, refMinIndex = dataIndex, refMaxIndex = dataIndex;
        int numPnts = 0;

        while(dataIndex+1 < lifData.size() && map.transform(lifData.at(dataIndex).x()*1e6) < pixel+1.0)
        {
            if(lifData.at(dataIndex).y() < lifMin)
            {
                lifMin = lifData.at(dataIndex).y();
                lifMinIndex = dataIndex;
            }
            if(lifData.at(dataIndex).y() > lifMax)
            {
                lifMax = lifData.at(dataIndex).y();
                lifMaxIndex = dataIndex;
            }
            if(d_currentTrace.hasRefData())
            {
                if(refData.at(dataIndex).y() < refMin)
                {
                    refMin = refData.at(dataIndex).y();
                    refMinIndex = dataIndex;
                }
                if(refData.at(dataIndex).y() > refMax)
                {
                    refMax = refData.at(dataIndex).y();
                    refMaxIndex = dataIndex;
                }
            }

            dataIndex++;
            numPnts++;
        }
        if(lifFiltered.isEmpty())
        {
            yMin = lifMin;
            yMax = lifMax;
            if(d_currentTrace.hasRefData())
            {
                yMin = qMin(lifMin,refMin);
                yMax = qMax(lifMax,refMax);
            }
        }
        else
        {
            yMin = qMin(lifMin,yMin);
            yMax = qMax(lifMax,yMax);
            if(d_currentTrace.hasRefData())
            {
                yMin = qMin(yMin,refMin);
                yMax = qMax(yMax,refMax);
            }
        }
        if(numPnts == 1)
        {
            lifFiltered.append(QPointF(lifData.at(dataIndex-1).x()*1e6,lifData.at(dataIndex-1).y()));
            if(d_currentTrace.hasRefData())
                refFiltered.append(QPointF(lifData.at(dataIndex-1).x()*1e6,refData.at(dataIndex-1).y()));
        }
        else if (numPnts > 1)
        {
            QPointF first(map.invTransform(pixel),lifData.at(lifMinIndex).y());
            QPointF second(map.invTransform(pixel),lifData.at(lifMaxIndex).y());
            lifFiltered.append(first);
            lifFiltered.append(second);
            if(d_currentTrace.hasRefData())
            {
                QPointF refFirst(map.invTransform(pixel),refData.at(refMinIndex).y());
                QPointF refSecond(map.invTransform(pixel),refData.at(refMaxIndex).y());
                refFiltered.append(refFirst);
                refFiltered.append(refSecond);
            }
        }
    }

    if(dataIndex < lifData.size())
    {
        QPointF p = lifData.at(dataIndex);
        p.setX(p.x()*1e6);
        lifFiltered.append(p);
        if(d_currentTrace.hasRefData())
        {
            p = refData.at(dataIndex);
            p.setX(p.x()*1e6);
            refFiltered.append(p);
        }
    }

    setAxisAutoScaleRange(QwtPlot::yLeft,yMin,yMax);
    //assign data to curve object
    p_lif->setSamples(lifFiltered);
    if(d_currentTrace.hasRefData())
        p_ref->setSamples(refFiltered);

}
