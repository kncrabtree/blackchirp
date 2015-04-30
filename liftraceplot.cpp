#include "liftraceplot.h"

#include <QSettings>
#include <QMenu>
#include <QColorDialog>
#include <QMouseEvent>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_zoneitem.h>
#include <qwt6/qwt_legend.h>
#include <qwt6/qwt_legend_label.h>


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
    p_lif->attach(this);

    p_ref = new QwtPlotCurve(QString("Ref"));
    p_ref->setPen(QPen(refColor));

    p_lifZone = new QwtPlotZoneItem();
    p_lifZone->setPen(QPen(lifColor));
    p_lifZone->setBrush(QBrush(zoneBrushColor));
    p_lifZone->setOrientation(Qt::Vertical);

    p_refZone = new QwtPlotZoneItem();
    p_refZone->setPen(QPen(lifColor));
    p_refZone->setBrush(QBrush(zoneBrushColor));
    p_refZone->setOrientation(Qt::Vertical);

    QwtLegend *leg = new QwtLegend(this);
    leg->contentsWidget()->installEventFilter(this);
    connect(leg,&QwtLegend::checked,this,&LifTracePlot::legendItemClicked);
    insertLegend(leg);

    initializeLabel(p_lif,true);

    connect(this,&LifTracePlot::plotRightClicked,this,&LifTracePlot::buildContextMenu);

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
    setAxisAutoScaleRange(QwtPlot::xBottom,lif.at(0).x()*1e6,lif.at(lif.size()-1).x()*1e6);

    p_lifZone->setInterval(lif.at(d_lifZoneRange.first).x()*1e6,lif.at(d_lifZoneRange.second).x()*1e6);
    if(p_lifZone->plot() != this)
        p_lifZone->attach(this);

    if(t.hasRefData())
    {
        QVector<QPointF> ref = t.refToXY();

        p_refZone->setInterval(ref.at(d_refZoneRange.first).x()*1e6,ref.at(d_refZoneRange.second).x()*1e6);
        if(p_ref->plot() != this)
        {
            p_ref->attach(this);
            initializeLabel(p_ref,true);
        }
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

    filterData();
    replot();
}

void LifTracePlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *m = contextMenu();

//    QAction *lifColorAction = m->addAction(QString("Change LIF Color..."));
//    connect(lifColorAction,&QAction::triggered,this,&LifTracePlot::changeLifColor);

//    QAction *refColorAction = m->addAction(QString("Change Ref Color..."));
//    connect(refColorAction,&QAction::triggered,this,&LifTracePlot::changeRefColor);

    m->popup(me->globalPos());
}

void LifTracePlot::changeLifColor()
{
    QColor currentColor = p_lif->pen().color();

    QColor newColor = QColorDialog::getColor(currentColor,this,QString("Choose New LIF Color"));
    if(!newColor.isValid())
        return;

    QSettings s;
    s.setValue(QString("lifTracePlot/lifColor"),newColor);

    p_lif->setPen(QPen(newColor));
    p_lifZone->setPen(QPen(newColor));

    replot();
}

void LifTracePlot::changeRefColor()
{
    QColor currentColor = p_ref->pen().color();

    QColor newColor = QColorDialog::getColor(currentColor,this,QString("Choose New Reference Color"));
    if(!newColor.isValid())
        return;

    QSettings s;
    s.setValue(QString("lifTracePlot/refColor"),newColor);

    p_ref->setPen(QPen(newColor));
    p_refZone->setPen(QPen(newColor));

    if(p_ref->plot() == this)
        replot();
}

void LifTracePlot::legendItemClicked(QVariant info, bool checked, int index)
{
    Q_UNUSED(index);

    QwtPlotCurve *c = dynamic_cast<QwtPlotCurve*>(infoToItem(info));
    if(c == nullptr)
        return;

    c->setVisible(checked);
    if(c == p_lif)
        p_lifZone->setVisible(checked);
    else if(c == p_ref)
        p_refZone->setVisible(checked);

    replot();
}

void LifTracePlot::initializeLabel(QwtPlotCurve *curve, bool isVisible)
{
    QwtLegendLabel* item = static_cast<QwtLegendLabel*>
            (static_cast<QwtLegend*>(legend())->legendWidget(itemToInfo(curve)));

    item->setItemMode(QwtLegendData::Checkable);
    item->setChecked(isVisible);
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

bool LifTracePlot::eventFilter(QObject *obj, QEvent *ev)
{
    if(ev->type() == QEvent::MouseButtonPress)
    {
        QwtLegend *l = static_cast<QwtLegend*>(legend());
        if(obj == l->contentsWidget())
        {
            QMouseEvent *me = dynamic_cast<QMouseEvent*>(ev);
            if(me != nullptr)
            {
                QwtLegendLabel *ll = dynamic_cast<QwtLegendLabel*>(l->contentsWidget()->childAt(me->pos()));
                if(ll != nullptr)
                {
                    QVariant item = l->itemInfo(ll);
                    QwtPlotCurve *c = dynamic_cast<QwtPlotCurve*>(infoToItem(item));
                    if(c == nullptr)
                    {
                        ev->ignore();
                        return true;
                    }

                    if(me->button() == Qt::RightButton)
                    {
                        if(c == p_lif)
                        {
                            changeLifColor();
                            ev->accept();
                            return true;
                        }
                        else if(c == p_ref)
                        {
                            changeRefColor();
                            ev->accept();
                            return true;
                        }
                    }
                }
            }
        }
    }

    return ZoomPanPlot::eventFilter(obj,ev);
}
