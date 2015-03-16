#include "zoompanplot.h"
#include <QApplication>
#include <QMouseEvent>
#include <qwt6/qwt_scale_div.h>

ZoomPanPlot::ZoomPanPlot(QWidget *parent) : QwtPlot(parent)
{
    d_config.axisList.append(AxisConfig(QwtPlot::xBottom));
    d_config.axisList.append(AxisConfig(QwtPlot::xTop));
    d_config.axisList.append(AxisConfig(QwtPlot::yLeft));
    d_config.axisList.append(AxisConfig(QwtPlot::yRight));

    canvas()->installEventFilter(this);
}

ZoomPanPlot::~ZoomPanPlot()
{

}

bool ZoomPanPlot::isAutoScale()
{
    for(int i=0; i<d_config.axisList.size(); i++)
    {
        if(d_config.axisList.at(i).autoScale == false)
            return false;
    }

    return true;
}

void ZoomPanPlot::resetPlot()
{
    detachItems();
    for(int i=0;i<d_config.axisList.size();i++)
        setAxisAutoScaleRange(d_config.axisList.at(i).type,0.0,1.0);

    autoScale();
}

void ZoomPanPlot::autoScale()
{
    for(int i=0; i<d_config.axisList.size(); i++)
        d_config.axisList[i].autoScale = true;

    replot();
}

void ZoomPanPlot::setAxisAutoScaleRange(QwtPlot::Axis axis, double min, double max)
{
    int i = getAxisIndex(axis);
    d_config.axisList[i].min = min;
    d_config.axisList[i].max = max;
}

void ZoomPanPlot::setAxisAutoScaleMin(QwtPlot::Axis axis, double min)
{
    int i = getAxisIndex(axis);
    d_config.axisList[i].min = min;
}

void ZoomPanPlot::setAxisAutoScaleMax(QwtPlot::Axis axis, double max)
{
    int i = getAxisIndex(axis);
    d_config.axisList[i].max = max;
}

void ZoomPanPlot::expandAutoScaleRange(QwtPlot::Axis axis, double newValueMin, double newValueMax)
{
    int i = getAxisIndex(axis);
    setAxisAutoScaleRange(axis,qMin(newValueMin,d_config.axisList.at(i).min),qMax(newValueMax,d_config.axisList.at(i).max));
}

void ZoomPanPlot::replot()
{
    bool redrawXAxis = false;
    for(int i=0; i<d_config.axisList.size(); i++)
    {
        const AxisConfig c = d_config.axisList.at(i);
        if(c.autoScale)
        {
            setAxisScale(c.type,c.min,c.max);
            if(c.type == QwtPlot::xBottom || c.type == QwtPlot::xTop)
                redrawXAxis = true;
        }
    }

    if(redrawXAxis)
    {
        updateAxes();
        QApplication::sendPostedEvents(this,QEvent::LayoutRequest);
    }

    if(d_config.xDirty)
    {
        d_config.xDirty = false;
        filterData();
    }

    QwtPlot::replot();
}

void ZoomPanPlot::resizeEvent(QResizeEvent *ev)
{
    QwtPlot::resizeEvent(ev);

    d_config.xDirty = true;
    replot();
}

bool ZoomPanPlot::eventFilter(QObject *obj, QEvent *ev)
{
    if(obj == this->canvas())
    {
        if(ev->type() == QEvent::MouseButtonPress)
        {
            QMouseEvent *me = dynamic_cast<QMouseEvent*>(ev);
            if(me != nullptr && me->button() == Qt::MiddleButton)
            {
                if(!isAutoScale())
                {
                    d_config.panClickPos = me->pos();
                    d_config.panning = true;
                    emit panningStarted();
                    ev->accept();
                    return true;
                }
            }
        }
        else if(ev->type() == QEvent::MouseButtonRelease)
        {
            QMouseEvent *me = dynamic_cast<QMouseEvent*>(ev);
            if(me != nullptr)
            {
                if(d_config.panning && me->button() == Qt::MiddleButton)
                {
                    d_config.panning = false;
                    emit panningFinished();
                    ev->accept();
                    return true;
                }
                else if(me->button() == Qt::LeftButton && (me->modifiers() & Qt::ControlModifier))
                {
                    autoScale();
                    ev->accept();
                    return true;
                }
                else if(me->button() == Qt::RightButton)
                {
                    emit plotRightClicked(me);
                    ev->accept();
                    return true;
                }
            }
        }
        else if(ev->type() == QEvent::MouseMove)
        {
            pan(dynamic_cast<QMouseEvent*>(ev));
            ev->accept();
            return true;
        }
        else if(ev->type() == QEvent::Wheel)
        {
            zoom(dynamic_cast<QWheelEvent*>(ev));
            ev->accept();
            return true;
        }
    }

    return QwtPlot::eventFilter(obj,ev);
}

void ZoomPanPlot::pan(QMouseEvent *me)
{
    if(me == nullptr)
        return;

    QPoint delta = d_config.panClickPos - me->pos();
    d_config.xDirty = true;

    for(int i=0; i<d_config.axisList.size(); i++)
    {
        const AxisConfig c = d_config.axisList.at(i);

        double scaleMin = axisScaleDiv(c.type).lowerBound();
        double scaleMax = axisScaleDiv(c.type).upperBound();

        double d;
        if(c.type == QwtPlot::xBottom || c.type == QwtPlot::xTop)
            d = (scaleMax - scaleMin)/(double)canvas()->width()*delta.x();
        else
            d = -(scaleMax - scaleMin)/(double)canvas()->height()*delta.y();

        if(scaleMin + d < c.min)
            d = c.min - scaleMin;
        if(scaleMax + d > c.max)
            d = c.max - scaleMax;

        setAxisScale(c.type,scaleMin + d, scaleMax + d);
    }

    d_config.panClickPos = me->pos();

    replot();
}

void ZoomPanPlot::zoom(QWheelEvent *we)
{
    if(we == nullptr)
        return;

    //ctrl-wheel: lock both vertical
    //shift-wheel: lock horizontal
    //meta-wheel: lock right axis
    //alt-wheel: lock left axis
    bool lockHorizontal = (we->modifiers() & Qt::ShiftModifier) || (we->modifiers() & Qt::AltModifier) || (we->modifiers() & Qt::MetaModifier);
    bool lockLeft = (we->modifiers() & Qt::ControlModifier) || (we->modifiers() & Qt::AltModifier);
    bool lockRight = (we->modifiers() & Qt::ControlModifier) || (we->modifiers() & Qt::MetaModifier);

    //one step, which is 15 degrees, will zoom 10%
    //the delta function is in units of 1/8th a degree
    int numSteps = we->delta()/8/15;

    for(int i=0; i<d_config.axisList.size(); i++)
    {
        const AxisConfig c = d_config.axisList.at(i);

        if((c.type == QwtPlot::xBottom || c.type == QwtPlot::xTop) && lockHorizontal)
            continue;
        if(c.type == QwtPlot::yLeft && lockLeft)
            continue;
        if(c.type == QwtPlot::yRight && lockRight)
            continue;

        double scaleMin = axisScaleDiv(c.type).lowerBound();
        double scaleMax = axisScaleDiv(c.type).upperBound();
        double factor = c.zoomFactor;
        int mousePosInt;
        if(c.type == QwtPlot::xBottom || c.type == QwtPlot::xTop)
        {
            mousePosInt = we->pos().x();
            d_config.xDirty = true;
        }
        else
            mousePosInt = we->pos().y();

        double mousePos = qBound(scaleMin,canvasMap(c.type).invTransform(mousePosInt),scaleMax);

        scaleMin += qAbs(mousePos-scaleMin)*factor*(double)numSteps;
        scaleMax -= qAbs(mousePos-scaleMax)*factor*(double)numSteps;

        scaleMin = qMax(scaleMin,c.min);
        scaleMax = qMin(scaleMax,c.max);

        if(scaleMin <= c.min && scaleMax >= c.max)
            d_config.axisList[i].autoScale = true;
        else
        {
            d_config.axisList[i].autoScale = false;
            setAxisScale(c.type,scaleMin,scaleMax);
        }
    }

    replot();
}

int ZoomPanPlot::getAxisIndex(QwtPlot::Axis a)
{
    int i;
    switch (a) {
    case QwtPlot::xBottom:
        i=0;
        break;
    case QwtPlot::xTop:
        i=1;
        break;
    case QwtPlot::yLeft:
        i=2;
        break;
    case QwtPlot::yRight:
        i=3;
        break;
    default:
        i = 0;
        break;
    }

    return i;
}

