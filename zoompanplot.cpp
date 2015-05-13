#include "zoompanplot.h"

#include <QApplication>
#include <QMouseEvent>
#include <QWidgetAction>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QSettings>
#include <QMenu>

#include <qwt6/qwt_scale_div.h>

ZoomPanPlot::ZoomPanPlot(QString name, QWidget *parent) : QwtPlot(parent)
{
    d_config.axisList.append(AxisConfig(QwtPlot::xBottom,QString("Bottom")));
    d_config.axisList.append(AxisConfig(QwtPlot::xTop,QString("Top")));
    d_config.axisList.append(AxisConfig(QwtPlot::yLeft,QString("Left")));
    d_config.axisList.append(AxisConfig(QwtPlot::yRight,QString("Right")));

    setName(name);

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

    d_config.xDirty = true;
    d_config.panning = false;

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

void ZoomPanPlot::setXRanges(const QwtScaleDiv &bottom, const QwtScaleDiv &top)
{
    setAxisScale(QwtPlot::xBottom,bottom.lowerBound(),bottom.upperBound());
    setAxisScale(QwtPlot::xTop,top.lowerBound(),top.upperBound());

    for(int i=0; i<d_config.axisList.size(); i++)
    {
        if(d_config.axisList.at(i).type == QwtPlot::xBottom || d_config.axisList.at(i).type == QwtPlot::xTop)
            d_config.axisList[i].autoScale = false;
    }

    d_config.xDirty = true;
    replot();
}

void ZoomPanPlot::setName(QString name)
{
    d_name = name;

    QSettings s;
    for(int i=0; i<d_config.axisList.size(); i++)
        d_config.axisList[i].zoomFactor = s.value(QString("zoomFactors/%1/%2").arg(d_name)
                                                  .arg(QVariant::fromValue(d_config.axisList.at(i).type).toString()),0.1).toDouble();
}

void ZoomPanPlot::replot()
{

    //figure out which axes to show
    QwtPlotItemList l = itemList();
    bool bottom = false, top = false, left = false, right = false;
    for(int i=0; i<l.size(); i++)
    {
        if(l.at(i)->yAxis() == QwtPlot::yLeft)
            left = true;
        if(l.at(i)->yAxis() == QwtPlot::yRight)
            right = true;
        if(l.at(i)->xAxis() == QwtPlot::xBottom)
            bottom = true;
        if(l.at(i)->xAxis() == QwtPlot::xTop)
            top = true;
    }

    if(!d_config.axisList.at(getAxisIndex(QwtPlot::yLeft)).override)
        enableAxis(QwtPlot::yLeft,left);
    if(!d_config.axisList.at(getAxisIndex(QwtPlot::yRight)).override)
        enableAxis(QwtPlot::yRight,right);
    if(!d_config.axisList.at(getAxisIndex(QwtPlot::xTop)).override)
        enableAxis(QwtPlot::xTop,top);
    if(!d_config.axisList.at(getAxisIndex(QwtPlot::xBottom)).override)
        enableAxis(QwtPlot::xBottom,bottom);

    if(!bottom || d_config.axisList.at(getAxisIndex(QwtPlot::xBottom)).override)
        d_config.axisList[getAxisIndex(QwtPlot::xBottom)].autoScale = true;
    if(!top || d_config.axisList.at(getAxisIndex(QwtPlot::xTop)).override)
        d_config.axisList[getAxisIndex(QwtPlot::xTop)].autoScale = true;
    if(!left || d_config.axisList.at(getAxisIndex(QwtPlot::yLeft)).override)
        d_config.axisList[getAxisIndex(QwtPlot::yLeft)].autoScale = true;
    if(!right || d_config.axisList.at(getAxisIndex(QwtPlot::yRight)).override)
        d_config.axisList[getAxisIndex(QwtPlot::yRight)].autoScale = true;


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
        d_config.xDirty = true;
    }

    if(d_config.xDirty)
    {
        d_config.xDirty = false;
        filterData();
    }

    QwtPlot::replot();
}

void ZoomPanPlot::setZoomFactor(QwtPlot::Axis a, double v)
{
    int i = getAxisIndex(a);
    d_config.axisList[i].zoomFactor = v;

    QSettings s;
    s.setValue(QString("zoomFactors/%1/%2").arg(d_name)
                       .arg(QVariant(d_config.axisList.at(i).type).toString()),v);
    s.sync();
}

void ZoomPanPlot::setAxisOverride(QwtPlot::Axis axis, bool override)
{
    d_config.axisList[getAxisIndex(axis)].override = override;
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
            if(d_config.panning)
            {
                pan(dynamic_cast<QMouseEvent*>(ev));
                ev->accept();
                return true;
            }
        }
        else if(ev->type() == QEvent::Wheel)
        {
            if(!itemList().isEmpty())
            {
                zoom(dynamic_cast<QWheelEvent*>(ev));
                ev->accept();
                return true;
            }
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
        if(c.override)
            continue;

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
    int numSteps = we->angleDelta().y()/8/15;

    for(int i=0; i<d_config.axisList.size(); i++)
    {
        const AxisConfig c = d_config.axisList.at(i);
        if(c.override)
            continue;

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

QMenu *ZoomPanPlot::contextMenu()
{
    QMenu *menu = new QMenu();
    menu->setAttribute(Qt::WA_DeleteOnClose);

    QAction *asAction = menu->addAction(QString("Autoscale"));
    connect(asAction,&QAction::triggered,this,&ZoomPanPlot::autoScale);

    if(!itemList().isEmpty())
    {
        QMenu *zoomMenu = menu->addMenu(QString("Wheel zoom factor"));
        QWidgetAction *wa = new QWidgetAction(zoomMenu);
        QWidget *w = new QWidget(zoomMenu);
        QFormLayout *fl = new QFormLayout(w);

        for(int i=0; i<d_config.axisList.size(); i++)
        {
            const AxisConfig c = d_config.axisList.at(i);
            if(!axisEnabled(c.type))
                continue;

            QDoubleSpinBox *box = new QDoubleSpinBox();
            box->setMinimum(0.001);
            box->setMaximum(0.5);
            box->setDecimals(3);
            box->setValue(c.zoomFactor);
            box->setSingleStep(0.005);
            box->setKeyboardTracking(false);
            connect(box,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),
                    this,[=](double val){ setZoomFactor(c.type,val); });
            fl->addRow(c.name,box);
        }

        w->setLayout(fl);
        wa->setDefaultWidget(w);
        zoomMenu->addAction(wa);
    }

    return menu;

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

