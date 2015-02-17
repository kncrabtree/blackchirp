#include "ftplot.h"
#include <QFont>
#include <qwt6/qwt_plot_canvas.h>
#include <qwt6/qwt_picker_machine.h>
#include <qwt6/qwt_scale_widget.h>
#include <qwt6/qwt_scale_div.h>
#include <QMouseEvent>
#include <QEvent>
#include <QSettings>
#include <QMenu>
#include <QAction>
#include <QActionGroup>
#include <QColorDialog>
#include <QApplication>

FtPlot::FtPlot(QWidget *parent) :
    QwtPlot(parent), d_autoScaleXY(QPair<bool,bool>(true,true)), d_autoScaleXRange(QPair<double,double>(-1.0,-1.0)), d_autoScaleYRange(QPair<double,double>(-1.0,-1.0)),
    d_processing(false), d_replotWhenDone(false)
{
    //make axis label font smaller
    this->setAxisFont(QwtPlot::xBottom,QFont(tr("sans-serif"),8));
    this->setAxisFont(QwtPlot::yLeft,QFont(tr("sans-serif"),8));

    //build axis titles with small font
    QwtText blabel(tr("Frequency (MHz)"));
    blabel.setFont(QFont(tr("sans-serif"),8));
    this->setAxisTitle(QwtPlot::xBottom,blabel);

    QwtText llabel(tr("FT"));
    llabel.setFont(QFont(tr("sans-serif"),8));
    this->setAxisTitle(QwtPlot::yLeft,llabel);

    //intialize panning to false. This will be enabled when the middle mouse button is pressed
    d_panning = false;

    QSettings s;

    //build and configure curve object
    p_curveData = new QwtPlotCurve();
    QPalette pal;
    QPen p;
    p.setColor(s.value(tr("ftcolor"),pal.color(QPalette::BrightText)).value<QColor>());
    p.setWidth(1);
    p_curveData->setPen(p);
    p_curveData->attach(this);

    QwtPlotPicker *picker = new QwtPlotPicker(this->canvas());
    picker->setAxis(QwtPlot::xBottom,QwtPlot::yLeft);
    picker->setStateMachine(new QwtPickerClickPointMachine);
    picker->setMousePattern(QwtEventPattern::MouseSelect1,Qt::RightButton);
    picker->setTrackerMode(QwtPicker::AlwaysOn);
    picker->setTrackerPen(QPen(QPalette().color(QPalette::Text)));
    picker->setEnabled(true);

    p_plotGrid = new QwtPlotGrid();
    p_plotGrid->enableX(true);
    p_plotGrid->enableXMin(true);
    p_plotGrid->enableY(true);
    p_plotGrid->enableYMin(true);

    p.setColor(s.value(tr("gridcolor"),pal.color(QPalette::Light)).value<QColor>());
    p.setStyle(Qt::DashLine);
    p_plotGrid->setMajorPen(p);



    p.setStyle(Qt::DotLine);
    p_plotGrid->setMinorPen(p);

    p_plotGrid->attach(this);

    this->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this,&FtPlot::customContextMenuRequested,this,&FtPlot::buildContextMenu);

    p_ftw = new FtWorker();
    //make signal/slot connections
    connect(p_ftw,&FtWorker::ftDone,this,&FtPlot::ftDone);
    connect(p_ftw,&FtWorker::fidDone,this,&FtPlot::fidDone);
    p_ftThread = new QThread(this);
    connect(p_ftThread,&QThread::finished,p_ftw,&FtWorker::deleteLater);
    p_ftw->moveToThread(p_ftThread);
    p_ftThread->start();

    setAxisAutoScale(QwtPlot::xBottom,false);

}

FtPlot::~FtPlot()
{
    p_ftThread->quit();
    p_ftThread->wait();
}

void FtPlot::newFid(const Fid f)
{
    d_currentFid = f;

    if(d_processing)
        d_replotWhenDone = true;
    else
        updatePlot();
}

void FtPlot::ftDone(QVector<QPointF> ft, double max)
{
    d_processing = false;
    d_currentFt = ft;
    d_autoScaleXRange.first = d_currentFt.at(0).x();
    d_autoScaleXRange.second = d_currentFt.at(d_currentFt.size()-1).x();
    d_autoScaleYRange.first = 0.0;
    d_autoScaleYRange.second = max;
    replot();

    if(d_replotWhenDone)
        updatePlot();
}

void FtPlot::filterData()
{
    if(d_currentFt.size() < 2)
        return;

    double firstPixel = 0.0;
    double lastPixel = canvas()->width();
    QwtScaleMap map = canvasMap(QwtPlot::xBottom);
    double scaleMin = map.invTransform(firstPixel);
//    double scaleMax = map.invTransform(lastPixel);

    QVector<QPointF> filtered;

    //find first data point that is in the range of the plot
    int dataIndex = 0;
    while(dataIndex+1 < d_currentFt.size() && map.transform(d_currentFt.at(dataIndex).x()) < firstPixel)
        dataIndex++;

    //add the previous point to the filtered array
    //this will make sure the curve always goes to the edge of the plot
    if(dataIndex-1 >= 0)
        filtered.append(d_currentFt.at(dataIndex-1));

    //at this point, dataIndex is at the first point within the range of the plot. loop over pixels, compressing data
    for(double pixel = firstPixel; pixel<lastPixel; pixel+=1.0)
    {
        double min = d_currentFt.at(dataIndex).y(), max = min;
        int minIndex = dataIndex, maxIndex = dataIndex;
//        double upperLimit = map.invTransform(pixel+1.0);
        int numPnts = 0;
        while(dataIndex+1 < d_currentFt.size() && map.transform(d_currentFt.at(dataIndex).x()) < pixel+1.0)
        {
            if(d_currentFt.at(dataIndex).y() < min)
            {
                min = d_currentFt.at(dataIndex).y();
                minIndex = dataIndex;
            }
            if(d_currentFt.at(dataIndex).y() > max)
            {
                max = d_currentFt.at(dataIndex).y();
                maxIndex = dataIndex;
            }
            dataIndex++;
            numPnts++;
        }
        if(numPnts == 1)
            filtered.append(d_currentFt.at(dataIndex-1));
        else if (numPnts > 1)
        {
            QPointF first(map.invTransform(pixel),d_currentFt.at(minIndex).y());
            QPointF second(map.invTransform(pixel+0.99),d_currentFt.at(maxIndex).y());
            filtered.append(first);
            filtered.append(second);
        }
    }

    if(dataIndex < d_currentFt.size())
        filtered.append(d_currentFt.at(dataIndex));

    //assign data to curve object
    p_curveData->setSamples(filtered);
}

void FtPlot::buildContextMenu(QPoint p)
{
    QMenu *m = new QMenu(this);

    QAction *ftColorAction = new QAction(QString("Change FT Color..."),m);
    connect(ftColorAction,&QAction::triggered,this,&FtPlot::ftColorSlot);
    m->addAction(ftColorAction);

    m->addAction(QString("Change Grid Color..."),this,SLOT(gridColorSlot()));

    m->popup(this->mapToGlobal(p));
}

void FtPlot::ftColorSlot()
{
    QColor c = getColor(p_curveData->pen().color());
    if(!c.isValid())
        return;

    QSettings s;
    s.setValue(tr("ftcolor"),c);
//    s.sync();

    QPen p(c);
    p.setWidth(1);
    p_curveData->setPen(p);
    replot(false);

}

void FtPlot::gridColorSlot()
{
    QColor c = getColor(p_plotGrid->majorPen().color());
    if(!c.isValid())
        return;

    QSettings s;
    s.setValue(tr("gridcolor"),c);
//    s.sync();

    QPen p(c);
    p.setStyle(Qt::DashLine);
    p_plotGrid->setMajorPen(p);

    p.setStyle(Qt::DotLine);
    p_plotGrid->setMinorPen(p);
    replot(false);
}

QColor FtPlot::getColor(QColor startingColor)
{
    return QColorDialog::getColor(startingColor,this,tr("Select Color"));
}

void FtPlot::pan(QMouseEvent *me)
{
    if(!me)
        return;

    double xScaleMin = axisScaleDiv(QwtPlot::xBottom).lowerBound();
    double xScaleMax = axisScaleDiv(QwtPlot::xBottom).upperBound();
    double yScaleMin = axisScaleDiv(QwtPlot::yLeft).lowerBound();
    double yScaleMax = axisScaleDiv(QwtPlot::yLeft).upperBound();

    QPoint delta = d_panClickPos - me->pos();
    double dx = (xScaleMax-xScaleMin)/(double)canvas()->width()*delta.x();
    double dy = (yScaleMax-yScaleMin)/(double)canvas()->height()*-delta.y();


    if(xScaleMin + dx < d_autoScaleXRange.first)
        dx = d_autoScaleXRange.first - xScaleMin;
    if(xScaleMax + dx > d_autoScaleXRange.second)
        dx = d_autoScaleXRange.second - xScaleMax;
    if(yScaleMin + dy < d_autoScaleYRange.first)
        dy = d_autoScaleYRange.first - yScaleMin;
    if(yScaleMax + dy > d_autoScaleYRange.second)
        dy = d_autoScaleYRange.second - yScaleMax;

    d_panClickPos -= delta;
    setAxisScale(QwtPlot::xBottom,xScaleMin + dx,xScaleMax + dx);
    if(dx)
    {
        updateAxes();
        //updateAxes() doesn't reprocess the scale labels, which might need to be recalculated.
        //this line will take care of that and avoid plot glitches
        QApplication::sendPostedEvents(this,QEvent::LayoutRequest);
    }
    setAxisScale(QwtPlot::yLeft,yScaleMin + dy,yScaleMax + dy);

    //only re-filter if we've moved horizontally (probably always!)
    replot(dx!=0.0);
}

void FtPlot::zoom(QWheelEvent *we)
{
    if(!we)
        return;

    //ctrl-wheel: zoom only horizontally
    //shift-wheel: zoom only vertically
    //no modifiers: zoom both

    //one step, which is 15 degrees, will zoom 10%
    //the delta function is in units of 1/8th a degree
    int numSteps = we->delta()/8/15;
    double factor = 0.1;

    //holding alt will make the x factor 5 times larger for faster zooming
    if(we->modifiers() & Qt::AltModifier)
        factor*=5.0;

    if(!(we->modifiers() & Qt::ShiftModifier))
    {
        //do horizontal rescaling
        //get current scale
        double scaleXMin = axisScaleDiv(QwtPlot::xBottom).lowerBound();
        double scaleXMax = axisScaleDiv(QwtPlot::xBottom).upperBound();

        //find mouse position on scale (and make sure it's on the scale!)
        double mousePos = canvasMap(QwtPlot::xBottom).invTransform(we->pos().x());
        mousePos = qMax(mousePos,scaleXMin);
        mousePos = qMin(mousePos,scaleXMax);

        //calculate distances from mouse to scale; remove or add 10% from/to each
        scaleXMin += fabs(mousePos - scaleXMin)*factor*(double)numSteps;
        scaleXMax -= fabs(mousePos - scaleXMax)*factor*(double)numSteps;

        //we can't go beyond range set by autoscale limits
        scaleXMin = qMax(scaleXMin,d_autoScaleXRange.first);
        scaleXMax = qMin(scaleXMax,d_autoScaleXRange.second);

        if(scaleXMin <= d_autoScaleXRange.first && scaleXMax >= d_autoScaleXRange.second)
            d_autoScaleXY.first = true;
        else
        {
            d_autoScaleXY.first = false;
            setAxisScale(QwtPlot::xBottom,scaleXMin,scaleXMax);
            updateAxes();
            //updateAxes() doesn't reprocess the scale labels, which might need to be recalculated.
            //this line will take care of that and avoid plot glitches
            QApplication::sendPostedEvents(this,QEvent::LayoutRequest);
        }
    }

    if(!(we->modifiers() & Qt::ControlModifier))
    {
        //do vertical rescaling
        //get current scale
        double scaleYMin = axisScaleDiv(QwtPlot::yLeft).lowerBound();
        double scaleYMax = axisScaleDiv(QwtPlot::yLeft).upperBound();

        //find mouse position on scale (and make sure it's on the scale!)
        double mousePos = canvasMap(QwtPlot::yLeft).invTransform(we->pos().y());
        mousePos = qMax(mousePos,scaleYMin);
        mousePos = qMin(mousePos,scaleYMax);

        //calculate distances from mouse to scale; remove or add 10% from/to each
        scaleYMin += fabs(mousePos - scaleYMin)*0.1*(double)numSteps;
        scaleYMax -= fabs(mousePos - scaleYMax)*0.1*(double)numSteps;

        //we can't go beyond range set by autoscale limits
        scaleYMin = qMax(scaleYMin,d_autoScaleYRange.first);
        scaleYMax = qMin(scaleYMax,d_autoScaleYRange.second);

        if(scaleYMin <= d_autoScaleYRange.first && scaleYMax >= d_autoScaleYRange.second)
            d_autoScaleXY.second = true;
        else
        {
            d_autoScaleXY.second = false;
            setAxisScale(QwtPlot::yLeft,scaleYMin,scaleYMax);
        }
    }

    replot();
}

void FtPlot::enableAutoScaling()
{
    d_autoScaleXY.first = true;
    d_autoScaleXY.second = true;
    replot();
}

void FtPlot::ftStartChanged(double s)
{
    QMetaObject::invokeMethod(p_ftw,"setStart",Q_ARG(double,s));
    if(!d_processing)
        updatePlot();
    else
        d_replotWhenDone = true;
}

void FtPlot::ftEndChanged(double e)
{
    QMetaObject::invokeMethod(p_ftw,"setEnd",Q_ARG(double,e));
    if(!d_processing)
        updatePlot();
    else
        d_replotWhenDone = true;

}

void FtPlot::updatePlot()
{
    QMetaObject::invokeMethod(p_ftw,"doFT",Q_ARG(const Fid,d_currentFid));
    d_processing = true;
    d_replotWhenDone = false;
}

void FtPlot::resizeEvent(QResizeEvent *e)
{
    //we don't care about the details of the event, pass it up the chain so Qt can handle the resizing
    QwtPlot::resizeEvent(e);

    //now that the plot has been resized, refilter the data if possible
    if(!d_currentFt.isEmpty())
        replot();
}

bool FtPlot::eventFilter(QObject *obj, QEvent *ev)
{
    //make sure object is the plot canvas
    if(obj == this->canvas())
    {
        if(ev->type() == QEvent::MouseButtonPress) //look for mouse button presses that would begin panning
        {
            QMouseEvent *me = static_cast<QMouseEvent*>(ev);
            if(me->button() == Qt::MiddleButton) //check to see if it's a middle click
            {
                if(!d_autoScaleXY.first || !d_autoScaleXY.second)
                {
                    //get coordinates of mouse click in terms of the plot scales
                    d_panClickPos = me->pos();
                    //begin panning session
                    d_panning=true;
                    ev->accept();
                    return true;
                }
            }
        }
        else if(d_panning && ev->type() == QEvent::MouseMove) //check for mouse movements
        {
            pan(static_cast<QMouseEvent*>(ev));
            ev->accept();
            return true;
        }
        else if(ev->type() == QEvent::MouseButtonRelease) //look for releases of mouse buttons
        {
            QMouseEvent *me = static_cast<QMouseEvent*>(ev);
            if(me->button() == Qt::MiddleButton)
            {
                d_panning = false;
                ev->accept();
                return true;
            }
            else if(me->button() == Qt::LeftButton && (me->modifiers()&Qt::ControlModifier))
            {
                enableAutoScaling();
                ev->accept();
                return true;
            }
        }
        else if (ev->type() == QEvent::Wheel)
        {
            zoom(static_cast<QWheelEvent*>(ev));
            ev->accept();
            return true;
        }
    }

    return QwtPlot::eventFilter(obj,ev);
}

void FtPlot::replot(bool filter)
{
    if(d_autoScaleXY.first)
    {
        setAxisScale(QwtPlot::xBottom,d_autoScaleXRange.first,d_autoScaleXRange.second);
        //x axes need to be updated before data are refiltered
        updateAxes();
        //updateAxes() doesn't reprocess the scale labels, which might need to be recalculated.
        //this line will take care of that and avoid plot glitches
        QApplication::sendPostedEvents(this,QEvent::LayoutRequest);
    }
    if(d_autoScaleXY.second)
        setAxisScale(QwtPlot::yLeft,d_autoScaleYRange.first,d_autoScaleYRange.second);

    if(filter)
        filterData();

    QwtPlot::replot();
}

