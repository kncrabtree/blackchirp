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
    ZoomPanPlot(parent), d_autoScaleXRange(qMakePair(0.0,1.0)), d_autoScaleYRange(qMakePair(0.0,1.0)),
    d_processing(false), d_replotWhenDone(false)
{
    //make axis label font smaller
    this->setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    this->setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));

    //build axis titles with small font
    QwtText blabel(QString("Frequency (MHz)"));
    blabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::xBottom,blabel);

    QwtText llabel(QString("FT"));
    llabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::yLeft,llabel);

    QSettings s;

    //build and configure curve object
    p_curveData = new QwtPlotCurve();
    QColor c = s.value(QString("ftcolor"),palette().color(QPalette::BrightText)).value<QColor>();
    p_curveData->setPen(QPen(c));
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
    QPen p;
    p.setColor(s.value(tr("gridcolor"),palette().color(QPalette::Light)).value<QColor>());
    p.setStyle(Qt::DashLine);
    p_plotGrid->setMajorPen(p);
    p.setStyle(Qt::DotLine);
    p_plotGrid->setMinorPen(p);
    p_plotGrid->attach(this);

    connect(this,&FtPlot::plotRightClicked,this,&FtPlot::buildContextMenu);

    p_ftw = new FtWorker();
    //make signal/slot connections
    connect(p_ftw,&FtWorker::ftDone,this,&FtPlot::ftDone);
    connect(p_ftw,&FtWorker::fidDone,this,&FtPlot::fidDone);
    p_ftThread = new QThread(this);
    connect(p_ftThread,&QThread::finished,p_ftw,&FtWorker::deleteLater);
    p_ftw->moveToThread(p_ftThread);
    p_ftThread->start();

    setAxisAutoScale(QwtPlot::xBottom,false);
    setAxisAutoScale(QwtPlot::yLeft,false);

    setAxisAutoScaleRange(QwtPlot::xBottom,d_autoScaleXRange.first,d_autoScaleXRange.second);
    setAxisAutoScaleRange(QwtPlot::yLeft,d_autoScaleYRange.first,d_autoScaleYRange.second);

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

    setAxisAutoScaleRange(QwtPlot::xBottom,d_autoScaleXRange.first,d_autoScaleXRange.second);
    setAxisAutoScaleRange(QwtPlot::yLeft,d_autoScaleYRange.first,d_autoScaleYRange.second);

    filterData();

    if(d_replotWhenDone)
        updatePlot();

    replot();
}

void FtPlot::filterData()
{
    if(d_currentFt.size() < 2)
        return;

    double firstPixel = 0.0;
    double lastPixel = canvas()->width();
    QwtScaleMap map = canvasMap(QwtPlot::xBottom);

    QVector<QPointF> filtered;
    filtered.reserve(canvas()->width()*2);

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
            QPointF second(map.invTransform(pixel),d_currentFt.at(maxIndex).y());
            filtered.append(first);
            filtered.append(second);
        }
    }

    if(dataIndex < d_currentFt.size())
        filtered.append(d_currentFt.at(dataIndex));

    //assign data to curve object
    p_curveData->setSamples(filtered);
}

void FtPlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *m = contextMenu();

    QAction *ftColorAction = m->addAction(QString("Change FT Color..."));
    connect(ftColorAction,&QAction::triggered,this,[=](){ changeFtColor(getColor(p_curveData->pen().color())); });

    QAction *gridColorAction = m->addAction(QString("Change Grid Color..."));
    connect(gridColorAction,&QAction::triggered,this,[=](){ changeGridColor(getColor(p_plotGrid->majorPen().color())); });

    m->popup(me->globalPos());
}

void FtPlot::changeFtColor(QColor c)
{
    if(!c.isValid())
        return;

    QSettings s;
    s.setValue(tr("ftcolor"),c);
    s.sync();

    p_curveData->setPen(QPen(c));
    replot();

}

void FtPlot::changeGridColor(QColor c)
{
    if(!c.isValid())
        return;

    QSettings s;
    s.setValue(tr("gridcolor"),c);
    s.sync();

    QPen p(c);
    p.setStyle(Qt::DashLine);
    p_plotGrid->setMajorPen(p);

    p.setStyle(Qt::DotLine);
    p_plotGrid->setMinorPen(p);
    replot();
}

QColor FtPlot::getColor(QColor startingColor)
{
    return QColorDialog::getColor(startingColor,this,tr("Select Color"));
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
