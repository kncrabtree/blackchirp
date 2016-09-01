#include "motorspectrogramplot.h"

#include <QMenu>
#include <QMouseEvent>

#include <qwt6/qwt_color_map.h>
#include <qwt6/qwt_scale_widget.h>
#include <qwt6/qwt_interval.h>

MotorSpectrogramPlot::MotorSpectrogramPlot(QWidget *parent) : ZoomPanPlot(QString("motorSpectrogramPlot"),parent)
{
    d_max = 1.0;

    setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));

    d_intervalList.insert(MotorScan::MotorX,QwtInterval(0.0,1.0));
    d_intervalList.insert(MotorScan::MotorY,QwtInterval(0.0,1.0));
    d_intervalList.insert(MotorScan::MotorZ,QwtInterval(0.0,1.0));
    d_intervalList.insert(MotorScan::MotorT,QwtInterval(0.0,1.0));

    connect(this,&MotorSpectrogramPlot::plotRightClicked,this,&MotorSpectrogramPlot::buildContextMenu);

    p_spectrogramData = new QwtMatrixRasterData;
    p_spectrogramData->setInterval(Qt::XAxis,QwtInterval(0.0,1.0));
    p_spectrogramData->setInterval(Qt::YAxis,QwtInterval(0.0,1.0));
    p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(0.0,1.0));
    p_spectrogramData->setValueMatrix(QVector<double>(),1);


    p_spectrogram = new QwtPlotSpectrogram();
    p_spectrogram->setDisplayMode(QwtPlotSpectrogram::ImageMode);
    p_spectrogram->setDisplayMode(QwtPlotSpectrogram::ContourMode,false);
    p_spectrogram->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_spectrogram->setData(p_spectrogramData);
    p_spectrogram->attach(this);

    enableAxis(QwtPlot::yRight);
    setAxisOverride(QwtPlot::yRight);


}

MotorSpectrogramPlot::~MotorSpectrogramPlot()
{
}

void MotorSpectrogramPlot::setLabelText(QwtPlot::Axis axis, QString text)
{
    QwtText label(text);
    label.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(axis,label);
}

void MotorSpectrogramPlot::prepareForScan(const MotorScan s)
{
    d_firstPoint = true;

    QwtLinearColorMap *map = new QwtLinearColorMap(Qt::black,Qt::red);
    map->addColorStop(0.05,Qt::darkBlue);
    map->addColorStop(0.1,Qt::blue);
    map->addColorStop(0.25,Qt::cyan);
    map->addColorStop(0.5,Qt::green);
    map->addColorStop(0.75,Qt::yellow);
    map->addColorStop(0.9,QColor(0xff,0x66,0));

    QwtScaleWidget *rightAxis = axisWidget( QwtPlot::yRight );
    rightAxis->setColorMap(QwtInterval(0.0,1.0),map);

    Q_FOREACH(const MotorScan::MotorDataAxis &axis, d_intervalList.keys())
    {
        QPair<double,double> p = s.interval(axis);
        d_intervalList[axis] = QwtInterval(p.first,p.second);
    }

    p_spectrogramData->setInterval(Qt::YAxis,d_intervalList.value(d_leftAxis));
    p_spectrogramData->setInterval(Qt::XAxis,d_intervalList.value(d_bottomAxis));
    p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(0.0,1.0));

    autoScale();
}

void MotorSpectrogramPlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *m = contextMenu();

    m->popup(me->globalPos());

}

void MotorSpectrogramPlot::updateData(QVector<double> data, int cols, double min, double max, MotorScan::MotorDataAxis leftAxis, MotorScan::MotorDataAxis bottomAxis)
{
    bool recalcRange = false;

    if(leftAxis != d_leftAxis)
    {
        recalcRange = true;
        setAxis(QwtPlot::yLeft,leftAxis);
        p_spectrogramData->setInterval(Qt::YAxis,d_intervalList.value(leftAxis));
        d_max = max;
        d_min = min;
    }

    if(bottomAxis != d_bottomAxis)
    {
        recalcRange = true;
        setAxis(QwtPlot::xBottom,bottomAxis);
        p_spectrogramData->setInterval(Qt::XAxis,d_intervalList.value(bottomAxis));
        d_max = max;
        d_min = min;
    }

    if(d_firstPoint)
    {
        recalcRange = true;
        d_firstPoint = false;
        d_max = max;
        d_min = min;
    }

    if(max > d_max)
    {
        recalcRange = true;
        d_max = max;
    }

    if(min < d_min)
    {
        recalcRange = true;
        d_min = min;
    }

    if(recalcRange)
        recalculateZRange();

    p_spectrogramData->setValueMatrix(data,cols);
    replot();
}

void MotorSpectrogramPlot::updatePoint(int row, int col, double val)
{
    bool recalcRange = false;
    if(val > d_max)
    {
        recalcRange = true;
        d_max = val;
    }

    if(val < d_min)
    {
        recalcRange = true;
        d_min = val;
    }

    if(recalcRange)
        recalculateZRange();

    p_spectrogramData->setValue(row,col,val);
    replot();
}

void MotorSpectrogramPlot::setAxis(QwtPlot::Axis plotAxis, MotorScan::MotorDataAxis motorAxis)
{
    QString text;

    switch (motorAxis) {
    case MotorScan::MotorX:
        text = QString("X (mm)");
        break;
    case MotorScan::MotorY:
        text = QString("Y (mm)");
        break;
    case MotorScan::MotorZ:
        text = QString("Z (mm)");
        break;
    case MotorScan::MotorT:
        text = QString::fromUtf16(u"T (µs)");
        break;
    }

    if(plotAxis == QwtPlot::yLeft || plotAxis == QwtPlot::yRight)
        d_leftAxis = motorAxis;
    else
        d_bottomAxis = motorAxis;

    setLabelText(plotAxis,text);
}



void MotorSpectrogramPlot::filterData()
{
}

void MotorSpectrogramPlot::recalculateZRange()
{
    p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(d_min,d_max));

    QwtLinearColorMap *map = new QwtLinearColorMap(Qt::black,Qt::red);
    map->addColorStop(0.05,Qt::darkBlue);
    map->addColorStop(0.1,Qt::blue);
    map->addColorStop(0.25,Qt::cyan);
    map->addColorStop(0.5,Qt::green);
    map->addColorStop(0.75,Qt::yellow);
    map->addColorStop(0.9,QColor(0xff,0x66,0));

    QwtScaleWidget *rightAxis = axisWidget( QwtPlot::yRight );
    rightAxis->setColorMap(QwtInterval(d_min,d_max),map);
    p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(d_min,d_max));
}


void MotorSpectrogramPlot::replot()
{
    setAxisAutoScaleRange(QwtPlot::xBottom,p_spectrogramData->interval(Qt::XAxis).minValue(),
                          p_spectrogramData->interval(Qt::XAxis).maxValue());
    setAxisAutoScaleRange(QwtPlot::yLeft,p_spectrogramData->interval(Qt::YAxis).minValue(),
                          p_spectrogramData->interval(Qt::YAxis).maxValue());
    setAxisAutoScaleRange(QwtPlot::yRight,p_spectrogramData->interval(Qt::ZAxis).minValue(),
                          p_spectrogramData->interval(Qt::ZAxis).maxValue());

    ZoomPanPlot::replot();

}
