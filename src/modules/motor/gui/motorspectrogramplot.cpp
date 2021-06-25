#include <modules/motor/gui/motorspectrogramplot.h>

#include <QMenu>
#include <QMouseEvent>
#include <algorithm>

#include <qwt6/qwt_color_map.h>
#include <qwt6/qwt_scale_widget.h>
#include <qwt6/qwt_interval.h>


#include <gui/plot/customtracker.h>

MotorSpectrogramPlot::MotorSpectrogramPlot(const QString name, QWidget *parent) : ZoomPanPlot(name,parent)
{
    d_max = 1.0;

    d_intervalList.insert(MotorScan::MotorX,QwtInterval(0.0,1.0));
    d_intervalList.insert(MotorScan::MotorY,QwtInterval(0.0,1.0));
    d_intervalList.insert(MotorScan::MotorZ,QwtInterval(0.0,1.0));
    d_intervalList.insert(MotorScan::MotorT,QwtInterval(0.0,1.0));

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

    if(name.contains("Small",Qt::CaseInsensitive))
    {
        auto y = getOrSetDefault(BC::Key::leftAxis,QVariant::fromValue(MotorScan::MotorY)).value<MotorScan::MotorAxis>();
        auto x = getOrSetDefault(BC::Key::bottomAxis,QVariant::fromValue(MotorScan::MotorX)).value<MotorScan::MotorAxis>();
        auto s1 = getOrSetDefault(BC::Key::slider1Axis,QVariant::fromValue(MotorScan::MotorZ)).value<MotorScan::MotorAxis>();
        auto s2 = getOrSetDefault(BC::Key::slider2Axis,QVariant::fromValue(MotorScan::MotorT)).value<MotorScan::MotorAxis>();
        setAxis(QwtPlot::yLeft,y);
        setAxis(QwtPlot::xBottom,x);
        Q_UNUSED(s1)
        Q_UNUSED(s2)
    }
    else
    {
        auto y = getOrSetDefault(BC::Key::leftAxis,QVariant::fromValue(MotorScan::MotorY)).value<MotorScan::MotorAxis>();
        auto x = getOrSetDefault(BC::Key::bottomAxis,QVariant::fromValue(MotorScan::MotorZ)).value<MotorScan::MotorAxis>();
        auto s1 = getOrSetDefault(BC::Key::slider1Axis,QVariant::fromValue(MotorScan::MotorX)).value<MotorScan::MotorAxis>();
        auto s2 = getOrSetDefault(BC::Key::slider2Axis,QVariant::fromValue(MotorScan::MotorT)).value<MotorScan::MotorAxis>();
        setAxis(QwtPlot::yLeft,y);
        setAxis(QwtPlot::xBottom,x);
        Q_UNUSED(s1)
        Q_UNUSED(s2)
    }


}

MotorSpectrogramPlot::~MotorSpectrogramPlot()
{
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

    Q_FOREACH(const MotorScan::MotorAxis &axis, d_intervalList.keys())
    {
        QPair<double,double> p = s.interval(axis);
        d_intervalList[axis] = QwtInterval(p.first,p.second);
    }

    p_spectrogramData->setInterval(Qt::YAxis,d_intervalList.value(d_leftAxis));
    p_spectrogramData->setInterval(Qt::XAxis,d_intervalList.value(d_bottomAxis));
    p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(0.0,1.0));

    autoScale();
}

void MotorSpectrogramPlot::updateData(QVector<double> data, int cols)
{
    bool recalcRange = false;
    auto mm = std::minmax_element(data.begin(),data.end());
    double min = *(mm.first);
    double max = *(mm.second);

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

void MotorSpectrogramPlot::setAxis(QwtPlot::Axis plotAxis, MotorScan::MotorAxis motorAxis)
{
    QString text;

    /// \todo Figure out how to update slider axes here too

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

    setPlotAxisTitle(plotAxis,text);
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
