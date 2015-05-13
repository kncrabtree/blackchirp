#include "lifspectrogramplot.h"

#include <qwt6/qwt_plot_spectrogram.h>
#include <qwt6/qwt_matrix_raster_data.h>
#include <qwt6/qwt_color_map.h>
#include <qwt6/qwt_scale_widget.h>

LifSpectrogramPlot::LifSpectrogramPlot(QWidget *parent) :
    ZoomPanPlot(QString("lifSpectrogram"),parent), d_enabled(false), d_zMax(0.0)
{
    QFont f(QString("sans-serif"),8);
    setAxisFont(QwtPlot::xBottom,f);
    setAxisFont(QwtPlot::yLeft,f);

    QwtText llabel(QString::fromUtf16(u"Delay (µs)"));
    llabel.setFont(f);
    setAxisTitle(QwtPlot::yLeft,llabel);

    QwtText blabel(QString::fromUtf16(u"Frequency (cm⁻¹)"));
    blabel.setFont(f);
    setAxisTitle(QwtPlot::xBottom,blabel);

    p_spectrogram = new QwtPlotSpectrogram();
    p_spectrogram->setDisplayMode(QwtPlotSpectrogram::ImageMode);
    p_spectrogram->setDisplayMode(QwtPlotSpectrogram::ContourMode,false);
    p_spectrogram->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_spectrogram->setData(new QwtMatrixRasterData);
    p_spectrogram->attach(this);

    QwtLinearColorMap *map = new QwtLinearColorMap(Qt::black,Qt::red);
    map->addColorStop(0.05,Qt::darkBlue);
    map->addColorStop(0.1,Qt::blue);
    map->addColorStop(0.25,Qt::cyan);
    map->addColorStop(0.5,Qt::green);
    map->addColorStop(0.75,Qt::yellow);
    map->addColorStop(0.9,QColor(0xff,0x66,0));
    p_spectrogram->setColorMap(map);

    QwtScaleWidget *rightAxis = axisWidget( QwtPlot::yRight );

    QwtText rLabel(QString("LIF (AU)"));
    rLabel.setFont(f);
    rightAxis->setTitle(rLabel);

    QwtLinearColorMap *map2 = new QwtLinearColorMap(Qt::black,Qt::red);
    map2->addColorStop(0.05,Qt::darkBlue);
    map2->addColorStop(0.1,Qt::blue);
    map2->addColorStop(0.25,Qt::cyan);
    map2->addColorStop(0.5,Qt::green);
    map2->addColorStop(0.75,Qt::yellow);
    map2->addColorStop(0.9,QColor(0xff,0x66,0));

    rightAxis->setColorMap(QwtInterval(0.0,1.0),map2);
    rightAxis->setColorBarEnabled(true);
    rightAxis->setColorBarWidth(10);

    setAxisAutoScaleRange(QwtPlot::yRight,0.0,1.0);
    enableAxis(QwtPlot::yRight);

    setAxisOverride(QwtPlot::yRight);



}

LifSpectrogramPlot::~LifSpectrogramPlot()
{

}

void LifSpectrogramPlot::setRasterData(QwtMatrixRasterData *dat)
{
    p_spectrogram->setData(dat);
}

void LifSpectrogramPlot::prepareForExperiment(double xMin, double xMax, double yMin, double yMax, bool enabled)
{
    d_enabled = enabled;

    setAxisAutoScaleRange(QwtPlot::xBottom,xMin,xMax);
    setAxisAutoScaleRange(QwtPlot::yLeft,yMin,yMax);
    setAxisAutoScaleRange(QwtPlot::yRight,0.0,1.0);

    autoScale();

    d_zMax = 0.0;

}

void LifSpectrogramPlot::replot()
{
    QwtInterval zint = p_spectrogram->data()->interval(Qt::ZAxis);
    if(d_zMax < zint.maxValue())
    {
        QwtLinearColorMap *map2 = new QwtLinearColorMap(Qt::black,Qt::red);
        map2->addColorStop(0.05,Qt::darkBlue);
        map2->addColorStop(0.1,Qt::blue);
        map2->addColorStop(0.25,Qt::cyan);
        map2->addColorStop(0.5,Qt::green);
        map2->addColorStop(0.75,Qt::yellow);
        map2->addColorStop(0.9,QColor(0xff,0x66,0));

        QwtScaleWidget *rightAxis = axisWidget( QwtPlot::yRight );
        rightAxis->setColorMap(zint,map2);

        d_zMax = zint.maxValue();
    }

    ZoomPanPlot::replot();
}

void LifSpectrogramPlot::filterData()
{
}
