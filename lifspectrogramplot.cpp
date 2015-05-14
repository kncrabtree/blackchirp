#include "lifspectrogramplot.h"

#include <QMouseEvent>

#include <qwt6/qwt_plot_spectrogram.h>
#include <qwt6/qwt_matrix_raster_data.h>
#include <qwt6/qwt_color_map.h>
#include <qwt6/qwt_scale_widget.h>
#include <qwt6/qwt_plot_marker.h>

LifSpectrogramPlot::LifSpectrogramPlot(QWidget *parent) :
    ZoomPanPlot(QString("lifSpectrogram"),parent), d_enabled(false), d_firstPoint(false), d_zMax(0.0),
    d_delayDragging(false), d_freqDragging(false), d_grabDelay(false), d_grabFreq(false)
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

    p_spectrogramData = new QwtMatrixRasterData;
    p_spectrogram->setData(p_spectrogramData);

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

    p_delayMarker = new QwtPlotMarker();
    p_delayMarker->setLineStyle(QwtPlotMarker::HLine);
    p_delayMarker->setLinePen(Qt::white,2.0);
    p_delayMarker->setVisible(false);
    p_delayMarker->attach(this);

    p_freqMarker = new QwtPlotMarker();
    p_freqMarker->setLineStyle(QwtPlotMarker::VLine);
    p_freqMarker->setLinePen(Qt::white,2.0);
    p_freqMarker->setVisible(false);
    p_freqMarker->attach(this);

    setAxisAutoScaleRange(QwtPlot::yRight,0.0,1.0);
    enableAxis(QwtPlot::yRight);

    setAxisOverride(QwtPlot::yRight);

    canvas()->installEventFilter(this);
    canvas()->setMouseTracking(true);


}

LifSpectrogramPlot::~LifSpectrogramPlot()
{

}

void LifSpectrogramPlot::prepareForExperiment(const LifConfig c)
{
    d_enabled = c.isEnabled();
    p_spectrogramData = new QwtMatrixRasterData;
    d_delayDragging = false;
    d_freqDragging = false;
    d_grabDelay = false;
    d_grabFreq = false;

    if(c.isEnabled())
    {
        d_firstPoint = true;

        QVector<double> specDat;
        specDat.resize(c.numDelayPoints()*c.numFrequencyPoints());
        p_spectrogramData->setValueMatrix(specDat,c.numDelayPoints());
        p_spectrogramData->setResampleMode(QwtMatrixRasterData::BilinearInterpolation);

        QPair<double,double> delayRange = c.delayRange();
        QPair<double,double> freqRange = c.frequencyRange();

        double fHalfStep = fabs(freqRange.second - freqRange.first)/static_cast<double>(p_spectrogramData->numRows()-1)/2.0;
        double dHalfStep = fabs(delayRange.second - delayRange.first)/static_cast<double>(p_spectrogramData->numColumns()-1)/2.0;

        double dMin = qMin(delayRange.first,delayRange.second) - dHalfStep;
        double dMax = qMax(delayRange.first,delayRange.second) + dHalfStep;
        double fMin = qMin(freqRange.first,freqRange.second) - fHalfStep;
        double fMax = qMax(freqRange.first,freqRange.second) + fHalfStep;

        d_delayRange = qMakePair(dMin,dMax);
        d_freqRange = qMakePair(fMin,fMax);

        p_spectrogramData->setInterval(Qt::YAxis,QwtInterval(dMin,dMax));
        p_spectrogramData->setInterval(Qt::XAxis,QwtInterval(fMin,fMax));

        setAxisAutoScaleRange(QwtPlot::xBottom,fMin,fMax);
        setAxisAutoScaleRange(QwtPlot::yLeft,dMin,dMax);

    }
    else
    {
        d_firstPoint = false;

        setAxisAutoScaleRange(QwtPlot::xBottom,0.0,1.0);
        setAxisAutoScaleRange(QwtPlot::yLeft,0.0,1.0);
    }

    p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(0.0,1.0));
    p_spectrogram->setData(p_spectrogramData);
    setAxisAutoScaleRange(QwtPlot::yRight,0.0,1.0);
    p_delayMarker->setVisible(false);
    p_freqMarker->setVisible(false);

    autoScale();

    d_zMax = 0.0;

}

void LifSpectrogramPlot::updatePoint(int row, int col, double val)
{
    if(val > d_zMax)
    {
        setAxisAutoScaleRange(QwtPlot::yRight,0.0,val);
        p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(0.0,val));

        QwtLinearColorMap *map2 = new QwtLinearColorMap(Qt::black,Qt::red);
        map2->addColorStop(0.05,Qt::darkBlue);
        map2->addColorStop(0.1,Qt::blue);
        map2->addColorStop(0.25,Qt::cyan);
        map2->addColorStop(0.5,Qt::green);
        map2->addColorStop(0.75,Qt::yellow);
        map2->addColorStop(0.9,QColor(0xff,0x66,0));

        QwtScaleWidget *rightAxis = axisWidget( QwtPlot::yRight );
        rightAxis->setColorMap(QwtInterval(0.0,val),map2);

        d_zMax = val;
    }

    if(d_firstPoint)
    {
        d_firstPoint = false;

        const QwtInterval dInt = p_spectrogramData->interval(Qt::YAxis);
        double dy = dInt.width()/static_cast<double>(p_spectrogramData->numRows());
        double delayVal = static_cast<double>(row)*dy + dInt.minValue() + dy/2.0;
        p_delayMarker->setYValue(delayVal);

        const QwtInterval fInt = p_spectrogramData->interval(Qt::XAxis);
        double dx = fInt.width()/static_cast<double>(p_spectrogramData->numColumns());
        double freqVal = static_cast<double>(col)*dx + fInt.minValue() + dx/2.0;
        p_freqMarker->setXValue(freqVal);

        p_delayMarker->setVisible(true);
        p_freqMarker->setVisible(true);
    }

    p_spectrogramData->setValue(row,col,val);
    replot();

}

void LifSpectrogramPlot::replot()
{
    if(!d_enabled || d_firstPoint)
    {
        QwtLinearColorMap *map2 = new QwtLinearColorMap(Qt::black,Qt::red);
        map2->addColorStop(0.05,Qt::darkBlue);
        map2->addColorStop(0.1,Qt::blue);
        map2->addColorStop(0.25,Qt::cyan);
        map2->addColorStop(0.5,Qt::green);
        map2->addColorStop(0.75,Qt::yellow);
        map2->addColorStop(0.9,QColor(0xff,0x66,0));

        QwtScaleWidget *rightAxis = axisWidget( QwtPlot::yRight );
        rightAxis->setColorMap(QwtInterval(0.0,1.0),map2);
    }

    ZoomPanPlot::replot();
}

void LifSpectrogramPlot::filterData()
{
}

bool LifSpectrogramPlot::eventFilter(QObject *obj, QEvent *ev)
{
    if(d_enabled && !d_firstPoint)
    {
        if(obj == canvas())
        {
            if(ev->type() == QEvent::MouseButtonPress)
            {
                if(d_grabDelay || d_grabFreq)
                {
                    QMouseEvent *me = static_cast<QMouseEvent*>(ev);
                    if(me->button() == Qt::LeftButton)
                    {
                        d_delayDragging = d_grabDelay;
                        d_freqDragging = d_grabFreq;
                    }
                }
            }

            if(ev->type() == QEvent::MouseButtonRelease)
            {
                QMouseEvent *me = static_cast<QMouseEvent*>(ev);
                if(me->button() == Qt::LeftButton)
                {
                    d_delayDragging = false;
                    d_freqDragging = false;
                }
            }

            if(ev->type() == QEvent::MouseMove)
            {
                QMouseEvent *me = static_cast<QMouseEvent*>(ev);

                //if we're not dragging, find out if we're close enough to grab either marker
                if(!d_delayDragging && !d_freqDragging)
                {
                    int grabMargin = 10; //how close we have to be to grab

                    //get pixel position of delay marker; find out if it's within the margin
                    int dPixel = canvasMap(QwtPlot::yLeft).transform(p_delayMarker->yValue());
                    int mPixel = me->pos().y();
                    if(qAbs(dPixel - mPixel) < grabMargin)
                        d_grabDelay = true;
                    else
                        d_grabDelay = false;


                    int fPixel = canvasMap(QwtPlot::xBottom).transform(p_freqMarker->xValue());
                    mPixel = me->pos().x();
                    if(qAbs(fPixel - mPixel) < grabMargin)
                        d_grabFreq = true;
                    else
                        d_grabFreq = false;

                    if(d_grabFreq && d_grabDelay)
                        canvas()->setCursor(Qt::SizeAllCursor);
                    else if(d_grabDelay)
                        canvas()->setCursor(Qt::SizeVerCursor);
                    else if(d_grabFreq)
                        canvas()->setCursor(Qt::SizeHorCursor);
                    else
                        canvas()->setCursor(Qt::CrossCursor);
                }
                else
                {
                    if(d_delayDragging)
                    {
                        //snap to nearest delay point
                        double mVal = canvasMap(QwtPlot::yLeft).invTransform(me->pos().y());
                        QwtInterval dInt = p_spectrogramData->interval(Qt::YAxis);
                        double dy = dInt.width()/static_cast<double>(p_spectrogramData->numRows());
                        int row = qBound(0,static_cast<int>(floor((mVal-dInt.minValue())/dy)),p_spectrogramData->numColumns()-1);
                        double delayVal = static_cast<double>(row)*dy + dInt.minValue() + dy/2.0;
                        p_delayMarker->setYValue(delayVal);
                        emit freqSlice(row);
                    }

                    if(d_freqDragging)
                    {
                        //snap to nearest freq point
                        double mVal = canvasMap(QwtPlot::xBottom).invTransform(me->pos().x());
                        QwtInterval fInt = p_spectrogramData->interval(Qt::XAxis);
                        double dx = fInt.width()/static_cast<double>(p_spectrogramData->numColumns());
                        int col = qBound(0,static_cast<int>(floor((mVal-fInt.minValue())/dx)),p_spectrogramData->numColumns()-1);
                        double freqVal = static_cast<double>(col)*dx + fInt.minValue() + dx/2.0;
                        p_freqMarker->setXValue(freqVal);
                        emit delaySlice(col);
                    }

                    replot();
                }
            }
        }
    }

    return ZoomPanPlot::eventFilter(obj,ev);
}
