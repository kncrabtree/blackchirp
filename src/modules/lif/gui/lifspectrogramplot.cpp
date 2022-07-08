#include "lifspectrogramplot.h"

#include <QMouseEvent>
#include <QMenu>
#include <math.h>

#include <modules/lif/hardware/liflaser/liflaser.h>

#include <qwt6/qwt_plot_spectrogram.h>
#include <qwt6/qwt_matrix_raster_data.h>
#include <qwt6/qwt_color_map.h>
#include <qwt6/qwt_scale_widget.h>
#include <qwt6/qwt_plot_marker.h>

LifSpectrogramPlot::LifSpectrogramPlot(QWidget *parent) :
    ZoomPanPlot(BC::Key::lifSpectrogram,parent), d_enabled(false),
    d_delayDragging(false), d_freqDragging(false), d_grabDelay(false), d_grabFreq(false)
{

    SettingsStorage s(BC::Key::LifLaser::key,SettingsStorage::Hardware);
    setPlotAxisTitle(QwtPlot::yLeft,QString::fromUtf16(u"Delay (µs)"));
    setPlotAxisTitle(QwtPlot::xBottom,
                 QString("Laser Postiion (")+s.get<QString>(BC::Key::LifLaser::units,"nm")+QString(")"));

    p_spectrogram = new QwtPlotSpectrogram();
    p_spectrogram->setDisplayMode(QwtPlotSpectrogram::ImageMode);
    p_spectrogram->setDisplayMode(QwtPlotSpectrogram::ContourMode,false);
    p_spectrogram->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_spectrogramData = nullptr;// new QwtMatrixRasterData;
//    p_spectrogram->setData(p_spectrogramData);
//    p_spectrogram->attach(this);


    QwtLinearColorMap *map = new QwtLinearColorMap(Qt::black,Qt::red);
    map->addColorStop(0.05,Qt::darkBlue);
    map->addColorStop(0.1,Qt::blue);
    map->addColorStop(0.25,Qt::cyan);
    map->addColorStop(0.5,Qt::green);
    map->addColorStop(0.75,Qt::yellow);
    map->addColorStop(0.9,QColor(0xff,0x66,0));
    p_spectrogram->setColorMap(map);

    QwtScaleWidget *rightAxis = axisWidget( QwtPlot::yRight );

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
    setPlotAxisTitle(QwtPlot::yRight,QString("LIF (AU)"));

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
//    d_enabled =
//    if(p_spectrogramData != nullptr)
//    {
//        p_spectrogram->setData(nullptr);
//        p_spectrogramData = nullptr;
//    }

//    d_delayDragging = false;
//    d_freqDragging = false;
//    d_grabDelay = false;
//    d_grabFreq = false;

//    if(c.isEnabled())
//    {
//        QVector<double> specDat;
//        specDat.resize(c.numDelayPoints()*c.numLaserPoints());
//        p_spectrogramData = new QwtMatrixRasterData;
//        p_spectrogramData->setValueMatrix(specDat,c.numLaserPoints());

//        QPair<double,double> delayRange = c.delayRange();
//        QPair<double,double> freqRange = c.laserRange();


//        double dMin = qMin(delayRange.first,delayRange.second) - c.delayStep()/2.0;
//        double dMax = qMax(delayRange.first,delayRange.second) + c.delayStep()/2.0;
//        double fMin = qMin(freqRange.first,freqRange.second) - c.laserStep()/2.0;
//        double fMax = qMax(freqRange.first,freqRange.second) + c.laserStep()/2.0;

//        p_spectrogramData->setInterval(Qt::YAxis,QwtInterval(dMin,dMax));
//        p_spectrogramData->setInterval(Qt::XAxis,QwtInterval(fMin,fMax));
//        p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(0.0,1.0));
//        p_spectrogramData->setResampleMode(QwtMatrixRasterData::BilinearInterpolation);

//        if(c.numDelayPoints() > 1)
//        {
//            p_delayMarker->setYValue(delayRange.first);
//            p_delayMarker->setVisible(true);
//        }

//        if(c.numLaserPoints() > 1)
//        {
//            p_freqMarker->setVisible(true);
//            p_freqMarker->setXValue(freqRange.first);
//        }

//        p_spectrogram->setData(p_spectrogramData);
//        p_spectrogram->attach(this);

//    }
//    else
//    {
//        p_delayMarker->setVisible(false);
//        p_freqMarker->setVisible(false);
//    }


    autoScale();

}

void LifSpectrogramPlot::updateData(const QVector<double> d, int numCols, double zMin, double zMax)
{
    if(d.size() < 2)
        return;

    p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(zMin,zMax));

    QwtLinearColorMap *map2 = new QwtLinearColorMap(Qt::black,Qt::red);
    map2->addColorStop(0.05,Qt::darkBlue);
    map2->addColorStop(0.1,Qt::blue);
    map2->addColorStop(0.25,Qt::cyan);
    map2->addColorStop(0.5,Qt::green);
    map2->addColorStop(0.75,Qt::yellow);
    map2->addColorStop(0.9,QColor(0xff,0x66,0));

    QwtScaleWidget *rightAxis = axisWidget( QwtPlot::yRight );
    rightAxis->setColorMap(QwtInterval(zMin,zMax),map2);

//    if(d_firstPoint)
//    {
//        d_firstPoint = false;


//        p_delayMarker->setVisible(true);
//        p_freqMarker->setVisible(true);

//        emit delaySlice(0);
//        emit freqSlice(0);
//    }

    p_spectrogramData->setValueMatrix(d,numCols);
    p_spectrogram->setData(p_spectrogramData);
    replot();

}

void LifSpectrogramPlot::replot()
{
    if(!d_enabled)
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

void LifSpectrogramPlot::moveFreqCursor(QPoint pos)
{
    //snap to nearest freq point
    double mVal = canvasMap(QwtPlot::xBottom).invTransform(pos.x());
    QwtInterval fInt = p_spectrogramData->interval(Qt::XAxis);
    double dx = fInt.width()/static_cast<double>(p_spectrogramData->numColumns());
    int col = qBound(0,static_cast<int>(floor((mVal-fInt.minValue())/dx)),p_spectrogramData->numColumns()-1);
    double freqVal = static_cast<double>(col)*dx + fInt.minValue() + dx/2.0;
    p_freqMarker->setXValue(freqVal);
    emit delaySlice(col);
}

void LifSpectrogramPlot::moveDelayCursor(QPoint pos)
{
    //snap to nearest delay point
    double mVal = canvasMap(QwtPlot::yLeft).invTransform(pos.y());
    QwtInterval dInt = p_spectrogramData->interval(Qt::YAxis);
    double dy = dInt.width()/static_cast<double>(p_spectrogramData->numRows());
    int row = qBound(0,static_cast<int>(floor((mVal-dInt.minValue())/dy)),p_spectrogramData->numRows()-1);
    double delayVal = static_cast<double>(row)*dy + dInt.minValue() + dy/2.0;
    p_delayMarker->setYValue(delayVal);
    emit freqSlice(row);
}

void LifSpectrogramPlot::buildContextMenu(QMouseEvent *me)
{
    if(!d_enabled)
        return;

    QMenu *menu = ZoomPanPlot::contextMenu();

    menu->addSeparator();

    //must copy mouse position here; me pointer not valid when slot invoked!
    QPoint pos = me->pos();

    QAction *delayCursorAction = menu->addAction(QString("Move delay cursor here"));
    connect(delayCursorAction,&QAction::triggered,[=](){ moveDelayCursor(pos); });

    QAction *freqCursorAction = menu->addAction(QString("Move frequency cursor here"));
    connect(freqCursorAction,&QAction::triggered,[=](){ moveFreqCursor(pos); });

    QAction *bothCursorAction = menu->addAction(QString("Move both cursors here"));
    connect(bothCursorAction,&QAction::triggered,[=](){ moveDelayCursor(pos); moveFreqCursor(pos); });

    menu->popup(me->globalPos());
}

bool LifSpectrogramPlot::eventFilter(QObject *obj, QEvent *ev)
{
    if(d_enabled)
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
                        moveDelayCursor(me->pos());
                    if(d_freqDragging)
                        moveFreqCursor(me->pos());

                    replot();
                }
            }
        }
    }

    return ZoomPanPlot::eventFilter(obj,ev);
}
