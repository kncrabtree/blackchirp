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
#include <qwt6/qwt_interval.h>
#include <qwt6/qwt_scale_map.h>

LifSpectrogramPlot::LifSpectrogramPlot(QWidget *parent) :
    ZoomPanPlot(BC::Key::lifSpectrogram,parent), d_enabled(false),
    d_delayDragging(false), d_freqDragging(false), d_grabDelay(false), d_grabFreq(false)
{
    setSpectrogramMode(true);

    SettingsStorage s(BC::Key::LifLaser::key,SettingsStorage::Hardware);
    setPlotAxisTitle(QwtPlot::yLeft,QString::fromUtf16(u"Delay (µs)"));
    setPlotAxisTitle(QwtPlot::xBottom,
                 QString("Laser Postiion (")+s.get<QString>(BC::Key::LifLaser::units,"nm")+QString(")"));

    p_spectrogram = new QwtPlotSpectrogram();
    p_spectrogram->setDisplayMode(QwtPlotSpectrogram::ImageMode);
    p_spectrogram->setDisplayMode(QwtPlotSpectrogram::ContourMode,false);
    p_spectrogram->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_spectrogramData = nullptr;


    QwtLinearColorMap *map = new QwtLinearColorMap(QColor(0x02,0x28,0x51),QColor(0xff,0xdf,0x00));
    p_spectrogram->setColorMap(map);

    QwtScaleWidget *rightAxis = axisWidget( QwtPlot::yRight );

    QwtLinearColorMap *map2 = new QwtLinearColorMap(QColor(0x02,0x28,0x51),QColor(0xff,0xdf,0x00));
    rightAxis->setColorMap(QwtInterval(0.0,1.0),map2);
    rightAxis->setColorBarEnabled(true);
    rightAxis->setColorBarWidth(10);
    setPlotAxisTitle(QwtPlot::yRight,QString("LIF (AU)"));

    p_delayMarker = new QwtPlotMarker();
    p_delayMarker->setLineStyle(QwtPlotMarker::HLine);
    p_delayMarker->setLinePen(Qt::white,2.0);
    p_delayMarker->setVisible(false);
    p_delayMarker->attach(this);

    p_laserMarker = new QwtPlotMarker();
    p_laserMarker->setLineStyle(QwtPlotMarker::VLine);
    p_laserMarker->setLinePen(Qt::white,2.0);
    p_laserMarker->setVisible(false);
    p_laserMarker->attach(this);

    enableAxis(QwtPlot::yRight);
    setAxisOverride(QwtPlot::yRight,true);

    canvas()->installEventFilter(this);
    canvas()->setMouseTracking(true);

}

LifSpectrogramPlot::~LifSpectrogramPlot()
{

}

void LifSpectrogramPlot::clear()
{
    if(p_spectrogramData != nullptr)
    {
        p_spectrogram->setData(nullptr);
        p_spectrogramData = nullptr;
    }

    d_enabled = false;
    d_live = true;
    d_liveDelayIndex = 0;
    d_liveLaserIndex = 0;
    d_delayDragging = false;
    d_freqDragging = false;
    d_grabDelay = false;
    d_grabFreq = false;
    d_dMin = 0.0;
    d_ddx = 1.0;
    d_lMin = 0.0;
    d_ldx = 1.0;

    p_delayMarker->setVisible(false);
    p_laserMarker->setVisible(false);

    autoScale();
}

void LifSpectrogramPlot::prepareForExperiment(const LifConfig &c)
{
    d_enabled = true;
    d_live = true;
    d_liveDelayIndex = 0;
    d_liveLaserIndex = 0;

    QVector<double> specDat;
    specDat.resize(c.d_delayPoints*c.d_laserPosPoints);
    p_spectrogramData = new QwtMatrixRasterData;
    p_spectrogramData->setValueMatrix(specDat,c.d_laserPosPoints);

    auto delayRange = c.delayRange();
    auto laserRange = c.laserRange();


    d_dMin = qMin(delayRange.first,delayRange.second);
    d_ddx = qAbs(c.d_delayStepUs);

    d_lMin = qMin(laserRange.first,laserRange.second);
    d_ldx = qAbs(c.d_laserPosStep);


    double dMin =  d_dMin - qAbs(c.d_delayStepUs)/2.0;
    double dMax = qMax(delayRange.first,delayRange.second) + qAbs(c.d_delayStepUs)/2.0;
    if(c.d_delayPoints == 1)
        dMax = dMin+1.0;
    double fMin = d_lMin - qAbs(c.d_laserPosStep)/2.0;
    double fMax = qMax(laserRange.first,laserRange.second) + qAbs(c.d_laserPosStep)/2.0;
    if(c.d_laserPosPoints == 1)
        fMax = fMin + 1.0;

    p_spectrogramData->setInterval(Qt::YAxis,QwtInterval(dMin,dMax));
    p_spectrogramData->setInterval(Qt::XAxis,QwtInterval(fMin,fMax));
    p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(0.0,1.0));
    p_spectrogramData->setResampleMode(QwtMatrixRasterData::BilinearInterpolation);

    if(c.d_delayPoints > 1)
    {
        p_delayMarker->setYValue(delayRange.first);
        p_delayMarker->setVisible(true);
    }

    if(c.d_laserPosPoints > 1)
    {
        p_laserMarker->setVisible(true);
        p_laserMarker->setXValue(laserRange.first);
    }

    p_spectrogram->setData(p_spectrogramData);
    p_spectrogram->attach(this);


    autoScale();
}

void LifSpectrogramPlot::updateData(const QVector<double> d, int numCols)
{
    if(d.size() < 2)
        return;

    double zMin = 0.0, zMax = 0.0;
    for(int i=0; i<d.size(); i++)
    {
        zMin = qMin(zMin,d.at(i));
        zMax = qMax(zMax,d.at(i));
    }
    p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(zMin,zMax));

    QwtLinearColorMap *map = new QwtLinearColorMap(QColor(0x02,0x28,0x51),QColor(0xff,0xdf,0x00));

    QwtScaleWidget *rightAxis = axisWidget( QwtPlot::yRight );
    rightAxis->setColorMap(QwtInterval(zMin,zMax),map);

    p_spectrogramData->setValueMatrix(d,numCols);
    p_spectrogram->setData(p_spectrogramData);
    overrideAxisAutoScaleRange(QwtPlot::yRight,zMin,zMax);
    replot();

}

void LifSpectrogramPlot::setLiveIndices(int di, int li)
{
    d_liveDelayIndex = di;
    d_liveLaserIndex = li;

    if(d_live)
    {
        moveDelayCursor(di);
        moveLaserCursor(li);
        replot();
    }
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

double LifSpectrogramPlot::delayVal(int index) const
{
    return static_cast<double>(index)*d_ddx + d_dMin;
}

double LifSpectrogramPlot::laserVal(int index) const
{
    return static_cast<double>(index)*d_ldx + d_lMin;
}

int LifSpectrogramPlot::currentDelayIndex() const
{
    return qBound(0,static_cast<int>(floor((p_delayMarker->yValue()-d_dMin)/d_ddx)),p_spectrogramData->numRows()-1);
}

int LifSpectrogramPlot::currentLaserIndex() const
{
    return qBound(0,static_cast<int>(floor((p_laserMarker->xValue()-d_lMin)/d_ldx)),p_spectrogramData->numColumns()-1);
}

void LifSpectrogramPlot::moveLaserCursor(QPoint pos)
{
    //snap to nearest freq point
    d_live = false;
    double mVal = canvasMap(QwtPlot::xBottom).invTransform(pos.x());
    int col = qBound(0,static_cast<int>(floor((mVal-d_lMin)/d_ldx)),p_spectrogramData->numColumns()-1);
    moveLaserCursor(col);
    emit delaySlice(col);
}

void LifSpectrogramPlot::moveLaserCursor(int index)
{
    p_laserMarker->setXValue(laserVal(index));
}

void LifSpectrogramPlot::moveDelayCursor(QPoint pos)
{
    //snap to nearest delay point
    d_live = false;
    double mVal = canvasMap(QwtPlot::yLeft).invTransform(pos.y());
    int row = qBound(0,static_cast<int>(floor((mVal-d_dMin)/d_ddx)),p_spectrogramData->numRows()-1);
    moveDelayCursor(row);
    emit laserSlice(row);
}

void LifSpectrogramPlot::moveDelayCursor(int index)
{
    p_delayMarker->setYValue(delayVal(index));
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
    connect(delayCursorAction,&QAction::triggered,this,[this,pos](){ moveDelayCursor(pos); });

    QAction *freqCursorAction = menu->addAction(QString("Move frequency cursor here"));
    connect(freqCursorAction,&QAction::triggered,this,[this,pos](){ moveLaserCursor(pos); });

    QAction *bothCursorAction = menu->addAction(QString("Move both cursors here"));
    connect(bothCursorAction,&QAction::triggered,this,[this,pos](){ moveDelayCursor(pos); moveLaserCursor(pos); });

    auto liveAction = menu->addAction(QString("Follow live data"));
    connect(liveAction,&QAction::triggered,[this](){
        d_live = true;
        moveDelayCursor(d_liveDelayIndex);
        moveLaserCursor(d_liveLaserIndex);
        replot();
    });

    menu->popup(me->globalPos());
}

double LifSpectrogramPlot::getldx() const
{
    return d_ldx;
}

double LifSpectrogramPlot::getlMin() const
{
    return d_lMin;
}

double LifSpectrogramPlot::getddx() const
{
    return d_ddx;
}

double LifSpectrogramPlot::getdMin() const
{
    return d_dMin;
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
                        ev->accept();
                        return true;
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


                    int fPixel = canvasMap(QwtPlot::xBottom).transform(p_laserMarker->xValue());
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
                        moveLaserCursor(me->pos());

                    ev->accept();
                    replot();
                    return true;
                }
            }
        }
    }

    return ZoomPanPlot::eventFilter(obj,ev);
}
