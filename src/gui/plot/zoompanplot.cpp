#include <gui/plot/zoompanplot.h>

#include <QApplication>
#include <QMouseEvent>
#include <QWidgetAction>
#include <QFormLayout>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QSpinBox>
#include <QLabel>
#include <QMenu>
#include <QColorDialog>
#include <QFileDialog>
#include <QSaveFile>
#include <QMessageBox>
#include <QtConcurrent/QtConcurrent>
#include <QMutexLocker>

#include <qwt6/qwt_scale_div.h>
#include <qwt6/qwt_plot_marker.h>
#include <qwt6/qwt_plot_spectrogram.h>
#include <gui/plot/blackchirpplotcurve.h>
#include <data/storage/blackchirpcsv.h>

#include <gui/plot/customtracker.h>


ZoomPanPlot::ZoomPanPlot(const QString name, QWidget *parent) : QwtPlot(parent),
    SettingsStorage(name,SettingsStorage::General), d_name(name), d_maxIndex(0), p_mutex{new QMutex}
{
    setAutoReplot(false);

    d_config.axisMap.insert({yLeft,AxisConfig(0,BC::Key::left)});
    d_config.axisMap.insert({yRight,AxisConfig(1,BC::Key::right)});
    d_config.axisMap.insert({xBottom,AxisConfig(2,BC::Key::bottom)});
    d_config.axisMap.insert({xTop,AxisConfig(3,BC::Key::top)});

    d_config.keyZoomYCenter = get(BC::Key::kzCenter,false);

    for(auto &[t,d] : d_config.axisMap)
        axisScaleEngine(t)->setAttribute(QwtScaleEngine::Floating);

    p_tracker = new CustomTracker(this->canvas());
    p_zoomerLB = new CustomZoomer(QwtPlot::xBottom,QwtPlot::yLeft,this->canvas());
    connect(p_zoomerLB,&QwtPlotZoomer::zoomed,[=](QRectF r) {
        zoom(r,xBottom,yLeft);
    });
    p_zoomerRT = new CustomZoomer(QwtPlot::xTop,QwtPlot::yRight,this->canvas());
    connect(p_zoomerRT,&QwtPlotZoomer::zoomed,[=](QRectF r) {
        zoom(r,xTop,yRight);
    });

    if(!containsArray(BC::Key::axes))
    {
        std::vector<SettingsMap> l;
        for(std::size_t i=0; i<d_config.axisMap.size(); i++)
            l.push_back({ {BC::Key::zoomFactor,0.1},
                          {BC::Key::trackerDecimals,4},
                          {BC::Key::trackerScientific,false}});
        setArray(BC::Key::axes,l);
    }

    for(auto &[t,d] : d_config.axisMap)
    {
        d.zoomFactor = getArrayValue<double>(BC::Key::axes,d.index,BC::Key::zoomFactor,0.1);
        int dec = getArrayValue<int>(BC::Key::axes,d.index,BC::Key::trackerDecimals,4);
        bool sci = getArrayValue<bool>(BC::Key::axes,d.index,BC::Key::trackerScientific,false);
        p_tracker->setDecimals(t,dec);
        p_tracker->setScientific(t,sci);
    }

    bool en = getOrSetDefault(BC::Key::trackerEn,false);
    p_tracker->setEnabled(en);

    p_grid = new QwtPlotGrid;
    p_grid->enableX(true);
    p_grid->enableXMin(true);
    p_grid->enableY(true);
    p_grid->enableYMin(true);
    p_grid->attach(this);
    configureGridMajorPen();
    configureGridMinorPen();

    canvas()->installEventFilter(this);
    connect(this,&ZoomPanPlot::plotRightClicked,this,&ZoomPanPlot::buildContextMenu);

    p_watcher = new QFutureWatcher<void>(this);
    connect(p_watcher,&QFutureWatcher<void>::finished,this,[this](){
        d_busy = false;
        QwtPlot::replot();
        if(d_config.xDirty)
        {
            updateAxes();
            QApplication::sendPostedEvents(this,QEvent::LayoutRequest);
            d_busy = true;
            d_config.xDirty = false;
            p_watcher->setFuture(QtConcurrent::run([this](){filterData();}));
        }
    },Qt::QueuedConnection);
}

ZoomPanPlot::~ZoomPanPlot()
{
    delete p_mutex;
}

bool ZoomPanPlot::isAutoScale()
{
    for(auto &[t,d] : d_config.axisMap)
    {
        if(d.autoScale == false)
            return false;
    }

    return true;
}

void ZoomPanPlot::resetPlot()
{
    detachItems();
    autoScale();
}

void ZoomPanPlot::setSpectrogramMode(bool b)
{
    d_config.spectrogramMode = b;
}

void ZoomPanPlot::autoScale()
{
    for(auto &[t,d] : d_config.axisMap)
        d.autoScale = true;

    d_config.xDirty = true;
    d_config.panning = false;

    replot();
}

void ZoomPanPlot::overrideAxisAutoScaleRange(QwtPlot::Axis a, double min, double max)
{
    auto &c = d_config.axisMap[a];
    c.overrideAutoScaleRange = true;
    if(a == QwtPlot::xTop || a == QwtPlot::xBottom)
    {
        c.overrideRect.setLeft(min);
        c.overrideRect.setRight(max);
    }
    else
    {
        c.overrideRect.setTop(min);
        c.overrideRect.setBottom(max);
    }

    replot();
}

void ZoomPanPlot::clearAxisAutoScaleOverride(QwtPlot::Axis a)
{
    auto &c = d_config.axisMap[a];
    c.overrideAutoScaleRange = false;
    c.overrideRect = {1.0,1.0,-2.0,-2.0};

    replot();
}

void ZoomPanPlot::setXRanges(const QwtScaleDiv &bottom, const QwtScaleDiv &top)
{
    setAxisScale(QwtPlot::xBottom,bottom.lowerBound(),bottom.upperBound());
    setAxisScale(QwtPlot::xTop,top.lowerBound(),top.upperBound());

    for(auto &[t,d] : d_config.axisMap)
    {
        if(t == QwtPlot::xBottom || t == QwtPlot::xTop)
            d.autoScale = false;
    }

    d_config.xDirty = true;
    replot();
}

void ZoomPanPlot::setPlotTitle(const QString text)
{
    QwtText t(text);
    setTitle(t);
}

void ZoomPanPlot::setPlotAxisTitle(QwtPlot::Axis a, const QString text)
{
    QwtText t(text);
    setAxisTitle(a,t);
}

void ZoomPanPlot::replot()
{
    if(!isVisible())
        return;

    //figure out which axes to show
    QwtPlotItemList l = itemList();
    bool bottom = false, top = false, left = false, right = false;

    p_mutex->lock();
    QRectF invalid{1.0,1.0,-2.0,-2.0};
    for(auto &[t,d] : d_config.axisMap)
        d.boundingRect = invalid;

    for(int i=0; i<l.size(); ++i)
    {
        if(l.at(i)->yAxis() == QwtPlot::yLeft)
            left = true;
        if(l.at(i)->yAxis() == QwtPlot::yRight)
            right = true;
        if(l.at(i)->xAxis() == QwtPlot::xBottom)
            bottom = true;
        if(l.at(i)->xAxis() == QwtPlot::xTop)
            top = true;

        //update bounding rects
        auto c = dynamic_cast<BlackchirpPlotCurveBase*>(l.at(i));
        if(c)
        {
            auto r = c->boundingRect();
            if(r.width() < 0.0 || r.height() < 0.0 || !c->isVisible())
                continue;

            auto &xa = d_config.axisMap[static_cast<Axis>(c->xAxis())];
            auto &ya = d_config.axisMap[static_cast<Axis>(c->yAxis())];

            if(xa.boundingRect.width() >=0.0)
                xa.boundingRect |= r;
            else
                xa.boundingRect = r;
            if(ya.boundingRect.height() >=0.0)
                ya.boundingRect |= r;
            else
                ya.boundingRect = r;
        }

        auto m = dynamic_cast<QwtPlotMarker*>(l.at(i));
        if(m && m->testItemAttribute(QwtPlotItem::AutoScale))
        {
            auto r = m->boundingRect();
            auto &xa = d_config.axisMap[static_cast<Axis>(m->xAxis())];
            auto &ya = d_config.axisMap[static_cast<Axis>(m->yAxis())];

            if(r.width() >= 0.0)
            {
                if(xa.boundingRect.width() >=0.0)
                    xa.boundingRect |= r;
                else
                    xa.boundingRect = r;
            }

            if(r.height() >= 0.0)
            {
                if(ya.boundingRect.height() >=0.0)
                    ya.boundingRect |= r;
                else
                    ya.boundingRect = r;
            }

        }

        auto sp = dynamic_cast<QwtPlotSpectrogram*>(l.at(i));
        if(sp)
        {
            d_config.axisMap[yLeft].boundingRect = sp->boundingRect();
            d_config.axisMap[xBottom].boundingRect = sp->boundingRect();
            auto in = sp->interval(Qt::ZAxis);
            d_config.axisMap[yRight].overrideRect.setBottom(in.maxValue());
            d_config.axisMap[yRight].overrideRect.setTop(in.minValue());
        }
    }

    if(!d_config.axisMap.at(yLeft).override)
        enableAxis(yLeft,left);
    if(!d_config.axisMap.at(yRight).override)
        enableAxis(yRight,right);
    if(!d_config.axisMap.at(xTop).override)
        enableAxis(xTop,top);
    if(!d_config.axisMap.at(xBottom).override)
        enableAxis(xBottom,bottom);

    if(!bottom || d_config.axisMap.at(xBottom).override)
        d_config.axisMap[xBottom].autoScale = true;
    if(!top || d_config.axisMap.at(xTop).override)
        d_config.axisMap[xTop].autoScale = true;
    if(!left || d_config.axisMap.at(yLeft).override)
        d_config.axisMap[yLeft].autoScale = true;
    if(!right || d_config.axisMap.at(yRight).override)
        d_config.axisMap[yRight].autoScale = true;


    bool redrawXAxis = false;
    QRectF zoomerLRect, zoomerRRect;
    for(auto &[t,d] : d_config.axisMap)
    {
        auto &r = zoomerLRect;
        if ((t == QwtPlot::xTop) || (t == QwtPlot::yRight))
            r = zoomerRRect;

        if((t == QwtPlot::xBottom) || (t == QwtPlot::xTop))
        {
            if(d.overrideAutoScaleRange)
            {
                if(d.autoScale)
                    setAxisScale(t,d.overrideRect.left(),d.overrideRect.right());
                r |= d.overrideRect;
            }
            else
            {
                r |= d.boundingRect;
                if(d.autoScale)
                {
                    if(d.boundingRect.width() < 0.0)
                        setAxisScale(t,0.0,1.0);
                    else
                        setAxisScale(t,d.boundingRect.left(),d.boundingRect.right());
                }
            }
            if(d.autoScale)
                redrawXAxis = true;
        }
        else
        {
            if(d.overrideAutoScaleRange)
            {
                r |= d.overrideRect;
                if(d.autoScale)
                    setAxisScale(t,d.overrideRect.top(),d.overrideRect.bottom());
            }
            else
            {
                r |= d.boundingRect;
                if(d.autoScale)
                {
                    if(d.boundingRect.height() < 0.0)
                        setAxisScale(t,0.0,1.0);
                    else
                        setAxisScale(t,d.boundingRect.top(),d.boundingRect.bottom());
                }
            }
        }
    }
    if (zoomerLRect != p_zoomerLB->zoomBase())
        p_zoomerLB->setZoomBase(zoomerLRect);
    if (zoomerRRect != p_zoomerRT->zoomBase())
        p_zoomerRT->setZoomBase(zoomerRRect);
    p_mutex->unlock();

    if(redrawXAxis)
    {
        updateAxes();
        QApplication::sendPostedEvents(this,QEvent::LayoutRequest);
        d_config.xDirty = true;
    }

    if(d_config.xDirty)
    {
        if(!d_busy)
        {
            d_config.xDirty = false;
            d_busy = true;
            p_watcher->setFuture(QtConcurrent::run([this](){filterData();}));
        }
    }
    else
        QwtPlot::replot();

}

void ZoomPanPlot::setZoomFactor(QwtPlot::Axis a, double v)
{
    auto &d = d_config.axisMap[a];
    d.zoomFactor = v;
    setArrayValue(BC::Key::axes,d.index,BC::Key::zoomFactor,v);
}

void ZoomPanPlot::setKeyZoomYCenter(bool en)
{
    set(BC::Key::kzCenter,en);
    d_config.keyZoomYCenter = en;
}

void ZoomPanPlot::setTrackerEnabled(bool en)
{
    set(BC::Key::trackerEn,en);
    p_tracker->setEnabled(en);
}

void ZoomPanPlot::setTrackerDecimals(QwtPlot::Axis a, int dec)
{
    auto &d = d_config.axisMap[a];
    setArrayValue(BC::Key::axes,d.index,BC::Key::trackerDecimals,dec);

    p_tracker->setDecimals(a,dec);
}

void ZoomPanPlot::setTrackerScientific(QwtPlot::Axis a, bool sci)
{
    auto &d = d_config.axisMap[a];
    setArrayValue(BC::Key::axes,d.index,BC::Key::trackerScientific,sci);

    p_tracker->setScientific(a,sci);
}

void ZoomPanPlot::exportCurve(BlackchirpPlotCurveBase *curve)
{
    QDir d = BlackchirpCSV::textExportDir();
    auto name = curve->name().append(".csv");
    auto saveFile = QFileDialog::getSaveFileName(nullptr,"Export XY Data",d.absoluteFilePath(name));
    if(saveFile.isEmpty())
        return;

    QSaveFile f(saveFile);
    if(!f.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        QMessageBox::critical(nullptr,"Export Error",QString("Could not open file %1 for writing.").arg(saveFile));
        return;
    }

    BlackchirpCSV::writeXY(f,curve->curveData(),name);
    f.commit();
}

void ZoomPanPlot::setCurveColor(BlackchirpPlotCurveBase *curve)
{
    auto c = QColorDialog::getColor(curve->pen().color(),this,
                           QString("Choose a color for the ")+curve->title().text()+QString(" curve"));
    if(c.isValid())
        curve->setColor(c);
    replot();
}

void ZoomPanPlot::setCurveLineThickness(BlackchirpPlotCurveBase *curve, double t)
{
    curve->setLineThickness(t);
    replot();
}

void ZoomPanPlot::setCurveLineStyle(BlackchirpPlotCurveBase *curve, Qt::PenStyle s)
{
    curve->setLineStyle(s);
    replot();
}

void ZoomPanPlot::setCurveMarker(BlackchirpPlotCurveBase *curve, QwtSymbol::Style s)
{
    curve->setMarkerStyle(s);
    replot();
}

void ZoomPanPlot::setCurveMarkerSize(BlackchirpPlotCurveBase *curve, int s)
{
    curve->setMarkerSize(s);
    replot();
}

void ZoomPanPlot::setCurveVisible(BlackchirpPlotCurveBase *curve, bool v)
{
    curve->setCurveVisible(v);
    replot();
}

void ZoomPanPlot::setCurveAxisY(BlackchirpPlotCurveBase *curve, QwtPlot::Axis a)
{
    curve->setCurveAxisY(a);
    replot();
}

void ZoomPanPlot::configureGridMajorPen()
{
    QPalette p;
    auto c = get<QColor>(BC::Key::majorGridColor,p.color(QPalette::Light));
    auto s = get<Qt::PenStyle>(BC::Key::majorGridStyle,Qt::NoPen);
    p_grid->setMajorPen(c,0.0,s);
}

void ZoomPanPlot::configureGridMinorPen()
{
    QPalette p;
    auto c = get<QColor>(BC::Key::minorGridColor,p.color(QPalette::Light));
    auto s = get<Qt::PenStyle>(BC::Key::minorGridStyle,Qt::NoPen);
    p_grid->setMinorPen(c,0.0,s);
}

void ZoomPanPlot::setAxisOverride(QwtPlot::Axis axis, bool override)
{
    d_config.axisMap[axis].override = override;
}

void ZoomPanPlot::filterData()
{
    if(d_config.spectrogramMode)
        return;

    auto l = itemList();
    p_mutex->lock();
    auto w = canvas()->width();
    p_mutex->unlock();

    for(auto item : l)
    {
        auto c = dynamic_cast<BlackchirpPlotCurveBase*>(item);
        if(c)
        {
            p_mutex->lock();
            auto map = canvasMap(c->xAxis());
            p_mutex->unlock();

            c->filter(w,map);
        }
    }

    p_mutex->lock();
    for(auto &[t,d] : d_config.axisMap)
        d.boundingRect = QRectF{ QPointF{1.0,1.0}, QPointF{-2.0,-2.0} };
    for(auto item : l)
    {
        auto c = dynamic_cast<BlackchirpPlotCurveBase*>(item);
        if(c)
        {
            auto r = c->boundingRect();
            if(r.width() <= 0.0 || r.height() <= 0.0)
                continue;

            auto &xa = d_config.axisMap[static_cast<Axis>(c->xAxis())];
            auto &ya = d_config.axisMap[static_cast<Axis>(c->yAxis())];

            if(ya.boundingRect.height() >=0.0)
                ya.boundingRect |= r;
            else
                ya.boundingRect = r;

            if(xa.boundingRect.width() >=0.0)
                xa.boundingRect |= r;
            else
                xa.boundingRect = r;
        }
    }
    p_mutex->unlock();
}

void ZoomPanPlot::resizeEvent(QResizeEvent *ev)
{
    for(auto &[t,d] : d_config.axisMap)
        setAxisFont(t,font());

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
                else if(me->button() == Qt::RightButton)
                {
                    emit plotRightClicked(me);
                    ev->accept();
                    return true;
                }
            }
        }
        else if(ev->type() == QEvent::MouseButtonDblClick)
        {
            autoScale();
            ev->accept();
            return true;
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
        else if(ev->type() == QEvent::KeyPress)
        {
            QKeyEvent *ke = dynamic_cast<QKeyEvent*>(ev);
            if(ke->key() == Qt::Key_Control)
            {
                p_zoomerLB->lockY(true);
                p_zoomerRT->lockY(true);
                d_config.zoomYLock = true;
            }
            if(ke->key() == Qt::Key_Shift)
            {
                p_zoomerLB->lockX(true);
                p_zoomerRT->lockX(true);
                d_config.zoomXLock = true;
            }
            if(ke->key() == Qt::Key_Home)
            {
                autoScale();
                ev->accept();
                return true;
            }
            if(ke->key() == Qt::Key_Up)
            {
                if(ke->modifiers() & Qt::ControlModifier)
                {
                    zoom(yLeft,1.0/1.5);
                    zoom(yRight,1.0/1.5);
                    ev->accept();
                    return true;
                }
                if(ke->modifiers() & Qt::ShiftModifier)
                {
                    zoom(yLeft,1.0/1.1);
                    zoom(yRight,1.0/1.1);
                    ev->accept();
                    return true;
                }
                if(ke->modifiers() & Qt::AltModifier)
                {
                    panV(0.1);
                    ev->accept();
                    return true;
                }
                panV(0.5);
                ev->accept();
                return true;
            }
            if(ke->key() == Qt::Key_Down)
            {
                if(ke->modifiers() & Qt::ControlModifier)
                {
                    zoom(yLeft,1.5);
                    zoom(yRight,1.5);
                    ev->accept();
                    return true;
                }
                if(ke->modifiers() & Qt::ShiftModifier)
                {
                    zoom(yLeft,1.1);
                    zoom(yRight,1.1);
                    ev->accept();
                    return true;
                }
                if(ke->modifiers() & Qt::AltModifier)
                {
                    panV(-0.1);
                    ev->accept();
                    return true;
                }
                panV(-0.5);
                ev->accept();
                return true;
            }
            if(ke->key() == Qt::Key_Left)
            {
                if(ke->modifiers() & Qt::ControlModifier)
                {
                    zoom(xBottom,1.5);
                    zoom(xTop,1.5);
                    ev->accept();
                    return true;
                }
                if(ke->modifiers() & Qt::ShiftModifier)
                {
                    zoom(xBottom,1.1);
                    zoom(xTop,1.1);
                    ev->accept();
                    return true;
                }
                if(ke->modifiers() & Qt::AltModifier)
                {
                    panH(-0.1);
                    ev->accept();
                    return true;
                }
                panH(-0.5);
                ev->accept();
                return true;
            }
            if(ke->key() == Qt::Key_Right)
            {
                if(ke->modifiers() & Qt::ControlModifier)
                {
                    zoom(xBottom,1.0/1.5);
                    zoom(xTop,1.0/1.5);
                    ev->accept();
                    return true;
                }
                if(ke->modifiers() & Qt::ShiftModifier)
                {
                    zoom(xBottom,1.0/1.1);
                    zoom(xTop,1.0/1.1);
                    ev->accept();
                    return true;
                }
                if(ke->modifiers() & Qt::AltModifier)
                {
                    panH(0.1);
                    ev->accept();
                    return true;
                }
                panH(0.5);
                ev->accept();
                return true;
            }

        }
        else if(ev->type() == QEvent::KeyRelease)
        {
            QKeyEvent *ke = dynamic_cast<QKeyEvent*>(ev);
            if(ke->key() == Qt::Key_Control)
            {
                p_zoomerLB->lockY(false);
                p_zoomerRT->lockY(false);
                d_config.zoomYLock = false;
            }
            if(ke->key() == Qt::Key_Shift)
            {
                p_zoomerLB->lockX(false);
                p_zoomerRT->lockX(false);
                d_config.zoomXLock = false;
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

    p_mutex->lock();
    for(const auto &[t,d] : d_config.axisMap)
    {
        if(d.override)
            continue;

        if(d_config.spectrogramMode && (t == QwtPlot::yRight || t == QwtPlot::xTop))
            continue;

        double scaleMin = axisScaleDiv(t).lowerBound();
        double scaleMax = axisScaleDiv(t).upperBound();

        double del;
        bool xAxis = ((t == QwtPlot::xBottom) || (t == QwtPlot::xTop));
        double min, max;
        if(xAxis)
        {
            del = (scaleMax - scaleMin)/(double)canvas()->width()*delta.x();
            min = d.boundingRect.left();
            max = d.boundingRect.right();
        }
        else
        {
            del = -(scaleMax - scaleMin)/(double)canvas()->height()*delta.y();
            min = d.boundingRect.top();
            max = d.boundingRect.bottom();
        }

        if(scaleMin + del < min)
            del = min - scaleMin;
        if(scaleMax + del > max)
            del = max - scaleMax;

        setAxisScale(t,scaleMin + del, scaleMax + del);
    }

    d_config.panClickPos = me->pos();
    p_mutex->unlock();

    replot();
}

void ZoomPanPlot::panH(double factor)
{
    auto br = getLimitRect(xBottom,yLeft);
    auto tr = getLimitRect(xTop,yRight);

    auto ll1 = br.left();
    auto ul1 = br.right();

    auto ll2 = tr.left();
    auto ul2 = tr.right();

    auto s1 = axisScaleDiv(xBottom);
    auto s2 = axisScaleDiv(xTop);

    auto d1 = qAbs(s1.upperBound() - s1.lowerBound())*factor;
    auto d2 = qAbs(s2.upperBound() - s2.lowerBound())*factor;

    if(s1.lowerBound() + d1 < ll1)
        d1 = ll1 - s1.lowerBound();
    if(s1.upperBound() + d1 > ul1)
        d1 = ul1 - s1.upperBound();

    if(s2.lowerBound() + d2 < ll2)
        d2 = ll2 - s2.lowerBound();
    if(s2.upperBound() + d2 > ul2)
        d2 = ul2 - s2.upperBound();

    setAxisScale(xBottom,s1.lowerBound() + d1,s1.upperBound() + d1);
    setAxisScale(xTop,s2.lowerBound() + d2,s2.upperBound() + d2);

    replot();
}

void ZoomPanPlot::panV(double factor)
{
    auto br = getLimitRect(xBottom,yLeft);
    auto tr = getLimitRect(xTop,yRight);

    auto ll1 = br.top();
    auto ul1 = br.bottom();

    auto ll2 = tr.top();
    auto ul2 = tr.bottom();

    auto s1 = axisScaleDiv(yLeft);
    auto s2 = axisScaleDiv(yRight);

    auto d1 = qAbs(s1.upperBound() - s1.lowerBound())*factor;
    auto d2 = qAbs(s2.upperBound() - s2.lowerBound())*factor;

    if(s1.lowerBound() + d1 < ll1)
        d1 = ll1 - s1.lowerBound();
    if(s1.upperBound() + d1 > ul1)
        d1 = ul1 - s1.upperBound();

    if(s2.lowerBound() + d2 < ll2)
        d2 = ll2 - s2.lowerBound();
    if(s2.upperBound() + d2 > ul2)
        d2 = ul2 - s2.upperBound();

    setAxisScale(yLeft,s1.lowerBound() + d1,s1.upperBound() + d1);
    setAxisScale(yRight,s2.lowerBound() + d2,s2.upperBound() + d2);

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
    auto mod = QApplication::keyboardModifiers();
    bool lockHorizontal = (mod & Qt::ShiftModifier) || (mod & Qt::AltModifier) || (mod & Qt::MetaModifier);
    bool lockLeft = (mod & Qt::ControlModifier) || (mod & Qt::AltModifier);
    bool lockRight = (mod & Qt::ControlModifier) || (mod & Qt::MetaModifier);

    //one step, which is 15 degrees, will zoom 10%
    //the delta function is in units of 1/8th a degree

    int numSteps = we->angleDelta().y()/8/15;
    if(numSteps == 0) //Qt might switch orientation of wheel event when alt is pressed
        numSteps = we->angleDelta().x()/8/15;

    p_mutex->lock();
    for(auto &[t,d] : d_config.axisMap)
    {
        if(d.override)
            continue;

        if(d_config.spectrogramMode && ((t == QwtPlot::yRight) || (t == QwtPlot::xTop)))
            continue;

        if(((t == QwtPlot::xBottom) || (t == QwtPlot::xTop)) && lockHorizontal)
            continue;
        if((t == QwtPlot::yLeft) && lockLeft)
            continue;
        if((t == QwtPlot::yRight) && lockRight)
            continue;

        double scaleMin = axisScaleDiv(t).lowerBound();
        double scaleMax = axisScaleDiv(t).upperBound();
        double factor = d.zoomFactor;
        int mousePosInt;

        bool xAxis = ((t == QwtPlot::xBottom) || (t == QwtPlot::xTop));
        double min,max;
        if(xAxis)
        {
            mousePosInt = we->position().x();
            d_config.xDirty = true;
            min = d.boundingRect.left();
            max = d.boundingRect.right();
        }
        else
        {
            mousePosInt = we->position().y();
            min = d.boundingRect.top();
            max = d.boundingRect.bottom();
        }

        double mousePos = qBound(scaleMin,canvasMap(t).invTransform(mousePosInt),scaleMax);

        scaleMin += qAbs(mousePos-scaleMin)*factor*(double)numSteps;
        scaleMax -= qAbs(mousePos-scaleMax)*factor*(double)numSteps;

        if(scaleMin > scaleMax)
            qSwap(scaleMin,scaleMax);

        if(scaleMin <= min && scaleMax >= max)
            d.autoScale = true;
        else
        {
            d.autoScale = false;
            setAxisScale(t,qMax(min,scaleMin),qMin(max,scaleMax));
        }
    }
    p_mutex->unlock();

    replot();
}

void ZoomPanPlot::zoom(const QRectF &rect, Axis xAx, Axis yAx)
{

    p_mutex->lock();
    auto xlock = d_config.zoomXLock;
    auto ylock = d_config.zoomYLock;
    d_config.axisMap[xAx].autoScale = false;
    d_config.axisMap[yAx].autoScale = false;
    p_mutex->unlock();

    auto r = getLimitRect(xAx,yAx) & rect;

    if(!xlock)
        setAxisScale(xAx,qMin(r.left(), r.right()),qMax(r.left(), r.right()));
    if(!ylock)
        setAxisScale(yAx,qMin(r.bottom(), r.top()),qMax(r.bottom(), r.top()));

    replot();
}

void ZoomPanPlot::zoom(Axis ax, double factor)
{
    double scaleMin = axisScaleDiv(ax).lowerBound();
    double scaleMax = axisScaleDiv(ax).upperBound();



    p_mutex->lock();
    d_config.axisMap[ax].autoScale = false;
    auto r = d_config.axisMap[ax].boundingRect;
    if(d_config.axisMap[ax].overrideAutoScaleRange)
        r = d_config.axisMap[ax].overrideRect;
    p_mutex->unlock();

    auto w = scaleMax - scaleMin;
    auto extend = w*(factor-1.0);
    auto newMin = scaleMin - extend;
    auto newMax = scaleMax + extend;

    if((ax == xBottom) || (ax==xTop))
    {
        newMin = qMax(r.left(),newMin);
        newMax = qMin(r.right(),newMax);
    }
    else
    {
        if(!d_config.keyZoomYCenter) //make zoom symmetric about 0
        {
            double corr = 2.0;
            if((r.top() < 0.0 && r.bottom() <= 0.0) || (r.bottom() > 0.0 && r.top()>=0.0))
                corr = 1.0;//correction if bounding rect is only positive or negative (eg ft plot)
            newMax = w/corr*(factor);
            newMin = -w/corr*(factor);
        }
        newMin = qMax(r.top(),newMin);
        newMax = qMin(r.bottom(),newMax);
    }

    setAxisScale(ax,newMin,newMax);

    replot();

}

QRectF ZoomPanPlot::getLimitRect(Axis xAx, Axis yAx) const
{
    QRectF LimitRect;
    p_mutex->lock();
    auto &xa = d_config.axisMap.at(xAx);
    auto &ya = d_config.axisMap.at(yAx);
    if(xa.overrideAutoScaleRange)
        LimitRect |= xa.overrideRect;
    else
        LimitRect |= xa.boundingRect;

    if(ya.overrideAutoScaleRange)
        LimitRect |= ya.overrideRect;
    else
        LimitRect |= ya.boundingRect;
    p_mutex->unlock();

    return LimitRect;
}

void ZoomPanPlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *m = contextMenu();
    m->popup(me->globalPos());
}

QMenu *ZoomPanPlot::contextMenu()
{
    QMenu *menu = new QMenu();
    menu->setAttribute(Qt::WA_DeleteOnClose);

    QAction *asAction = menu->addAction(QString("Autoscale"));
    connect(asAction,&QAction::triggered,this,&ZoomPanPlot::autoScale);

    QMenu *zoomMenu = menu->addMenu(QString("Zoom Settings"));
    QWidgetAction *zwa = new QWidgetAction(zoomMenu);
    QWidget *zw = new QWidget(zoomMenu);
    QFormLayout *zfl = new QFormLayout(zw);

    auto *kzBox = new QCheckBox();
    kzBox->setChecked(d_config.keyZoomYCenter);
    kzBox->setToolTip(QString("If checked, zooming with up/down arrows will be symmetric about the center of the plot.\nOtherwise, the zoom will be symmetric about 0."));
    connect(kzBox,&QCheckBox::toggled,this,&ZoomPanPlot::setKeyZoomYCenter);
    zfl->addRow(QString("Y Center?"),kzBox);

    auto *zlbl = new QLabel("Wheel Zoom Factors");
    zlbl->setAlignment(Qt::AlignCenter);
    zfl->addWidget(zlbl);

    QMenu *trackMenu = menu->addMenu(QString("Tracker"));
    QWidgetAction *twa = new QWidgetAction(trackMenu);
    QWidget *tw = new QWidget(trackMenu);
    QFormLayout *tfl = new QFormLayout(tw);

    QCheckBox *enBox = new QCheckBox();
    enBox->setChecked(p_tracker->isEnabled());
    connect(enBox,&QCheckBox::toggled,this,&ZoomPanPlot::setTrackerEnabled);
    tfl->addRow(QString("Enabled?"),enBox);

    for(const auto &[t,d] : d_config.axisMap)
    {
        if(!axisEnabled(t))
            continue;

        QDoubleSpinBox *box = new QDoubleSpinBox();
        box->setMinimum(0.001);
        box->setMaximum(0.5);
        box->setDecimals(3);
        box->setValue(d.zoomFactor);
        box->setSingleStep(0.005);
        box->setKeyboardTracking(false);
        connect(box,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),
                this,[=](double val){ setZoomFactor(t,val); });

        auto zlbl = new QLabel(d.name);
        zlbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
        zlbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
        zfl->addRow(zlbl,box);

        QSpinBox *decBox = new QSpinBox;
        decBox->setRange(0,9);
        decBox->setValue(p_tracker->axisDecimals(t));
        decBox->setKeyboardTracking(false);
        connect(decBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,[=](int dec){ setTrackerDecimals(t,dec); });

        auto lbl = new QLabel(QString("%1 Decimals").arg(d.name));
        lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
        lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
        tfl->addRow(lbl,decBox);

        QCheckBox *sciBox = new QCheckBox;
        sciBox->setChecked(p_tracker->axisScientific(t));
        connect(sciBox,&QCheckBox::toggled,this,[=](bool sci){ setTrackerScientific(t,sci); });

        auto lbl2 = new QLabel(QString("%1 Scientific").arg(d.name));
        lbl2->setAlignment(Qt::AlignRight|Qt::AlignCenter);
        lbl2->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
        tfl->addRow(lbl2,sciBox);

    }

    zw->setLayout(zfl);
    zwa->setDefaultWidget(zw);
    zoomMenu->addAction(zwa);

    tw->setLayout(tfl);
    twa->setDefaultWidget(tw);
    trackMenu->addAction(twa);

    auto gridMenu = menu->addMenu(QString("Grid"));
    auto majorColorAct = gridMenu->addAction(QString("Major Color..."));
    connect(majorColorAct,&QAction::triggered,[=](){
        auto c = QColorDialog::getColor(get<QColor>(BC::Key::majorGridColor,Qt::white),this,QString("Select major grid color"));
        if(c.isValid())
            set(BC::Key::majorGridColor,c,false);
        configureGridMajorPen();
        replot();
    });

    QWidgetAction *majorwa = new QWidgetAction(gridMenu);
    QWidget *majorgw = new QWidget(gridMenu);
    QFormLayout *majorfl = new QFormLayout(majorgw);

    QComboBox *majorPenBox = new QComboBox;
    majorPenBox->addItem(QString("None"),QVariant::fromValue(Qt::NoPen));
    majorPenBox->addItem(QString::fromUtf16(u"⸻ "),QVariant::fromValue(Qt::SolidLine));
    majorPenBox->addItem(QString("- - - "),QVariant::fromValue(Qt::DashLine));
    majorPenBox->addItem(QString::fromUtf16(u"· · · "),QVariant::fromValue(Qt::DotLine));
    majorPenBox->addItem(QString::fromUtf16(u"-·-·-"),QVariant::fromValue(Qt::DashDotLine));
    majorPenBox->addItem(QString::fromUtf16(u"-··-··"),QVariant::fromValue(Qt::DashDotDotLine));
    majorPenBox->setCurrentIndex(majorPenBox->findData(QVariant::fromValue(get<Qt::PenStyle>(BC::Key::majorGridStyle,Qt::NoPen))));
    connect(majorPenBox,static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),[=](int i){
        set(BC::Key::majorGridStyle,majorPenBox->itemData(i).toInt());
        configureGridMajorPen();
        replot();
    });
    majorfl->addRow(QString("Major Line Style"),majorPenBox);
    for(int i=0; i<majorfl->rowCount(); ++i)
    {
        auto lbl = qobject_cast<QLabel*>(majorfl->itemAt(i,QFormLayout::LabelRole)->widget());
        if(lbl)
        {
            lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
            lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
        }
    }

    majorgw->setLayout(majorfl);
    majorwa->setDefaultWidget(majorgw);
    gridMenu->addAction(majorwa);


    auto minorColorAct = gridMenu->addAction(QString("Minor Color..."));
    connect(minorColorAct,&QAction::triggered,[=](){
        auto c = QColorDialog::getColor(get<QColor>(BC::Key::minorGridColor,Qt::white),this,QString("Select Minor grid color"));
        if(c.isValid())
            set(BC::Key::minorGridColor,c,false);
        configureGridMinorPen();
        replot();
    });

    QWidgetAction *minorwa = new QWidgetAction(gridMenu);
    QWidget *minorgw = new QWidget(gridMenu);
    QFormLayout *minorfl = new QFormLayout(minorgw);

    QComboBox *minorPenBox = new QComboBox;
    minorPenBox->addItem(QString("None"),QVariant::fromValue(Qt::NoPen));
    minorPenBox->addItem(QString::fromUtf16(u"⸻ "),QVariant::fromValue(Qt::SolidLine));
    minorPenBox->addItem(QString("- - - "),QVariant::fromValue(Qt::DashLine));
    minorPenBox->addItem(QString::fromUtf16(u"· · · "),QVariant::fromValue(Qt::DotLine));
    minorPenBox->addItem(QString::fromUtf16(u"-·-·-"),QVariant::fromValue(Qt::DashDotLine));
    minorPenBox->addItem(QString::fromUtf16(u"-··-··"),QVariant::fromValue(Qt::DashDotDotLine));
    minorPenBox->setCurrentIndex(minorPenBox->findData(QVariant::fromValue(get<Qt::PenStyle>(BC::Key::minorGridStyle,Qt::NoPen))));
    connect(minorPenBox,static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),[=](int i){
        set(BC::Key::minorGridStyle,minorPenBox->itemData(i).toInt());
        configureGridMinorPen();
        replot();
    });
    minorfl->addRow(QString("Minor Line Style"),minorPenBox);
    for(int i=0; i<minorfl->rowCount(); ++i)
    {
        auto lbl = qobject_cast<QLabel*>(minorfl->itemAt(i,QFormLayout::LabelRole)->widget());
        if(lbl)
        {
            lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
            lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
        }
    }

    minorgw->setLayout(minorfl);
    minorwa->setDefaultWidget(minorgw);
    gridMenu->addAction(minorwa);


    auto curveMenu = menu->addMenu(QString("Curves"));
    int count = 0;
    for(auto item : itemList(QwtPlotItem::Rtti_PlotCurve))
    {
        auto curve = dynamic_cast<BlackchirpPlotCurveBase*>(item);
        if(curve != nullptr)
        {
            ++count;
            auto m = curveMenu->addMenu(curve->title().text());

            auto exportAct = m->addAction("Export XY");
            if(curve->curveData().isEmpty())
                exportAct->setEnabled(false);
            connect(exportAct,&QAction::triggered,[this,curve](){ exportCurve(curve); });


            auto colorAct = m->addAction(QString("Color..."));
            connect(colorAct,&QAction::triggered,this,[=](){ setCurveColor(curve); });

            auto curveWa = new QWidgetAction(m);
            auto curveWidget = new QWidget(m);
            auto cfl = new QFormLayout(curveWidget);

            auto thicknessBox = new QDoubleSpinBox;
            thicknessBox->setRange(0.0,10.0);
            thicknessBox->setDecimals(1);
            thicknessBox->setSingleStep(0.5);
            thicknessBox->setValue(curve->pen().widthF());
            connect(thicknessBox,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),
                    [=](double v){ setCurveLineThickness(curve, v); });
            cfl->addRow(QString("Line Width"),thicknessBox);

            auto lineStyleBox = new QComboBox;
            lineStyleBox->addItem(QString("None"),QVariant::fromValue(Qt::NoPen));
            lineStyleBox->addItem(QString::fromUtf16(u"⸻ "),QVariant::fromValue(Qt::SolidLine));
            lineStyleBox->addItem(QString("- - - "),QVariant::fromValue(Qt::DashLine));
            lineStyleBox->addItem(QString::fromUtf16(u"· · · "),QVariant::fromValue(Qt::DotLine));
            lineStyleBox->addItem(QString::fromUtf16(u"-·-·-"),QVariant::fromValue(Qt::DashDotLine));
            lineStyleBox->addItem(QString::fromUtf16(u"-··-··"),QVariant::fromValue(Qt::DashDotDotLine));
            lineStyleBox->setCurrentIndex(lineStyleBox->findData(QVariant::fromValue(curve->pen().style())));
            connect(lineStyleBox,static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
                    [=](int i){ setCurveLineStyle(curve,lineStyleBox->itemData(i).value<Qt::PenStyle>()); });
            cfl->addRow(QString("Line Style"),lineStyleBox);

            auto markerBox = new QComboBox;
            markerBox->addItem(QString("None"),QVariant::fromValue(QwtSymbol::NoSymbol));
            markerBox->addItem(QString::fromUtf16(u"●"),QVariant::fromValue(QwtSymbol::Ellipse));
            markerBox->addItem(QString::fromUtf16(u"■"),QVariant::fromValue(QwtSymbol::Rect));
            markerBox->addItem(QString::fromUtf16(u"⬥"),QVariant::fromValue(QwtSymbol::Diamond));
            markerBox->addItem(QString::fromUtf16(u"▲"),QVariant::fromValue(QwtSymbol::UTriangle));
            markerBox->addItem(QString::fromUtf16(u"▼"),QVariant::fromValue(QwtSymbol::DTriangle));
            markerBox->addItem(QString::fromUtf16(u"◀"),QVariant::fromValue(QwtSymbol::LTriangle));
            markerBox->addItem(QString::fromUtf16(u"▶"),QVariant::fromValue(QwtSymbol::RTriangle));
            markerBox->addItem(QString::fromUtf16(u"＋"),QVariant::fromValue(QwtSymbol::Cross));
            markerBox->addItem(QString::fromUtf16(u"⨯"),QVariant::fromValue(QwtSymbol::XCross));
            markerBox->addItem(QString::fromUtf16(u"—"),QVariant::fromValue(QwtSymbol::HLine));
            markerBox->addItem(QString::fromUtf16(u"︱"),QVariant::fromValue(QwtSymbol::VLine));
            markerBox->addItem(QString::fromUtf16(u"✳"),QVariant::fromValue(QwtSymbol::Star1));
            markerBox->addItem(QString::fromUtf16(u"✶"),QVariant::fromValue(QwtSymbol::Star2));
            markerBox->addItem(QString::fromUtf16(u"⬢"),QVariant::fromValue(QwtSymbol::Hexagon));
            markerBox->setCurrentIndex(markerBox->findData(QVariant::fromValue(curve->symbol()->style())));
            connect(markerBox,static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
                    [=](int i){ setCurveMarker(curve,markerBox->itemData(i).value<QwtSymbol::Style>()); });
            cfl->addRow(QString("Marker"),markerBox);

            auto markerSizeBox = new QSpinBox;
            markerSizeBox->setRange(1,20);
            markerSizeBox->setValue(curve->symbol()->size().width());
            connect(markerSizeBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),
                    [=](int s){ setCurveMarkerSize(curve,s); });
            cfl->addRow(QString("Marker Size"),markerSizeBox);

            auto visBox = new QCheckBox;
            visBox->setChecked(curve->isVisible());
            connect(visBox,&QCheckBox::toggled,[=](bool v){ setCurveVisible(curve,v); });
            cfl->addRow(QString("Visible"),visBox);

            for(int i=0; i<cfl->rowCount(); ++i)
            {
                auto lbl = qobject_cast<QLabel*>(cfl->itemAt(i,QFormLayout::LabelRole)->widget());
                if(lbl)
                {
                    lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
                    lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
                }
            }

            curveWidget->setLayout(cfl);
            curveWa->setDefaultWidget(curveWidget);
            m->addAction(curveWa);


            m->addSection(QString("Y Axis"));
            QActionGroup *axisGroup = new QActionGroup(menu);
            axisGroup->setExclusive(true);
            QAction *lAction = axisGroup->addAction(QString("Left"));
            QAction *rAction = axisGroup->addAction(QString("Right"));
            lAction->setCheckable(true);
            rAction->setCheckable(true);
            if(curve->yAxis() == QwtPlot::yLeft)
            {
                lAction->setEnabled(false);
                lAction->setChecked(true);
                connect(rAction,&QAction::triggered,this,[=](){ setCurveAxisY(curve,QwtPlot::yRight); });
            }
            else
            {
                rAction->setEnabled(false);
                rAction->setChecked(true);
                connect(lAction,&QAction::triggered,this,[=](){ setCurveAxisY(curve,QwtPlot::yLeft); });
            }
            m->addActions(axisGroup->actions());

            auto c = dynamic_cast<BlackchirpPlotCurve*>(curve);
            if(c && d_maxIndex > 0)
            {
                QMenu *moveMenu = m->addMenu(QString("Change plot"));
                QActionGroup *moveGroup = new QActionGroup(moveMenu);
                moveGroup->setExclusive(true);
                for(int j=0; j<d_maxIndex+1; j++)
                {
                    QAction *a = moveGroup->addAction(QString("Move to plot %1").arg(j+1));
                    a->setCheckable(true);
                    if(j == (qMax(c->plotIndex(),0) % (d_maxIndex+1)))
                    {
                        a->setEnabled(false);
                        a->setChecked(true);
                    }
                    else
                    {
                        connect(a,&QAction::triggered,this, [=](){ emit curveMoveRequested(c,j); });
                        a->setChecked(false);
                    }
                    moveMenu->addActions(moveGroup->actions());
                }
            }

        }
    }
    if(count == 0)
        curveMenu->setEnabled(false);

    return menu;

}

QSize ZoomPanPlot::sizeHint() const
{
    return QSize(150,100);
}

QSize ZoomPanPlot::minimumSizeHint() const
{
    return QSize(150,100);
}


void ZoomPanPlot::showEvent(QShowEvent *event)
{
    replot();
    QWidget::showEvent(event);
}
