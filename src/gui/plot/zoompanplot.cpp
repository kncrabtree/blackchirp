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

#include <qwt6/qwt_scale_div.h>
#include <qwt6/qwt_plot_marker.h>
#include <gui/plot/blackchirpplotcurve.h>

#include <gui/plot/customtracker.h>


ZoomPanPlot::ZoomPanPlot(const QString name, QWidget *parent) : QwtPlot(parent),
    SettingsStorage(name,SettingsStorage::General,false), d_name(name), d_maxIndex(0)
{
    //it is important for the axes to be appended in the order of their enum values
    d_config.axisList.append(AxisConfig(QwtPlot::yLeft,BC::Key::left));
    d_config.axisList.append(AxisConfig(QwtPlot::yRight,BC::Key::right));
    d_config.axisList.append(AxisConfig(QwtPlot::xBottom,BC::Key::bottom));
    d_config.axisList.append(AxisConfig(QwtPlot::xTop,BC::Key::top));

    for(auto d : d_config.axisList)
    {
        setAxisFont(d.type,QApplication::font());
        axisScaleEngine(d.type)->setAttribute(QwtScaleEngine::Floating);
    }

    p_tracker = new CustomTracker(this->canvas());

    if(!containsArray(BC::Key::axes))
    {
        std::vector<SettingsMap> l;
        for(int i=0; i<d_config.axisList.size(); i++)
            l.push_back({ {BC::Key::zoomFactor,0.1},
                          {BC::Key::trackerDecimals,4},
                          {BC::Key::trackerScientific,false}});
        setArray(BC::Key::axes,l);
    }

    for(int i=0; i<d_config.axisList.size(); i++)
    {
        d_config.axisList[i].zoomFactor = getArrayValue<double>(BC::Key::axes,i,BC::Key::zoomFactor,0.1);
        int dec = getArrayValue<int>(BC::Key::axes,i,BC::Key::trackerDecimals,4);
        bool sci = getArrayValue<bool>(BC::Key::axes,i,BC::Key::trackerScientific,false);
        p_tracker->setDecimals(d_config.axisList.at(i).type,dec);
        p_tracker->setScientific(d_config.axisList.at(i).type,sci);
    }

    bool en = getOrSetDefault(BC::Key::trackerEn,false);
    p_tracker->setEnabled(en);

    p_grid = new QwtPlotGrid;
    p_grid->enableX(true);
    p_grid->enableXMin(true);
    p_grid->enableY(true);
    p_grid->enableYMin(true);
    configureGridMajorPen();
    configureGridMinorPen();
    p_grid->attach(this);

    canvas()->installEventFilter(this);
    connect(this,&ZoomPanPlot::plotRightClicked,this,&ZoomPanPlot::buildContextMenu);
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

void ZoomPanPlot::setPlotTitle(const QString text)
{
    QwtText t(text);
    t.setFont(QApplication::font());
    setTitle(t);
}

void ZoomPanPlot::setPlotAxisTitle(QwtPlot::Axis a, const QString text)
{
    QwtText t(text);
    t.setFont(QApplication::font());
    setAxisTitle(a,t);
}

void ZoomPanPlot::replot()
{
    if(!isVisible())
        return;

    //figure out which axes to show
    QwtPlotItemList l = itemList();
    bool bottom = false, top = false, left = false, right = false;

    QRectF invalid{1.0,1.0,-2.0,-2.0};
    for(auto &a : d_config.axisList)
        a.boundingRect = invalid;

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
        auto c = dynamic_cast<BlackchirpPlotCurve*>(l.at(i));
        if(c)
        {
            auto r = c->boundingRect();
            if(r.width() < 0.0 || r.height() < 0.0 || !c->isVisible())
                continue;

            if(d_config.axisList.at(c->xAxis()).boundingRect.width() >=0.0)
                d_config.axisList[c->xAxis()].boundingRect |= r;
            else
                d_config.axisList[c->xAxis()].boundingRect = r;
            if(d_config.axisList.at(c->yAxis()).boundingRect.height() >=0.0)
                d_config.axisList[c->yAxis()].boundingRect |= r;
            else
                d_config.axisList[c->yAxis()].boundingRect = r;
        }

        auto m = dynamic_cast<QwtPlotMarker*>(l.at(i));
        if(m && m->testItemAttribute(QwtPlotItem::AutoScale))
        {
            auto r = m->boundingRect();

            if(r.width() >= 0.0)
            {
                if(d_config.axisList.at(m->xAxis()).boundingRect.width() >=0.0)
                    d_config.axisList[m->xAxis()].boundingRect |= r;
                else
                    d_config.axisList[m->xAxis()].boundingRect = r;
            }

            if(r.height() >= 0.0)
            {
                if(d_config.axisList.at(m->yAxis()).boundingRect.height() >=0.0)
                    d_config.axisList[m->yAxis()].boundingRect |= r;
                else
                    d_config.axisList[m->yAxis()].boundingRect = r;
            }

        }
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

            if((c.type == QwtPlot::xBottom) || (c.type == QwtPlot::xTop))
            {
                setAxisScale(c.type,c.boundingRect.left(),c.boundingRect.right());
                redrawXAxis = true;
            }
            else
            {
                setAxisScale(c.type,c.boundingRect.top(),c.boundingRect.bottom());
            }
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

    setArrayValue(BC::Key::axes,i,BC::Key::zoomFactor,v);
}

void ZoomPanPlot::setTrackerEnabled(bool en)
{
    set(BC::Key::trackerEn,en);
    p_tracker->setEnabled(en);
}

void ZoomPanPlot::setTrackerDecimals(QwtPlot::Axis a, int dec)
{
    int i = getAxisIndex(a);
    setArrayValue(BC::Key::axes,i,BC::Key::trackerDecimals,dec);

    p_tracker->setDecimals(a,dec);
}

void ZoomPanPlot::setTrackerScientific(QwtPlot::Axis a, bool sci)
{
    int i = getAxisIndex(a);
    setArrayValue(BC::Key::axes,i,BC::Key::trackerScientific,sci);

    p_tracker->setScientific(a,sci);
}

void ZoomPanPlot::setCurveColor(BlackchirpPlotCurve *curve)
{
    auto c = QColorDialog::getColor(curve->pen().color(),this,
                           QString("Choose a color for the ")+curve->title().text()+QString(" curve"));
    if(c.isValid())
        curve->setColor(c);
    replot();
}

void ZoomPanPlot::setCurveLineThickness(BlackchirpPlotCurve *curve, double t)
{
    curve->setLineThickness(t);
    replot();
}

void ZoomPanPlot::setCurveLineStyle(BlackchirpPlotCurve *curve, Qt::PenStyle s)
{
    curve->setLineStyle(s);
    replot();
}

void ZoomPanPlot::setCurveMarker(BlackchirpPlotCurve *curve, QwtSymbol::Style s)
{
    curve->setMarkerStyle(s);
    replot();
}

void ZoomPanPlot::setCurveMarkerSize(BlackchirpPlotCurve *curve, int s)
{
    curve->setMarkerSize(s);
    replot();
}

void ZoomPanPlot::setCurveVisible(BlackchirpPlotCurve *curve, bool v)
{
    curve->setCurveVisible(v);
    replot();
}

void ZoomPanPlot::setCurveAxisY(BlackchirpPlotCurve *curve, QwtPlot::Axis a)
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
    replot();

}

void ZoomPanPlot::configureGridMinorPen()
{
    QPalette p;
    auto c = get<QColor>(BC::Key::minorGridColor,p.color(QPalette::Light));
    auto s = get<Qt::PenStyle>(BC::Key::minorGridStyle,Qt::NoPen);
    p_grid->setMinorPen(c,0.0,s);
    replot();
}

void ZoomPanPlot::setAxisOverride(QwtPlot::Axis axis, bool override)
{
    d_config.axisList[getAxisIndex(axis)].override = override;
}

void ZoomPanPlot::filterData()
{
    auto l = itemList();
    for(int i=0; i<d_config.axisList.size(); ++i)
        d_config.axisList[i].boundingRect = QRectF{ QPointF{1.0,1.0}, QPointF{-2.0,-2.0} };
    for(auto item : l)
    {
        auto c = dynamic_cast<BlackchirpPlotCurve*>(item);
        if(c)
        {
            c->filter();
            auto r = c->boundingRect();
            if(r.width() <= 0.0 || r.height() <= 0.0)
                continue;

            if(d_config.axisList.at(c->yAxis()).boundingRect.height() >=0.0)
                d_config.axisList[c->yAxis()].boundingRect |= r;
            else
                d_config.axisList[c->yAxis()].boundingRect = r;

            if(d_config.axisList.at(c->xAxis()).boundingRect.width() >=0.0)
                d_config.axisList[c->xAxis()].boundingRect |= r;
            else
                d_config.axisList[c->xAxis()].boundingRect = r;
        }
    }
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

    for(auto c : d_config.axisList)
    {
        if(c.override)
            continue;

        double scaleMin = axisScaleDiv(c.type).lowerBound();
        double scaleMax = axisScaleDiv(c.type).upperBound();

        double d;
        bool xAxis = (c.type == QwtPlot::xBottom || c.type == QwtPlot::xTop);
        double min, max;
        if(xAxis)
        {
            d = (scaleMax - scaleMin)/(double)canvas()->width()*delta.x();
            min = c.boundingRect.left();
            max = c.boundingRect.right();
        }
        else
        {
            d = -(scaleMax - scaleMin)/(double)canvas()->height()*delta.y();
            min = c.boundingRect.top();
            max = c.boundingRect.bottom();
        }

        if(scaleMin + d < min)
            d = min - scaleMin;
        if(scaleMax + d > max)
            d = max - scaleMax;


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

        bool xAxis = (c.type == QwtPlot::xBottom || c.type == QwtPlot::xTop);
        double min,max;
        if(xAxis)
        {
            mousePosInt = we->pos().x();
            d_config.xDirty = true;
            min = c.boundingRect.left();
            max = c.boundingRect.right();
        }
        else
        {
            mousePosInt = we->pos().y();
            min = c.boundingRect.top();
            max = c.boundingRect.bottom();
        }

        double mousePos = qBound(scaleMin,canvasMap(c.type).invTransform(mousePosInt),scaleMax);

        scaleMin += qAbs(mousePos-scaleMin)*factor*(double)numSteps;
        scaleMax -= qAbs(mousePos-scaleMax)*factor*(double)numSteps;

//        scaleMin = qMax(scaleMin,min);
//        scaleMax = qMin(scaleMax,max);

        if(scaleMin <= min && scaleMax >= max)
            d_config.axisList[i].autoScale = true;
        else
        {
            d_config.axisList[i].autoScale = false;
            setAxisScale(c.type,qMax(min,scaleMin),qMin(max,scaleMax));
        }
    }

    replot();
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

    QMenu *zoomMenu = menu->addMenu(QString("Wheel zoom factor"));
    QWidgetAction *zwa = new QWidgetAction(zoomMenu);
    QWidget *zw = new QWidget(zoomMenu);
    QFormLayout *zfl = new QFormLayout(zw);


    QMenu *trackMenu = menu->addMenu(QString("Tracker"));
    QWidgetAction *twa = new QWidgetAction(trackMenu);
    QWidget *tw = new QWidget(trackMenu);
    QFormLayout *tfl = new QFormLayout(tw);

    QCheckBox *enBox = new QCheckBox();
    enBox->setChecked(p_tracker->isEnabled());
    connect(enBox,&QCheckBox::toggled,this,&ZoomPanPlot::setTrackerEnabled);
    tfl->addRow(QString("Enabled?"),enBox);

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

        auto zlbl = new QLabel(c.name);
        zlbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
        zlbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
        zfl->addRow(zlbl,box);

        QSpinBox *decBox = new QSpinBox;
        decBox->setRange(0,9);
        decBox->setValue(p_tracker->axisDecimals(c.type));
        decBox->setKeyboardTracking(false);
        connect(decBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,[=](int dec){ setTrackerDecimals(c.type,dec); });

        auto lbl = new QLabel(QString("%1 Decimals").arg(c.name));
        lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
        lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);
        tfl->addRow(lbl,decBox);

        QCheckBox *sciBox = new QCheckBox;
        sciBox->setChecked(p_tracker->axisScientific(c.type));
        connect(sciBox,&QCheckBox::toggled,this,[=](bool sci){ setTrackerScientific(c.type,sci); });

        auto lbl2 = new QLabel(QString("%1 Scientific").arg(c.name));
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
        set(BC::Key::majorGridStyle,majorPenBox->itemData(i));
        configureGridMajorPen();
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
        set(BC::Key::minorGridStyle,minorPenBox->itemData(i));
        configureGridMinorPen();
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
        auto curve = dynamic_cast<BlackchirpPlotCurve*>(item);
        if(curve != nullptr)
        {
            ++count;
            auto m = curveMenu->addMenu(curve->title().text());


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

            if(d_maxIndex > 0)
            {
                QMenu *moveMenu = m->addMenu(QString("Change plot"));
                QActionGroup *moveGroup = new QActionGroup(moveMenu);
                moveGroup->setExclusive(true);
                for(int j=0; j<d_maxIndex+1; j++)
                {
                    QAction *a = moveGroup->addAction(QString("Move to plot %1").arg(j+1));
                    a->setCheckable(true);
                    if(j == (curve->plotIndex() % (d_maxIndex+1)))
                    {
                        a->setEnabled(false);
                        a->setChecked(true);
                    }
                    else
                    {
                        connect(a,&QAction::triggered,this, [=](){ emit curveMoveRequested(curve,j); });
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

int ZoomPanPlot::getAxisIndex(QwtPlot::Axis a)
{
    return static_cast<int>(a);
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
