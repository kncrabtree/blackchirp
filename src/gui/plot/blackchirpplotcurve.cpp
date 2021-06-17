#include "blackchirpplotcurve.h"

#include <QPalette>

BlackchirpPlotCurve::BlackchirpPlotCurve(const QString name,Qt::PenStyle defaultLineStyle, QwtSymbol::Style defaultMarker) :
    SettingsStorage({BC::Key::bcCurve,name},General,false)
{
    setTitle(name);

    getOrSetDefault(BC::Key::bcCurveStyle,static_cast<int>(defaultLineStyle));
    getOrSetDefault(BC::Key::bcCurveMarker,static_cast<int>(defaultMarker));

    configurePen();
    configureSymbol();
    setRenderHint(QwtPlotItem::RenderAntialiased);

    setAxes(get<QwtPlot::Axis>(BC::Key::bcCurveAxisX,QwtPlot::xBottom),
            get<QwtPlot::Axis>(BC::Key::bcCurveAxisY,QwtPlot::yLeft));
    setVisible(get<bool>(BC::Key::bcCurveVisible,true));

}

void BlackchirpPlotCurve::setColor(const QColor c)
{
    set(BC::Key::bcCurveColor,c);
    configurePen();
    configureSymbol();
}

void BlackchirpPlotCurve::setLineThickness(double t)
{
    set(BC::Key::bcCurveThickness,t);
    configurePen();
}

void BlackchirpPlotCurve::setLineStyle(Qt::PenStyle s)
{
    set(BC::Key::bcCurveStyle,static_cast<int>(s));
    configurePen();
}

void BlackchirpPlotCurve::setMarkerStyle(QwtSymbol::Style s)
{
    set(BC::Key::bcCurveMarker,static_cast<int>(s));
    configureSymbol();
}

void BlackchirpPlotCurve::setMarkerSize(int s)
{
    set(BC::Key::bcCurveMarkerSize,s);
    configureSymbol();
}

void BlackchirpPlotCurve::setCurveVisible(bool v)
{
    set(BC::Key::bcCurveVisible,v);
    setVisible(v);
}

void BlackchirpPlotCurve::setCurveAxisX(QwtPlot::Axis a)
{
    set(BC::Key::bcCurveAxisX,static_cast<int>(a));
    setXAxis(a);
}

void BlackchirpPlotCurve::setCurveAxisY(QwtPlot::Axis a)
{
    set(BC::Key::bcCurveAxisY,static_cast<int>(a));
    setYAxis(a);
}

void BlackchirpPlotCurve::setCurvePlotIndex(int i)
{
    set(BC::Key::bcCurvePlotIndex,i);
}

void BlackchirpPlotCurve::updateFromSettings()
{
    configurePen();
    configureSymbol();
    setAxes(get<QwtPlot::Axis>(BC::Key::bcCurveAxisX,QwtPlot::xBottom),
            get<QwtPlot::Axis>(BC::Key::bcCurveAxisY,QwtPlot::yLeft));
    setVisible(get<bool>(BC::Key::bcCurveVisible,true));
}

void BlackchirpPlotCurve::configurePen()
{
    QPen p;
    QPalette pal;
    p.setColor(get<QColor>(BC::Key::bcCurveColor,pal.color(QPalette::BrightText)));
    p.setWidthF(get<double>(BC::Key::bcCurveThickness,1.0));
    p.setStyle(get<Qt::PenStyle>(BC::Key::bcCurveStyle,Qt::SolidLine));
    setPen(p);
}

void BlackchirpPlotCurve::configureSymbol()
{
    auto sym = new QwtSymbol();
    QPalette pal;
    sym->setStyle(get<QwtSymbol::Style>(BC::Key::bcCurveMarker,QwtSymbol::NoSymbol));
    sym->setColor(get<QColor>(BC::Key::bcCurveColor,pal.color(QPalette::BrightText)));
    sym->setPen(get<QColor>(BC::Key::bcCurveColor,pal.color(QPalette::BrightText)));
    auto s = get<int>(BC::Key::bcCurveMarkerSize,5);
    sym->setSize(QSize(s,s));
    setSymbol(sym);
}
