#include "customzoomer.h"

#include <QPen>
#include <QPalette>
#include <QMouseEvent>
#include <QKeyEvent>

#include <qwt6/qwt_plot.h>

CustomZoomer::CustomZoomer(QwtAxisId x, QwtAxisId y, QWidget *canvas) :
    QwtPlotZoomer(x,y,canvas)
{
    QPalette p;
    setRubberBandPen(QPen(p.text().color()));
    setTrackerMode(QwtPicker::AlwaysOff);

    setKeyPattern(QwtEventPattern::KeyAbort,Qt::Key_Z);

}

CustomZoomer::~CustomZoomer()
{
}

QPolygon CustomZoomer::adjustedPoints(const QPolygon &p) const
{
    if(p.count() < 2)
        return p;

    auto r = plot()->canvas()->rect();
    auto rect = QRect(p.first(),p.last());

    if(d_xLocked)
    {
        rect.setLeft(r.left());
        rect.setRight(r.right());
    }

    if(d_yLocked)
    {
        rect.setBottom(r.bottom());
        rect.setTop(r.top());
    }

    auto out = QPolygon(QVector<QPoint>({rect.topLeft(),rect.bottomRight()}));
    return out;

}


bool CustomZoomer::mouseMatch(const MousePattern &pattern, const QMouseEvent *ev) const
{
    if (ev == nullptr)
        return false;

    const MousePattern mousePattern( ev->button(), Qt::NoModifier );
    return mousePattern == pattern;
}

// bool CustomZoomer::keyMatch(const KeyPattern &, const QKeyEvent *ke) const
// {
//     Q_UNUSED(ke)
//     return false;
// }
