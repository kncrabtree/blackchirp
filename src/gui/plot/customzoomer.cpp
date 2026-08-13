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

    // Rubber-band zooming is a left-button gesture only. ZoomPanPlot claims
    // the middle button for panning and the right button for the context
    // menu, and Qwt's default patterns would otherwise map those onto
    // zoom-stack navigation, which applies a stale stack rect to the axes.
    if (ev->button() != Qt::LeftButton)
        return false;

    // Modifiers are dropped so a modifier held for an unrelated purpose does
    // not suppress the zoom; this also keeps the shift/alt variants of the
    // left-button patterns from matching.
    const MousePattern mousePattern( ev->button(), Qt::NoModifier );
    return mousePattern == pattern;
}

// bool CustomZoomer::keyMatch(const KeyPattern &, const QKeyEvent *ke) const
// {
//     Q_UNUSED(ke)
//     return false;
// }
