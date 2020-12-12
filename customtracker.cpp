#include "customtracker.h"

CustomTracker::CustomTracker(QWidget *canvas) : QwtPlotPicker(canvas)
{
}

QwtText CustomTracker::trackerTextF(const QPointF &pos) const
{

    QPalette p;
    QColor bg( p.window().color() );
    bg.setAlpha( 128 );

    QwtText text = QwtPlotPicker::trackerTextF( pos );
    text.setColor(p.text().color());
    text.setBackgroundBrush( QBrush( bg ) );
    return text;
}
