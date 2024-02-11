#ifndef CUSTOMZOOMER_H
#define CUSTOMZOOMER_H

#include <qwt6/qwt_plot_zoomer.h>


class CustomZoomer : public QwtPlotZoomer
{
public:
    CustomZoomer(QwtAxisId x, QwtAxisId y, QWidget *canvas);
    virtual ~CustomZoomer();

    void lockY( bool l ) { d_yLocked = l; }
    void lockX( bool l ) { d_xLocked = l; }

    bool xLocked() const { return d_xLocked; }
    bool yLocked() const { return d_yLocked; }

    // QwtPicker interface
protected:
    QPolygon adjustedPoints(const QPolygon &p) const override;

    bool d_yLocked { false };
    bool d_xLocked { false };

    // QwtEventPattern interface
protected:
    bool mouseMatch(const MousePattern &pattern, const QMouseEvent *ev) const override;
    // bool keyMatch(const KeyPattern &, const QKeyEvent *ke) const override;
};

#endif // CUSTOMZOOMER_H
