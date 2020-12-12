#ifndef CUSTOMTRACKER_H
#define CUSTOMTRACKER_H

#include <QPalette>
#include <qwt6/qwt_plot_picker.h>

class CustomTracker : public QwtPlotPicker
{
    Q_OBJECT
public:
    CustomTracker(QWidget* canvas);

    virtual QwtText trackerTextF( const QPointF &pos ) const;
};

#endif // CUSTOMTRACKER_H
