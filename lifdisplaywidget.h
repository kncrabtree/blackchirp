#ifndef LIFDISPLAYWIDGET_H
#define LIFDISPLAYWIDGET_H

#include <QWidget>

#include <QVector>

#include "datastructs.h"
#include "liftrace.h"
#include "lifconfig.h"

namespace Ui {
class LifDisplayWidget;
}

class LifDisplayWidget : public QWidget
{
    Q_OBJECT

public:
    explicit LifDisplayWidget(QWidget *parent = 0);
    ~LifDisplayWidget();

signals:
    void lifColorChanged();

public slots:
    void checkLifColors();
    void resetLifPlot();
    void lifShotAcquired(const LifTrace t);
    void prepareForExperiment(const LifConfig c);

private:
    Ui::LifDisplayWidget *ui;

    int d_numColumns;
    int d_numRows;
    QVector<BlackChirp::LifPoint> d_lifData;

protected:
    void resizeEvent(QResizeEvent *ev);
};

#endif // LIFDISPLAYWIDGET_H
