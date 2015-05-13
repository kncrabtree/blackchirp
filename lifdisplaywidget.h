#ifndef LIFDISPLAYWIDGET_H
#define LIFDISPLAYWIDGET_H

#include <QWidget>

#include <QVector>

#include "datastructs.h"
#include "liftrace.h"
#include "lifconfig.h"

class QwtMatrixRasterData;

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
    void updatePoint(QPair<QPoint,BlackChirp::LifPoint> val);

private:
    Ui::LifDisplayWidget *ui;

    int d_numColumns;
    int d_numRows;
    bool d_delayReverse, d_freqReverse;
    QVector<BlackChirp::LifPoint> d_lifData;
    QwtMatrixRasterData *p_spectrogramData;
    double d_spectrogramZMax;

protected:
    void resizeEvent(QResizeEvent *ev);
};

#endif // LIFDISPLAYWIDGET_H
