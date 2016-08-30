#ifndef MOTORZPLOT_H
#define MOTORZPLOT_H

#include "motorspectrogramplot.h"

class MotorZPlot : public MotorSpectrogramPlot
{
    Q_OBJECT
public:
    MotorZPlot(QWidget *parent = nullptr);

    // MotorSpectrogramPlot interface
    void buildContextMenu(QMouseEvent *me);
    void prepareForScan(const MotorScan s);

public slots:
    void updateData(QVector<double> data, int cols, double max, MotorScan::MotorDataAxis leftAxis, MotorScan::MotorDataAxis bottomAxis);

private:
    QwtMatrixRasterData *p_yzData;
    QwtMatrixRasterData *p_xzData;

};

#endif // MOTORZPLOT_H
