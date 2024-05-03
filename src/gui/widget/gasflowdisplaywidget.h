#ifndef GASFLOWDISPLAYWIDGET_H
#define GASFLOWDISPLAYWIDGET_H

#include "hardwarestatusbox.h"

class QLabel;
class QDoubleSpinBox;
class Led;

using FlowWidgets = std::tuple<QLabel*,QDoubleSpinBox*,Led*>;

class GasFlowDisplayBox : public HardwareStatusBox
{
    Q_OBJECT
public:
    explicit GasFlowDisplayBox(const QString key, QWidget *parent = nullptr);

public slots:
    void applySettings();
    void updateFlow(const QString key, int ch, double val);
    void updateFlowName(const QString key, int ch, const QString name);
    void updateFlowSetpoint(const QString key, int ch, double val);
    void updatePressureControl(const QString key, bool en);
    void updatePressure(const QString key, double p);

signals:

private:
    QVector<FlowWidgets> d_flowWidgets;
    QDoubleSpinBox *p_pressureBox;
    Led *p_pressureLed;

};

#endif // GASFLOWDISPLAYWIDGET_H
