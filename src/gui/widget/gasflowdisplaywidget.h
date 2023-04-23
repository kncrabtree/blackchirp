#ifndef GASFLOWDISPLAYWIDGET_H
#define GASFLOWDISPLAYWIDGET_H

#include <QGroupBox>

class QLabel;
class QDoubleSpinBox;
class Led;

using FlowWidgets = std::tuple<QLabel*,QDoubleSpinBox*,Led*>;

class GasFlowDisplayBox : public QGroupBox
{
    Q_OBJECT
public:
    explicit GasFlowDisplayBox(QWidget *parent = nullptr);

public slots:
    void applySettings();
    void updateFlow(int ch, double val);
    void updateFlowName(int ch, const QString name);
    void updateFlowSetpoint(int ch, double val);
    void updatePressureControl(bool en);
    void updatePressure(double p);

signals:

private:
    QVector<FlowWidgets> d_flowWidgets;
    QDoubleSpinBox *p_pressureBox;
    Led *p_pressureLed;

};

#endif // GASFLOWDISPLAYWIDGET_H
