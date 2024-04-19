#ifndef GASCONTROLWIDGET_H
#define GASCONTROLWIDGET_H

#include <QWidget>
#include <QVector>

#include <data/storage/settingsstorage.h>
#include <hardware/optional/flowcontroller/flowconfig.h>

class QLineEdit;
class QDoubleSpinBox;
class QPushButton;
class QLabel;

using GasWidgets = std::tuple<QLineEdit*,QDoubleSpinBox*>;

namespace BC::Key::GasControl {
static const QString key{"GasControlWidget.%1.%2"};
static const QString channels{"channels"};
static const QString gasName{"name"};
}

class GasControlWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit GasControlWidget(const FlowConfig &cfg, QWidget *parent = nullptr);
    ~GasControlWidget() {}
    FlowConfig &getFlowConfig();

public slots:
    void applySettings();
    void updateGasSetpoint(int i, double sp);
    void updatePressureSetpoint(double sp);
    void updatePressureControl(bool en);

signals:
    void nameUpdate(int,QString);
    void gasSetpointUpdate(int,double);
    void pressureSetpointUpdate(double);
    void pressureControlUpdate(bool);

private:
    void initialize(const FlowConfig &cfg);

    FlowConfig d_config;
    QVector<GasWidgets> d_widgets;
    QDoubleSpinBox *p_pressureSetpointBox;
    QPushButton* p_pressureControlButton;


    // QWidget interface
public:
    QSize sizeHint() const override;
};

#endif // GASCONTROLWIDGET_H
