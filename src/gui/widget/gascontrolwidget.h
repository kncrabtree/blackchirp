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
static const QString key{"GasControlWidget"};
static const QString channels{"channels"};
static const QString gasName{"name"};
}

class GasControlWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit GasControlWidget(const FlowConfig &cfg, QWidget *parent = nullptr);
    ~GasControlWidget() {}
    FlowConfig &toConfig();

public slots:
    void applySettings();
    void updateGasSetpoint(const QString key, int i, double sp);
    void updatePressureSetpoint(const QString key, double sp);
    void updatePressureControl(const QString key, bool en);

signals:
    void nameUpdate(QString,int,QString);
    void gasSetpointUpdate(QString,int,double);
    void pressureSetpointUpdate(QString,double);
    void pressureControlUpdate(QString,bool);

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
