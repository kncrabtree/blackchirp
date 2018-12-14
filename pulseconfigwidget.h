#ifndef PULSECONFIGWIDGET_H
#define PULSECONFIGWIDGET_H

#include <QWidget>
#include <QList>

#include "pulsegenconfig.h"
#include "lifconfig.h"
#include "ftmwconfig.h"

class QLabel;
class QDoubleSpinBox;
class QPushButton;
class QToolButton;
class QLineEdit;
class QComboBox;

namespace Ui {
class PulseConfigWidget;
}

class PulseConfigWidget : public QWidget
{
    Q_OBJECT

public:
    explicit PulseConfigWidget(QWidget *parent = 0);
    ~PulseConfigWidget();

    struct ChWidgets {
        QLabel *label;
        QDoubleSpinBox *delayBox;
        QDoubleSpinBox *widthBox;
        QPushButton *onButton;
        QToolButton *cfgButton;
        QLineEdit *nameEdit;
        QPushButton *levelButton;
        QDoubleSpinBox *delayStepBox;
        QDoubleSpinBox *widthStepBox;
        QComboBox *roleBox;
    };

    PulseGenConfig getConfig() const;

    void configureForWizard();

    void configureFtmw(const FtmwConfig c);
#ifdef BC_LIF
    void configureLif(const LifConfig c);
#endif

signals:
    void changeSetting(int,BlackChirp::PulseSetting,QVariant);
    void changeRepRate(double);

public slots:
    void launchChannelConfig(int ch);
    void newSetting(int index,BlackChirp::PulseSetting s,QVariant val);
    void setFromConfig(const PulseGenConfig c);
    void newRepRate(double r);
    void updateHardwareLimits();
    void setRepRate(const double r);

private:
    Ui::PulseConfigWidget *ui;

    QList<ChWidgets> d_widgetList;
    PulseGenConfig d_config;


};

#endif // PULSECONFIGWIDGET_H
