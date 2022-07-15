#ifndef PULSECONFIGWIDGET_H
#define PULSECONFIGWIDGET_H

#include <QWidget>
#include <QList>

#include <gui/widget/enumcombobox.h>
#include <data/storage/settingsstorage.h>
#include <hardware/optional/pulsegenerator/pulsegenconfig.h>
#include <data/experiment/ftmwconfig.h>

#ifdef BC_LIF
#include <modules/lif/data/lifconfig.h>
#endif

class QLabel;
class QDoubleSpinBox;
class QPushButton;
class QToolButton;
class QLineEdit;
class QComboBox;
class PulsePlot;
class QSpinBox;

namespace BC::Key::PulseWidget {
static const QString key{"PulseWidget"};
static const QString name{"name"};
static const QString channels{"channels"};
static const QString delayStep{"delayStepUs"};
static const QString widthStep{"widthStepUs"};
static const QString role{"role"};
}

class PulseConfigWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit PulseConfigWidget(QWidget *parent = 0);
    ~PulseConfigWidget();

    struct ChWidgets {
        QLabel *label;
        QComboBox *syncBox;
        QDoubleSpinBox *delayBox;
        QDoubleSpinBox *widthBox;
        EnumComboBox<PulseGenConfig::ChannelMode> *modeBox;
        QPushButton *onButton;
        QToolButton *cfgButton;
        QLineEdit *nameEdit;
        EnumComboBox<PulseGenConfig::Role> *roleBox;
        EnumComboBox<PulseGenConfig::ActiveLevel> *levelBox;
        QSpinBox *dutyOnBox;
        QSpinBox *dutyOffBox;
        QDoubleSpinBox *delayStepBox;
        QDoubleSpinBox *widthStepBox;
        bool locked{false};
    };

    PulseGenConfig getConfig() const;

    void configureForWizard();

    void configureFtmw(const FtmwConfig &c);
#ifdef BC_LIF
    void configureLif(const LifConfig &c);
#endif

signals:
    void changeSetting(int,PulseGenConfig::Setting,QVariant);
    void changeRepRate(double);
    void changeSysPulsing(bool);
    void changeSysMode(PulseGenConfig::PGenMode);

public slots:
    void launchChannelConfig(int ch);
    void newSetting(int index,PulseGenConfig::Setting s,QVariant val);
    void setFromConfig(const PulseGenConfig &c);
    void newRepRate(double r);
    void updateFromSettings();
    void setRepRate(const double r);
    void unlockAll();

private:
    void lockChannel(int i, bool locked = true);
    QList<ChWidgets> d_widgetList;
    PulseGenConfig d_config;
    PulsePlot *p_pulsePlot;
    QDoubleSpinBox *p_repRateBox;
    QPushButton *p_sysOnOffButton;
    EnumComboBox<PulseGenConfig::PGenMode> *p_sysModeBox;




    // QWidget interface
public:
    QSize sizeHint() const override;
};

#endif // PULSECONFIGWIDGET_H
