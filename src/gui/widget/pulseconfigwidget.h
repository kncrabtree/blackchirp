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
    explicit PulseConfigWidget(const PulseGenConfig &cfg, QWidget *parent = 0);
    ~PulseConfigWidget();

    bool d_wizardOk{true};

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

    std::shared_ptr<PulseGenConfig> getConfig() const;

    void configureForWizard();

    void configureFtmw(const FtmwConfig &c);
#ifdef BC_LIF
    void configureLif(const LifConfig &c);
#endif

signals:
    void changeSetting(QString,int,PulseGenConfig::Setting,QVariant);

public slots:
    void launchChannelConfig(int ch);
    void setFromConfig(QString key, const PulseGenConfig &c);
    void newSetting(QString key, int index, PulseGenConfig::Setting s, QVariant val);
    void updateFromSettings();
    void unlockAll();

private:
    QString d_key;
    bool d_wizardMode{false};
    void lockChannel(int i, bool locked = true);
    void updateRoles();
    QList<ChWidgets> d_widgetList;
    PulsePlot *p_pulsePlot;
    QDoubleSpinBox *p_repRateBox;
    QPushButton *p_sysOnOffButton;
    EnumComboBox<PulseGenConfig::PGenMode> *p_sysModeBox;

    std::shared_ptr<PulseGenConfig> ps_config;


    // QWidget interface
public:
    QSize sizeHint() const override;
};

#endif // PULSECONFIGWIDGET_H
