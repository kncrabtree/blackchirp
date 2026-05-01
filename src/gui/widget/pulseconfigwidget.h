#ifndef PULSECONFIGWIDGET_H
#define PULSECONFIGWIDGET_H

#include <QWidget>
#include <QList>

#include <gui/widget/enumcombobox.h>
#include <data/storage/settingsstorage.h>
#include <data/experiment/hardware/optional/pulsegenerator/pulsegenconfig.h>
#include <data/experiment/ftmwconfig.h>

#include <data/lif/lifconfig.h>

class QLabel;
class QDoubleSpinBox;
class QPushButton;
class QToolButton;
class QLineEdit;
class QComboBox;
class PulsePlot;
class QSpinBox;

namespace BC::Key::PulseWidget {
inline constexpr QLatin1StringView key{"PulseWidget"};
inline constexpr QLatin1StringView channels{"channels"};
inline constexpr QLatin1StringView delayStep{"delayStepUs"};
inline constexpr QLatin1StringView widthStep{"widthStepUs"};
}

class PulseConfigWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit PulseConfigWidget(const PulseGenConfig &cfg, QWidget *parent = 0);
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

    const PulseGenConfig &getConfig() const;

    void configureForWizard();

signals:
    void changeSetting(const QString&, int, PulseGenConfig::Setting, QVariant);

public slots:
    void launchChannelConfig(int ch);
    void setFromConfig(const QString &key, const PulseGenConfig &c);
    void newSetting(const QString &key, int index, PulseGenConfig::Setting s, QVariant val);
    void updateFromSettings();

private:
    QString d_key;
    bool d_wizardMode{false};
    QList<ChWidgets> d_widgetList;
    PulsePlot *p_pulsePlot;
    QDoubleSpinBox *p_repRateBox;
    QPushButton *p_sysOnOffButton;
    EnumComboBox<PulseGenConfig::PGenMode> *p_sysModeBox;

    std::shared_ptr<PulseGenConfig> ps_config;
};

#endif // PULSECONFIGWIDGET_H
