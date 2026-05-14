#ifndef PULSECONFIGWIDGET_H
#define PULSECONFIGWIDGET_H

#include <QWidget>
#include <QList>

#include <gui/widget/enumcombobox.h>
#include <data/storage/settingsstorage.h>
#include <data/experiment/hardware/optional/pulsegenerator/pulsegenconfig.h>
#include <data/experiment/ftmwconfig.h>

#include <data/lif/lifconfig.h>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QPushButton;
class QSpinBox;
class QTableWidget;
class QToolButton;
class PulsePlot;

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
    explicit PulseConfigWidget(const PulseGenConfig &cfg, QWidget *parent = nullptr);
    ~PulseConfigWidget();

    struct ChWidgets {
        QComboBox *syncBox;
        QDoubleSpinBox *delayBox;
        QDoubleSpinBox *widthBox;
        EnumComboBox<PulseGenConfig::ChannelMode> *modeBox;
        QToolButton *onButton;
        EnumComboBox<PulseGenConfig::Role> *roleBox;
        QCheckBox *invBox;
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
    void setFromConfig(const QString &key, const PulseGenConfig &c);
    void newSetting(const QString &key, int index, PulseGenConfig::Setting s, QVariant val);
    void updateFromSettings();

public:
    QSize sizeHint() const override;

private:
    QString d_key;
    bool d_wizardMode{false};
    QList<ChWidgets> d_widgetList;
    QGroupBox *p_mainGb{nullptr};
    QTableWidget *p_standardTable{nullptr};
    QTableWidget *p_advancedTable{nullptr};
    PulsePlot *p_pulsePlot{nullptr};
    QDoubleSpinBox *p_repRateBox;
    QPushButton *p_sysOnOffButton;
    EnumComboBox<PulseGenConfig::PGenMode> *p_sysModeBox;

    void applyChannelName(int ch, const QString &name);

    std::shared_ptr<PulseGenConfig> ps_config;
};

#endif // PULSECONFIGWIDGET_H
