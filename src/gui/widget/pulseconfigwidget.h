#ifndef PULSECONFIGWIDGET_H
#define PULSECONFIGWIDGET_H

#include <QWidget>
#include <QList>

#include <data/storage/settingsstorage.h>
#include <data/experiment/pulsegenconfig.h>
#include <modules/lif/data/lifconfig.h>
#include <data/experiment/ftmwconfig.h>

class QLabel;
class QDoubleSpinBox;
class QPushButton;
class QToolButton;
class QLineEdit;
class QComboBox;

namespace Ui {
class PulseConfigWidget;
}

namespace BC::Key::PulseWidget {
static const QString key{"PulseWidget"};
static const QString name{"name"};
static const QString channels("channels");
static const QString delayStep{"delayStepUs"};
static const QString widthStep("widthStepUs");
static const QString role("role");
}

class PulseConfigWidget : public QWidget, public SettingsStorage
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
    void changeSetting(int,PulseGenConfig::Setting,QVariant);
    void changeRepRate(double);

public slots:
    void launchChannelConfig(int ch);
    void newSetting(int index,PulseGenConfig::Setting s,QVariant val);
    void setFromConfig(const PulseGenConfig c);
    void newRepRate(double r);
    void updateFromSettings();
    void setRepRate(const double r);

private:
    Ui::PulseConfigWidget *ui;

    QList<ChWidgets> d_widgetList;
    PulseGenConfig d_config;


};

#endif // PULSECONFIGWIDGET_H
