#ifndef DRSCANCONFIGWIDGET_H
#define DRSCANCONFIGWIDGET_H

#include <QWidget>
#include <data/storage/settingsstorage.h>

class Experiment;
class QDoubleSpinBox;
class QSpinBox;

namespace BC::Key::WizDR {
inline constexpr QLatin1StringView key{"WizardDrPage"};
inline constexpr QLatin1StringView title{"DR Scan"};
inline constexpr QLatin1StringView start{"startFreqMHz"};
inline constexpr QLatin1StringView step{"stepSizeMHz"};
inline constexpr QLatin1StringView numSteps{"numSteps"};
inline constexpr QLatin1StringView shots{"numShots"};
}

class DRScanConfigWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit DRScanConfigWidget(Experiment *exp, QWidget *parent = nullptr);

    void initialize();
    bool validate();
    void apply();

signals:
    void error(QString);
    void warning(QString);

public slots:
    void updateEndBox();

private:
    Experiment *p_exp;
    QDoubleSpinBox *p_startBox, *p_stepSizeBox, *p_endBox;
    QSpinBox *p_numStepsBox, *p_shotsBox;
};

#endif // DRSCANCONFIGWIDGET_H
