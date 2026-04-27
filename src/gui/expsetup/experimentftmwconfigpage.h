#ifndef EXPERIMENTFTMWCONFIGPAGE_H
#define EXPERIMENTFTMWCONFIGPAGE_H

#include "experimentconfigpage.h"

class FtmwConfigWidget;
class RfConfigWidget;

namespace BC::Key::WizFtmw {
inline constexpr QLatin1StringView key{"WizardFtmwConfigPage"};
inline constexpr QLatin1StringView title{"FTMW Configuration"};
}

class ExperimentFtmwConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentFtmwConfigPage(Experiment *exp,
                             const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &clocks,
                             QWidget *parent = nullptr);

    RfConfigWidget *rfConfigWidget() const;

public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;
    void commitFtmwPreset();

signals:
    void presetChanged();

private:
    FtmwConfigWidget *p_widget;
};

#endif // EXPERIMENTFTMWCONFIGPAGE_H
