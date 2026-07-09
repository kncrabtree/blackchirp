#ifndef EXPERIMENTLIFCONFIGPAGE_H
#define EXPERIMENTLIFCONFIGPAGE_H

#include <gui/expsetup/experimentconfigpage.h>

class LifConfigWidget;
class LifControlWidget;
class LifConversionWidget;

namespace BC::Key::WizLif {
inline constexpr QLatin1StringView key{"WizardLifConfigPage"};
inline constexpr QLatin1StringView title{"LIF Configuration"};
}

class ExperimentLifConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentLifConfigPage(Experiment *exp, QWidget *parent = nullptr);

    LifControlWidget *lifControlWidget();
    LifConversionWidget *lifConversionWidget();

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;
    void commitLifPreset();

signals:
    //! Forwarded from LifConversionWidget::edited(): the conversion topology
    //! changed in a way that may affect other pages' validation (mirrors
    //! ExperimentFtmwConfigPage::presetChanged).
    void presetChanged();

private:
    LifConfigWidget *p_widget;
};

#endif // EXPERIMENTLIFCONFIGPAGE_H
