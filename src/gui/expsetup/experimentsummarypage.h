#ifndef EXPERIMENTSUMMARYPAGE_H
#define EXPERIMENTSUMMARYPAGE_H

#include "experimentconfigpage.h"

class ExperimentSummaryWidget;

namespace BC::Key::WizSummary {
inline constexpr QLatin1StringView key{"SummaryPage"};
inline constexpr QLatin1StringView title{"Experiment Summary"};
}

class ExperimentSummaryPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    explicit ExperimentSummaryPage(Experiment *exp, QWidget *parent = nullptr);

private:
    ExperimentSummaryWidget *p_summaryWidget;

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;
};

#endif // EXPERIMENTSUMMARYPAGE_H
