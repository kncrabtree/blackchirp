#ifndef WIZARDLIFCONFIGPAGE_H
#define WIZARDLIFCONFIGPAGE_H

#include <gui/wizard/experimentwizardpage.h>

class LifControlWidget;

namespace BC::Key::WizLif {
static const QString key{"WizardLifConfigPage"};
}

class WizardLifConfigPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardLifConfigPage(QWidget *parent = nullptr);
    ~WizardLifConfigPage() override;

    // QWizardPage interface
    void initializePage() override;
    bool validatePage() override;
    int nextId() const override;

    LifControlWidget *controlWidget() { return p_lifControl; }

private:
    LifControlWidget *p_lifControl;


};

#endif // WIZARDLIFCONFIGPAGE_H
