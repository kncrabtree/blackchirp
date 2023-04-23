#ifndef WIZARDIOBOARDCONFIGPAGE_H
#define WIZARDIOBOARDCONFIGPAGE_H

#include <gui/wizard/experimentwizardpage.h>

#include <gui/widget/ioboardconfigwidget.h>

namespace BC::Key::WizIOB {
static const QString key{"WizardIOBoardConfigPage"};
}

class WizardIOBoardConfigPage : public ExperimentWizardPage
{
public:
    WizardIOBoardConfigPage(QWidget *parent = nullptr);

private:
    IOBoardConfigWidget *p_iobWidget;

    // QWizardPage interface
public:
    void initializePage() override;
    bool validatePage() override;
    int nextId() const override;
};

#endif // WIZARDIOBOARDCONFIGPAGE_H
