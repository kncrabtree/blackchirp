#ifndef WIZARDVALIDATIONPAGE_H
#define WIZARDVALIDATIONPAGE_H

#include <QWizardPage>

#include <QTableView>

#include "ioboardconfig.h"

class WizardValidationPage : public QWizardPage
{
public:
    explicit WizardValidationPage(QWidget *parent = nullptr);

private:
    QTableView *p_analogView, *p_digitalView, *p_validationView;
    IOBoardConfig d_config;


    // QWizardPage interface
public:
    int nextId() const;
};

#endif // WIZARDVALIDATIONPAGE_H
