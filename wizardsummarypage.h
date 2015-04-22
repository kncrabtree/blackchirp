#ifndef WIZARDSUMMARYPAGE_H
#define WIZARDSUMMARYPAGE_H

#include <QWizardPage>

class QTableWidget;

class WizardSummaryPage : public QWizardPage
{
    Q_OBJECT
public:
    WizardSummaryPage(QWidget *parent = 0);
    ~WizardSummaryPage();

    // QWizardPage interface
    void initializePage();
    int nextId() const;

private:
    QTableWidget *p_tw;
};

#endif // WIZARDSUMMARYPAGE_H
