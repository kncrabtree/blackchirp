#ifndef EXPERIMENTSETUPWIDGET_H
#define EXPERIMENTSETUPWIDGET_H

#include <QDialog>
#include <data/experiment/rfconfig.h>

class QTreeWidget;
class QStackedWidget;
class QTextEdit;
class ExperimentSummaryWidget;
class QPushButton;
class Experiment;
class ExperimentConfigPage;
class QTreeWidgetItem;

class ExperimentSetupDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ExperimentSetupDialog(Experiment *exp, const std::map<QString,QString> &hw, const QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks, const std::map<QString, QStringList> &valKeys, QWidget *parent = nullptr);

signals:

public slots:
    void pageChanged(QTreeWidgetItem *newItem, QTreeWidgetItem *prevItem);
    bool validateAll(bool apply=false);
    bool validate(QTreeWidgetItem *item, bool apply=false);
    void warning(const QString text);
    void error(const QString text);

private:

    struct PageData {
        int index;
        QString key;
        ExperimentConfigPage *page;
    };

    QTreeWidget *p_navTree;
    QStackedWidget *p_configWidget;
    ExperimentSummaryWidget *p_summaryWidget;
    QTextEdit *p_statusTextEdit;
    QPushButton *p_validateButton;

    Experiment *p_exp;

    std::map<QString,PageData> d_pages;


    // QDialog interface
public slots:
    void reject() override;
};

#endif // EXPERIMENTSETUPWIDGET_H
