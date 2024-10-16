#ifndef EXPERIMENTSETUPWIDGET_H
#define EXPERIMENTSETUPWIDGET_H

#include <QDialog>
#include <QTreeWidgetItem>
#include <QStackedWidget>

#include <hardware/core/hardwareobject.h>
#include <data/experiment/rfconfig.h>

class QTreeWidget;
class QStackedWidget;
class QTextEdit;
class ExperimentSummaryWidget;
class QPushButton;
class Experiment;
class ExperimentConfigPage;
class QTreeWidgetItem;

#ifdef BC_LIF
class LifControlWidget;
#endif

class ExperimentSetupDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ExperimentSetupDialog(Experiment *exp, const std::map<QString,QString> &hw, const QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks, const std::map<QString, QStringList> &valKeys, QWidget *parent = nullptr);

#ifdef BC_LIF
    LifControlWidget *lifControlWidget();
#endif

signals:

public slots:
    void pageChanged(QTreeWidgetItem *newItem, QTreeWidgetItem *prevItem);
    bool validateAll(bool apply=true);
    bool validate(QTreeWidgetItem *item, bool apply=true);
    void enableChildren(QTreeWidgetItem *item, bool enable = true);
    void warning(const QString text);
    void error(const QString text);

private:

    struct PageData {
        int index;
        QString key;
        ExperimentConfigPage *page;
        bool enabled = true;
    };

    QTreeWidget *p_navTree;
    QStackedWidget *p_configWidget;
    ExperimentSummaryWidget *p_summaryWidget;
    QTextEdit *p_statusTextEdit;
    QPushButton *p_validateButton, *p_startButton;

    Experiment *p_exp;

    std::map<QString,PageData> d_pages;

    template<typename T> std::tuple<T*,QTreeWidgetItem*> addConfigPage(QString k, QTreeWidgetItem *parentItem, bool en = true)
    {
            SettingsStorage s(k,SettingsStorage::Hardware);
            auto title = s.get(BC::Key::HW::name,k);
            auto page = new T(p_exp);
            auto i = p_configWidget->addWidget(page);
            d_pages.insert({k,{i,k,page,en}});
            auto item = new QTreeWidgetItem(parentItem,{page->d_title});
            page->setEnabled(en);
            item->setDisabled(!en);
            item->setData(0,Qt::UserRole,k);

            return {page,item};
    }

    template<typename T> void addOptHwPages(QString hwKey, const std::map<QString, QString> &hw, QTreeWidgetItem *expTypeItem)
    {
        auto index = 0;
        auto it = hw.end();
        do
        {
            auto k = BC::Key::hwKey(hwKey,index);
            it = hw.find(k);
            if(it != hw.end())
            {
                SettingsStorage s(k,SettingsStorage::Hardware);
                auto title = s.get(BC::Key::HW::name,k);
                auto page = new T(k,title,p_exp);
                auto i = p_configWidget->addWidget(page);
                d_pages.insert({k,{i,k,page,true}});
                auto item = new QTreeWidgetItem(expTypeItem,{page->d_title});
                item->setData(0,Qt::UserRole,k);
                index++;
            }
        } while (it != hw.end());
    }


    // QDialog interface
public slots:
    void reject() override;
    void accept() override;
};

#endif // EXPERIMENTSETUPWIDGET_H
