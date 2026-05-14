#ifndef EXPERIMENTSETUPWIDGET_H
#define EXPERIMENTSETUPWIDGET_H

#include <QDialog>
#include <QTreeWidgetItem>
#include <QStackedWidget>

#include <hardware/core/hardwareobject.h>
#include <hardware/core/runtimehardwareconfig.h>
#include <data/experiment/rfconfig.h>
#include <data/settings/hardwarekeys.h>

class QTreeWidget;
class QStackedWidget;
class QTextEdit;
class QPushButton;
class Experiment;
class ExperimentConfigPage;
class QTreeWidgetItem;

class LifControlWidget;
class RfConfigWidget;

class ExperimentSetupDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ExperimentSetupDialog(Experiment *exp, const QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks, const std::map<QString, QStringList, std::less<>> &valKeys, QWidget *parent = nullptr);

    LifControlWidget *lifControlWidget();
    RfConfigWidget *rfConfigWidget();

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
    QTextEdit *p_statusTextEdit;
    QPushButton *p_validateButton, *p_startButton;

    Experiment *p_exp;

    std::map<QString,PageData,std::less<>> d_pages;

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

    template<typename T> void addOptHwPages(QString hwType, QTreeWidgetItem *expTypeItem)
    {
        auto hw = RuntimeHardwareConfig::constInstance().getCurrentHardware();
        for(const auto &[k, impl] : hw)
        {
            auto [type, label] = BC::Key::parseKey(k);
            if(type != hwType)
                continue;

            SettingsStorage s(k,SettingsStorage::Hardware);
            auto title = s.get(BC::Key::HW::name,k);
            auto page = new T(k,title,p_exp);
            auto i = p_configWidget->addWidget(page);
            d_pages.insert({k,{i,k,page,true}});
            auto item = new QTreeWidgetItem(expTypeItem,{page->d_title});
            item->setData(0,Qt::UserRole,k);
        }
    }


    // QDialog interface
public slots:
    void reject() override;
    void accept() override;

private slots:
    void onClockHwChanged();
};

#endif // EXPERIMENTSETUPWIDGET_H
