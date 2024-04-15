#ifndef EXPERIMENTCONFIGPAGE_H
#define EXPERIMENTCONFIGPAGE_H

#include <QWidget>

#include <data/storage/settingsstorage.h>
#include <data/experiment/experiment.h>

class ExperimentConfigPage : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit ExperimentConfigPage(QString key, QString title, Experiment *exp, QWidget *parent = nullptr);

    const QString d_title;

protected:
    Experiment *p_exp;

signals:
    void warning(QString);
    void error(QString);

public slots:
    virtual void initialize()=0;
    virtual bool validate()=0;
    virtual void apply() =0;

};

#endif // EXPERIMENTCONFIGPAGE_H
