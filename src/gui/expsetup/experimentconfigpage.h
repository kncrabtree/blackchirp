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
    ///
    /// \brief Initialize settings that depend on values from another page
    ///
    /// Note that initialization a previous experiment or from settings should be done in the
    /// constructor. This function should only make changes that are necessary based on values
    /// selected on another page.
    ///
    virtual void initialize()=0;
    virtual bool validate()=0;
    virtual void apply() =0;

};

#endif // EXPERIMENTCONFIGPAGE_H
