#ifndef FTMWPROCESSINGWIDGET_H
#define FTMWPROCESSINGWIDGET_H

#include <QWidget>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QComboBox>

#include <src/data/storage/settingsstorage.h>
#include <src/data/experiment/experiment.h>
#include <src/data/analysis/ftworker.h>

namespace BC::Key {
static const QString ftmwProcWidget("ftmwProcessingWidget");
static const QString fidStart("startUs");
static const QString fidEnd("endUs");
static const QString zeroPad("zeroPad");
static const QString removeDC("removeDC");
static const QString ftUnits("ftUnits");
static const QString autoscaleIgnore("autoscaleIgnoreMHz");
static const QString ftWinf("windowFunction");
}

class FtmwProcessingWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit FtmwProcessingWidget(QWidget *parent = 0);
    FtWorker::FidProcessingSettings getSettings();

signals:
    void settingsUpdated(FtWorker::FidProcessingSettings);

public slots:
    void prepareForExperient(const Experiment &e);
    void readSettings();

private:
    QDoubleSpinBox *p_startBox, *p_endBox, *p_autoScaleIgnoreBox;
    QSpinBox *p_zeroPadBox;
    QCheckBox *p_removeDCBox;
    QComboBox *p_unitsBox, *p_winfBox;


};

#endif // FTMWPROCESSINGWIDGET_H
